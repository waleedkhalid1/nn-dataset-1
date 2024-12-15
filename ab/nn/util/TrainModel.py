import torch
import torch.nn as nn
from tqdm import tqdm


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


class MetricMIoU(object):
    """Computes mIoU metric scores
    """

    def __init__(self, nclass):
        super(MetricMIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        inter, union = batch_intersection_union(preds, labels, self.nclass)

        if self.total_inter.device != inter.device:
            self.total_inter = self.total_inter.to(inter.device)
            self.total_union = self.total_union.to(union.device)
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        mIoU
        """
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)

class TrainModel:
    def __init__(self, model_source_package, train_dataset, test_dataset, output_dimension: int, lr: float, momentum: float, batch_size: int,
                 manual_args: list = None, **kwargs):
        """
        Universal class for training CV, Text Generation and other models.
        :param model_source_package: Path to the model's package (string).
        :param train_dataset: dataset for training.
        :param test_dataset: dataset for testing.
        :param lr: Learning rate.
        :param momentum: Momentum for SGD.
        :param batch_size: Mini-batch size.
        :param manual_args: List of manual arguments (if varies from original arguments).
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_dimension = output_dimension
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr
        self.momentum = momentum
        self.batch_size = max(2, batch_size)
        self.args = None
        self.task = kwargs.get('task_type')

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model
        if isinstance(model_source_package, str):
            # Load the model class
            model_class = getattr(
                __import__(model_source_package, fromlist=["Net"]),
                "Net"
            )
            # Try loading arguments from args.py
            if manual_args is not None:
                self.args = manual_args
            else:
                self.args = []

            # Initialize the model with arguments
            try:
                self.model = model_class(*self.args)
            except TypeError:
                raise ValueError(f"Arguments required for {model_class.__name__} are missing. "
                                 f"Please provide them manually via manual_args or add default arguments "
                                 f"to the model code.")

        elif isinstance(model_source_package, torch.nn.Module):
            # If a pre-initialized model is passed
            self.model = model_source_package
        else:
            raise ValueError(
                "model_source_package must be a string (path to the model) or an instance of torch.nn.Module.")

        self.model.to(self.device)

    def forward_pass(self, inputs):
        """
        Runs a forward pass through the model and removes auxiliary outputs if present.
        """
        outputs = self.model(inputs)
        if isinstance(outputs, (tuple, list)):  # For models like InceptionV3 that may have multiple outputs
            outputs = outputs[0]  # Keep only the main output
        return outputs

    def evaluate(self, num_epochs):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )
        if self.task == "img_segmentation":
            params_list = []
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
            if hasattr(self.model, 'backbone'):
                params_list.append({'params': self.model.backbone.parameters(), 'lr': self.lr})
            if hasattr(self.model, 'exclusive'):
                for module in self.model.exclusive:
                    params_list.append({'params': getattr(self.model, module).parameters(), 'lr': self.lr * 10})
            optimizer = torch.optim.SGD(params_list, lr=self.lr, momentum=self.momentum)
        else:
            criterion = torch.nn.CrossEntropyLoss().to(self.device)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Training loop
        for _ in tqdm(range(num_epochs), desc="Training"):
            self.model.train()
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                if hasattr(self.model, "init_zero_hidden"):  # For RNN/LSTM
                    hidden = self.model.init_zero_hidden(self.batch_size)
                    if isinstance(hidden, tuple):  # For LSTM
                        hidden = tuple(h.to(self.device) for h in hidden)
                    else:  # For RNN
                        hidden = hidden.to(self.device)

                    outputs = []
                    targets = []
                    for c in range(inputs.size(1)):  # Iterate over sequence length
                        step_input = inputs[:, c].unsqueeze(1)  # [batch_size, 1, input_size]
                        out, hidden = self.model(step_input, hidden)
                        outputs.append(out)
                        targets.append(labels[:, c].long())

                    outputs = torch.cat(outputs, dim=0)  # [batch_size * seq_len, output_size]
                    targets = torch.cat(targets, dim=0)  # [batch_size * seq_len]
                else:  # For other models
                    outputs = self.forward_pass(inputs)
                    targets = labels

                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                optimizer.step()

        # Evaluation loop
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            if self.task == "img_segmentation":
                mIoU = MetricMIoU(self.output_dimension)
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if hasattr(self.model, "init_zero_hidden"):  # For RNN/LSTM
                    hidden = self.model.init_zero_hidden(self.batch_size)
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.to(self.device) for h in hidden)
                    else:
                        hidden = hidden.to(self.device)

                    outputs = []
                    targets = []
                    for c in range(inputs.size(1)):
                        step_input = inputs[:, c].unsqueeze(1)
                        out, hidden = self.model(step_input, hidden)
                        outputs.append(out)
                        targets.append(labels[:, c].long())

                    outputs = torch.cat(outputs, dim=0)
                    targets = torch.cat(targets, dim=0)
                elif self.task == "img_segmentation":  # Image Segmentation uses mIoU
                    outputs = self.forward_pass(inputs)
                    targets = labels
                    mIoU.update(outputs, targets)
                else:  # For other models
                    outputs = self.forward_pass(inputs)
                    targets = labels

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        metric = correct / total
        if self.task == "img_segmentation":
            metric = mIoU.get()
        return metric

    def get_args(self):
        return self.args
