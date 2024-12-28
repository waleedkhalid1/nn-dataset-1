import importlib

import torch
from tqdm import tqdm

from ab.nn.util.Util import nn_mod, get_attr


class Train:
    def __init__(self, model_source_package, task_type, train_dataset, test_dataset, metric, output_dimension: int,
                 lr: float, momentum: float, batch_size: int):
        """
        Universal class for training CV, Text Generation and other models.
        :param model_source_package: Path to the model's package as a string (e.g., 'ab.nn.dataset.ResNet').
        :param task_type: e.g., 'img_segmentation' to specify the task type.
        :param train_dataset: The dataset used for training the model (torch.utils.data.Dataset).
        :param test_dataset: The dataset used for evaluating/testing the model (torch.utils.data.Dataset).
        :param metric: The name of the evaluation metric (e.g., 'acc', 'iou').
        :param output_dimension: The output dimension of the model (number of classes for classification tasks).
        :param lr: Learning rate value for the optimizer.
        :param momentum: Momentum value for the SGD optimizer.
        :param batch_size: Batch size used for both training and evaluation.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.output_dimension = output_dimension
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.task = task_type

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.metric_name = metric
        self.metric_function = self.load_metric_function(metric)

        # Load model
        self.model = get_attr(model_source_package, "Net")()
        self.model.to(self.device)

    def load_metric_function(self, metric_name):
        """
        Dynamically load the metric function or class based on the metric_name.
        :param metric_name: Name of the metric (e.g., 'accuracy', 'iou').
        :return: Loaded metric function or initialized class.
        """
        try:
            module = importlib.import_module(nn_mod('metric', metric_name))
            if metric_name.lower() == "iou":
                return module.MIoU(self.output_dimension)
            else:
                return getattr(module, "compute")
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist.") \
                from e


    def evaluate(self, num_epochs):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        self.model.train_setup(self.device, {'lr': self.lr, 'momentum': self.momentum})
        res = None
        # --- Training --- #
        for epoch in range(num_epochs):
            print(f"epoch {epoch}", flush=True)
            self.model.train()
            it = tqdm(train_loader)
            for inputs, labels in it:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.model.learn(inputs, labels)
            res = self.eval(test_loader)
        return res

    def eval(self, test_loader):
        """ Evaluation """
        self.model.eval()
        total_correct, total_samples = 0, 0
        if hasattr(self.metric_function, "reset"):  # Check for reset()
            self.metric_function.reset()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                if hasattr(self.metric_function, "update"):  # For mIoU
                    self.metric_function.update(outputs, labels)
                else:  # For accuracy and others
                    correct, total = self.metric_function(outputs, labels)
                    total_correct += correct
                    total_samples += total
        # Metric result
        if hasattr(self.metric_function, "get"):
            result = self.metric_function.get()
        else:
            result = total_correct / total_samples
        return result