import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from datasets import load_dataset
from tqdm import tqdm
import optuna
import numpy as np

# Reduce COCOS classes:
CLASS_LIST = [0, 1, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                5, 64, 20, 63, 7, 72]
NUM_CLASSES = len(CLASS_LIST)




class TrainModel:
    def __init__(self, model_source_package, train_dataset, test_dataset, lr: float, momentum: float, batch_size: int, task_type='image_classification', manual_args=None):
        """
        Universal class for training CV and Text Generation models.
        :param model_source_package: Path to the model's package (string).
        :param train_dataset: Dataset for training.
        :param test_dataset: Dataset for testing.
        :param lr: Learning rate.
        :param momentum: Momentum for SGD.
        :param batch_size: Mini-batch size.
        :param task_type: Task type.
        :param manual_args: List of manual arguments for model initialization if args.py is not available.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr
        self.momentum = momentum
        self.batch_size = max(2, batch_size)
        self.task_type = task_type
        self.args = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model
        if isinstance(model_source_package, str):
            # Handle special case for InceptionV3
            if "InceptionV3" in model_source_package:
                from torchvision.models import inception_v3
                self.model = inception_v3(aux_logits=False)  # Disable aux_logits for InceptionV3
                print("Loaded InceptionV3 with aux_logits disabled.")
            else:
                # Load the model class
                model_class = getattr(
                    __import__(model_source_package + ".code", fromlist=["Net"]),
                    "Net"
                )

                # Try loading arguments from args.py
                try:
                    self.args = getattr(
                        __import__(model_source_package + ".args", fromlist=["args"]),
                        "args"
                    )
                except ImportError:
                    if manual_args:
                        self.args = manual_args
                        print(f"No args.py found. Using manual_args: {self.args}")
                    else:
                        raise ValueError(f"Arguments required for {model_class.__name__} are missing. Please provide them manually via manual_args.")

                # Initialize the model with arguments
                self.model = model_class(*self.args)

        elif isinstance(model_source_package, torch.nn.Module):
            # If a pre-initialized model is passed
            self.model = model_source_package
        else:
            raise ValueError("model_source_package must be a string (path to the model) or an instance of torch.nn.Module.")

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
        if task=="image_segmentation":
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
            if task=="image_segmentation":
                matric = SegmentationMetric(NUM_CLASSES)
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
                elif task=="image_segmentation":
                    outputs = self.forward_pass(inputs)
                    targets = labels
                    matric.update(outputs,targets)
                else:  # For other models
                    outputs = self.forward_pass(inputs)
                    targets = labels

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        if task=="image_segmentation":
            _,accuracy = matric.get()
        return accuracy

    def get_args(self):
        return self.args


class DatasetLoader:
    _handlers = {}

    @staticmethod
    def register_handler(task, dataset_name):
        """
        Decorator for registering dataset handlers for a specific task and dataset name.
        """
        def decorator(handler):
            DatasetLoader._handlers[(task, dataset_name)] = handler
            return handler
        return decorator

    @staticmethod
    def load_dataset(task, dataset_name, **kwargs):
        """
        Load dataset based on task and dataset name.
        :param task: Task type (e.g., 'image_classification', 'text_generation').
        :param dataset_name: Dataset name (e.g., 'CIFAR10', 'Wikitext').
        :param kwargs: Additional parameters for the dataset loader.
        :return: Train and test datasets or other necessary objects.
        """
        handler = DatasetLoader._handlers.get((task, dataset_name))
        if handler is None:
            raise ValueError(f"No handler registered for task '{task}' and dataset '{dataset_name}'")
        return handler(**kwargs)

@DatasetLoader.register_handler('image_classification', 'CIFAR10')
def load_cifar10(transform=None, download=False):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    train_set = torchvision.datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    test_set = torchvision.datasets.CIFAR10(root='data', train=False, transform=transform, download=True)
    return train_set, test_set

@DatasetLoader.register_handler('text_generation', 'Wikitext')
def load_wikitext(dataset_name="Salesforce/wikitext", config="wikitext-2-raw-v1", seq_length=100):
    dataset = load_dataset(dataset_name, config)
    data = "\n".join(dataset["train"]["text"]).lower()
    text_dataset = TextDatasetPreparation(data, seq_length)
    return text_dataset, text_dataset

@DatasetLoader.register_handler('image_segmentation', 'COCOSeg')
def load_cocos(path="./cocos",resize=(128,128), **kwargs):
    from DataLoader.CocoDataset import COCOSegDataset
    train_set = COCOSegDataset(root=path,spilt="train",enable_list=True,cat_list=CLASS_LIST,resize=resize,preprocess=True)
    val_set = COCOSegDataset(root=path,spilt="val",enable_list=True,cat_list=CLASS_LIST,resize=resize,preprocess=True)
    return train_set, val_set

class TextDatasetPreparation(Dataset):
    """
    A dataset class for preparing character-level text data for tasks like text generation.
    :param text_data: Input text as a single string.
    :param seq_length: Length of each input sequence (default: 25).
    :return: Pairs of input and target sequences as tensors.
    """
    def __init__(self, text_data: str, seq_length: int = 25):
        self.chars = sorted(list(set(text_data)))
        self.data_size, self.vocab_size = len(text_data), len(self.chars)
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length
        self.X = self.string_to_vector(text_data)

    def __len__(self) -> int:
        return int(len(self.X) / self.seq_length - 1)

    def __getitem__(self, index) -> tuple:
        start_idx = index * self.seq_length
        end_idx = (index + 1) * self.seq_length
        X = torch.tensor(self.X[start_idx:end_idx]).float()
        y = torch.tensor(self.X[start_idx + 1:end_idx + 1]).float()
        return X, y

    def string_to_vector(self, name: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in name]

    def vector_to_string(self, vector: list[int]) -> str:
        return ''.join([self.idx_to_char[i] for i in vector])


def ensure_directory_exists(model_dir):
    """
    Ensures that the directory for the given path exists.
    :param model_dir: Path to the target directory or file.
    :return: Creates the directory if it does not exist.
    """
    directory = os.path.dirname(model_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
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

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

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

def save_results(model_name, study, task, n_epochs, n_optuna_trials):
    """
    Save Optuna study results for a given model.
    :param model_name: Model name.
    :param study: Optuna study object.
    :param task: Task type.
    :param n_epochs: Number of epochs.
    :param n_optuna_trials: Number of trials.
    """
    best_trial = {
        "accuracy": float(study.best_trial.value),
        "batch_size": int(study.best_trial.params["batch_size"]),
        "lr": float(study.best_trial.params["lr"]),
        "momentum": float(study.best_trial.params["momentum"])
    }

    model_dir = f"./Dataset/{model_name}/{task}/{n_epochs}/"
    ensure_directory_exists(model_dir)

    # Save best_trial.json
    with open(f"{model_dir}/best_trial.json", "w") as f:
        json.dump(best_trial, f, indent=4)

    # Save all trials as optuna_<n_optuna_trials>.json
    trials_df = study.trials_dataframe()
    filtered_trials = trials_df[["value", "params_batch_size", "params_lr", "params_momentum"]]

    filtered_trials = filtered_trials.rename(columns={
        "value": "accuracy",
        "params_batch_size": "batch_size",
        "params_lr": "lr",
        "params_momentum": "momentum"
    })

    filtered_trials = filtered_trials.astype({
        "accuracy": float,
        "batch_size": int,
        "lr": float,
        "momentum": float
    })

    trials_dict = filtered_trials.to_dict(orient='records')
    with open(f"{model_dir}/optuna_{n_optuna_trials}.json", "w") as f:
        json.dump(trials_dict, f, indent=4)

    print(f"Trials for {model_name} saved at {model_dir}")


def main(task, model_names, n_epochs, n_optuna_trials=100, dataset_params=None, manual_args=None):
    """
    Main function for training models using Optuna optimization.
    :param task: Task type ('image_classification' or 'text_generation').
    :param model_names: List of model names or 'all' to include all models in the directory.
    :param n_epochs: Number of epochs for training.
    :param n_optuna_trials: Number of Optuna trials.
    :param dataset_params: Parameters specific to dataset loading.
    """
    if dataset_params is None:
        dataset_params = {}

    # if all models
    if model_names == "all":
        model_names = [
            model for model in os.listdir("./Dataset")
            if os.path.isdir(os.path.join("./Dataset", model))
        ]

    # If the specified models are selected
    for model_name in model_names:
        print(f"\nStarting training for model: {model_name}")

        # Configure Optuna for the current model
        def objective(trial):
            if task == 'image_segmentation':
                lr = trial.suggest_float('lr', 1e-5, 1e-3, log=False)
                momentum = trial.suggest_float('momentum', 0.8, 0.99, log=True)
                batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
            else:
                lr = trial.suggest_float('lr', 1e-4, 1, log=False)
                momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
                batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])

            print(f"Initializing ModelEvaluator with lr = {lr}, momentum = {momentum}, batch_size = {batch_size}")

            if task == 'image_classification':
                trainer = TrainModel(
                    model_source_package=f"Dataset.{model_name}",
                    train_dataset=train_set,
                    test_dataset=test_set,
                    lr=lr,
                    momentum=momentum,
                    batch_size=batch_size,
                    task_type=task,
                    manual_args=manual_args.get(model_name) if manual_args else None
                )
            elif task == 'text_generation':
                # Dynamically import RNN or LSTM model
                if model_name.lower() == 'rnn':
                    from Dataset.RNN.code import Net as RNNNet
                    model = RNNNet(1, 256, len(train_set.chars), batch_size)
                elif model_name.lower() == 'lstm':
                    from Dataset.LSTM.code import Net as LSTMNet
                    model = LSTMNet(1, 256, len(train_set.chars), batch_size, num_layers=2)
                else:
                    raise ValueError(f"Unsupported text generation model: {model_name}")

                trainer = TrainModel(
                    model_source_package=f"Dataset.{model_name}",
                    train_dataset=train_set,
                    test_dataset=test_set,
                    lr=lr,
                    momentum=momentum,
                    batch_size=batch_size,
                    task_type=task,
                    manual_args=manual_args.get(model_name) if manual_args else None
                )
            elif task == 'image_segmentation':
                trainer = TrainModel(
                    model_source_package=f"Dataset.{model_name}",
                    train_dataset=train_set,
                    test_dataset=test_set,
                    lr=lr,
                    momentum=momentum,
                    batch_size=batch_size,
                    task_type=task,
                    manual_args=manual_args.get(model_name) if manual_args else None
                )
            else:
                raise ValueError(f"Unsupported task type: {task}")

            return trainer.evaluate(n_epochs)

        # Launch Optuna for the current model
        study_name = f"{model_name}_study"
        study = optuna.create_study(study_name=study_name, direction='maximize')
        study.optimize(objective, n_trials=n_optuna_trials)

        # Save results
        save_results(model_name, study, task, n_epochs, n_optuna_trials)


if __name__ == "__main__":
    # Training parameters
    task = 'image_classification'  # or 'text_generation'
    model_names = "all"  # Iterating over all models in the ./Dataset directory
    # model_names = ["ResNet", "DenseNet"] # Or select the only models you need
    dataset_name = 'CIFAR10'  # Specify the dataset to use
    n_model_epochs = 1
    n_optuna_trials = 2

    # Dataset parameters for image classification task
    dataset_params = {
        'transform': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        'download': True
    }

    # Dataset parameters for text generation task
    # dataset_params = {
    #     'dataset_name': "Salesforce/wikitext",
    #     'config': "wikitext-2-raw-v1",
    #     'seq_length': 100
    # }

    # Load the dataset
    train_set, test_set = DatasetLoader.load_dataset(task, dataset_name, **dataset_params)

    # Run training with Optuna
    main(task, model_names, n_model_epochs, n_optuna_trials, {'train_set': train_set, 'test_set': test_set})

