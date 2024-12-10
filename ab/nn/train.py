import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import optuna
import numpy as np

# Reduce COCOS classes:
CLASS_LIST = [0, 1, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                5, 64, 20, 63, 7, 72]
NUM_CLASSES = len(CLASS_LIST)




class TrainModel:
    def __init__(self, model_source_package, train_dataset, test_dataset, lr: float, momentum: float, batch_size: int, manual_args=None, **kwargs):
        """
        Universal class for training CV, Text Generation and other models.
        :param model_source_package: Path to the model's package (string).
        :param train_dataset: dataset for training.
        :param test_dataset: dataset for testing.
        :param lr: Learning rate.
        :param momentum: Momentum for SGD.
        :param batch_size: Mini-batch size.
        :param manual_args: List of manual arguments for model initialization if args.py is not available.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
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
        if self.task=="image_segmentation":
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
            if self.task=="image_segmentation":
                mIoU = MetricMIoU(NUM_CLASSES)
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
                elif self.task=="image_segmentation": # Image Segmentation uses mIoU
                    outputs = self.forward_pass(inputs)
                    targets = labels
                    mIoU.update(outputs,targets)
                else:  # For other models
                    outputs = self.forward_pass(inputs)
                    targets = labels

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        if self.task=="image_segmentation":
            accuracy = mIoU.get()
        return accuracy

    def get_args(self):
        return self.args


class DatasetLoader:
    @staticmethod
    def load_dataset(loader_path, transform_path=None, **kwargs):
        """
        Dynamically load dataset and transformation based on the provided paths.
        :param loader_path: Path to the dataset loader (e.g., 'ab.loader.cifar10.loader').
        :param transform_path: Path to the dataset transformation (e.g., 'ab.transform.normalization_cifar10.transform').
        :param kwargs: Additional parameters for the loader and transform.
        :return: Train and test datasets.
        """
        # Dynamically load the transform function if provided
        transform = None
        if transform_path:
            transform_module, transform_func = transform_path.rsplit('.', 1)
            transform = getattr(__import__(transform_module, fromlist=[transform_func]), transform_func)(**kwargs)

        # Dynamically load the loader function
        loader_module, loader_func = loader_path.rsplit('.', 1)
        loader = getattr(__import__(loader_module, fromlist=[loader_func]), loader_func)

        # Call the loader function with the dynamically loaded transform
        return loader(transform=transform, **kwargs)

def parse_model_config(directory_name):
    """
    Parse the model configuration to extract task, dataset, and optional transformation.
    :param directory_name: Name of the directory (e.g., "image_classification-cifar10-normalization_cifar10-AlexNet").
    :return: Parsed task, dataset name, and transformation name (if any).
    """
    parts = directory_name.split('-')
    task = parts[0]
    dataset_name = parts[1]
    transform_name = parts[2] if len(parts) > 2 else None
    model_name = parts[3]
    return task, dataset_name, transform_name, model_name


def ensure_directory_exists(model_dir):
    """
    Ensures that the directory for the given path exists.
    :param model_dir: Path to the target directory or file.
    :return: Creates the directory if it does not exist.
    """
    directory = os.path.dirname(model_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def save_results(config_model_name, study, n_epochs, n_optuna_trials):
    """
    Save Optuna study results for a given model in JSON-format.
    :param config_model_name: Config (Task, Dataset, Normalization) and Model name.
    :param study: Optuna study object.
    :param n_epochs: Number of epochs.
    :param n_optuna_trials: Number of trials.
    """
    best_trial = {
        "accuracy": float(study.best_trial.value),
        "batch_size": int(study.best_trial.params["batch_size"]),
        "lr": float(study.best_trial.params["lr"]),
        "momentum": float(study.best_trial.params["momentum"])
    }

    model_dir = f"stat/{config_model_name}/{n_epochs}/"
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

    print(f"Trials for {config_model_name} saved at {model_dir}")


def main(config='all', n_epochs=1, n_optuna_trials=100, dataset_params=None, manual_args=None):
    """
    Main function for training models using Optuna optimization.
    :param config: Configuration specifying the models to train.
    :param n_epochs: Number of epochs for training.
    :param n_optuna_trials: Number of Optuna trials.
    :param dataset_params: Parameters specific to dataset loading.
    :param manual_args: Manually provided arguments for model initialization.
    """
    if dataset_params is None:
        dataset_params = {}

    # Determine configurations based on the provided config
    if config == 'all':
        # Collect all configurations from the 'stat' directory
        sub_configs = [
            sub_config
            for sub_config in os.listdir("stat")
            if os.path.isdir(os.path.join("stat", sub_config))
        ]
    elif len(config.split('-')) == 3:  # Partial configuration, e.g., 'image_classification-cifar10-normalization_cifar10'
        # Collect models matching the given configuration prefix
        config_prefix = config + '-'
        sub_configs = [
            sub_config
            for sub_config in os.listdir("stat")
            if sub_config.startswith(config_prefix) and os.path.isdir(os.path.join("stat", sub_config))
        ]
    else:  # Specific configuration, e.g., 'image_classification-cifar10-normalization_cifar10-AlexNet'
        sub_configs = [config]

    print("Configurations found for training:")
    for idx, sub_config in enumerate(sub_configs, start=1):
        print(f"{idx}. {sub_config}")

    for config_name in sub_configs:
        sub_configs = [config_name]

        for sub_config in sub_configs:
            try:
                task, dataset_name, transform_name, model_name = parse_model_config(sub_config)
            except (ValueError, IndexError) as e:
                print(f"Skipping config '{sub_config}': failed to parse. Error: {e}")
                continue

<<<<<<< HEAD:train.py
        # apply class filter with cocos
        if task=="image_segmentation":
            dataset_params['class_list']=CLASS_LIST
            dataset_params['path']="./cocos"
        # Load dataset
        try:
            train_set, test_set = DatasetLoader.load_dataset(loader_path, transform_path, **dataset_params)
        except Exception as e:
            print(f"Skipping model '{model_name}': failed to load dataset. Error: {e}")
            continue

        # Configure Optuna for the current model
        def objective(trial):
            if task == 'image_segmentation':
                lr = trial.suggest_float('lr', 1e-4, 1e-2, log=False)
                momentum = trial.suggest_float('momentum', 0.8, 0.99, log=True)
                batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
            else:
                lr = trial.suggest_float('lr', 1e-4, 1, log=False)
                momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
                batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
=======
            print(f"\nStarting training for model: {model_name}, Task: {task}, Dataset: {dataset_name}, Transform: {transform_name}")

            # Paths for loader and transform
            loader_path = f"loader.{dataset_name}.loader"
            transform_path = f"transform.{transform_name}.transform" if transform_name else None
>>>>>>> 646ad0bf5992b8ef55439b9b691760e06b2af02d:ab/nn/train.py

            # Load dataset
            try:
                train_set, test_set = DatasetLoader.load_dataset(loader_path, transform_path, **dataset_params)
            except Exception as e:
                print(f"Skipping model '{model_name}': failed to load dataset. Error: {e}")
                continue

            # Configure Optuna for the current model
            def objective(trial):
                lr = trial.suggest_float('lr', 1e-4, 1, log=False)
                momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
                batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])

                print(f"Initialize training with lr = {lr}, momentum = {momentum}, batch_size = {batch_size}")

                if task == 'image_classification':
                    trainer = TrainModel(
                        model_source_package=f"dataset.{model_name}",
                        train_dataset=train_set,
                        test_dataset=test_set,
                        lr=lr,
                        momentum=momentum,
                        batch_size=batch_size,
                        manual_args=manual_args.get(model_name) if manual_args else None
                    )
                elif task == 'text_generation':
                    # Dynamically import RNN or LSTM model
                    if model_name.lower() == 'rnn':
                        from dataset.RNN.code import Net as RNNNet
                        model = RNNNet(1, 256, len(train_set.chars), batch_size)
                    elif model_name.lower() == 'lstm':
                        from dataset.LSTM.code import Net as LSTMNet
                        model = LSTMNet(1, 256, len(train_set.chars), batch_size, num_layers=2)
                    else:
                        raise ValueError(f"Unsupported text generation model: {model_name}")

                    trainer = TrainModel(
                        model_source_package=f"dataset.{model_name}",
                        train_dataset=train_set,
                        test_dataset=test_set,
                        lr=lr,
                        momentum=momentum,
                        batch_size=batch_size,
                        manual_args=manual_args.get(model_name) if manual_args else None
                    )
                else:
                    raise ValueError(f"Unsupported task type: {task}")

<<<<<<< HEAD:train.py
                trainer = TrainModel(
                    model_source_package=f"Dataset.{model_name}",
                    train_dataset=train_set,
                    test_dataset=test_set,
                    lr=lr,
                    momentum=momentum,
                    batch_size=batch_size,
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
=======
                return trainer.evaluate(n_epochs)
>>>>>>> 646ad0bf5992b8ef55439b9b691760e06b2af02d:ab/nn/train.py

            # Launch Optuna for the current model
            study_name = f"{model_name}_study"
            study = optuna.create_study(study_name=study_name, direction='maximize')
            study.optimize(objective, n_trials=n_optuna_trials)

            # Save results
            save_results(sub_config, study, n_epochs, n_optuna_trials)



if __name__ == "__main__":
    # Training parameters
    # config ='all' # For all configurations
    config = 'image_classification-cifar10-normalization_cifar10' # For a particular configuration for all models
    # config = 'image_classification-cifar10-normalization_cifar10-ResNet' # For a particular configuration and model
    n_model_epochs = 2
    n_optuna_trials = 100

    # Run training with Optuna
    main(config, n_model_epochs, n_optuna_trials)