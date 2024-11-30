import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from tqdm import tqdm
import optuna


class TrainModel:
    def __init__(self, model, train_dataset, test_dataset, lr, momentum, batch_size, task_type='image_classification'):
        """
        Universal class for training CV and Text Generation models.
        :param model: PyTorch model (CNN, LSTM, RNN, etc.).
        :param train_dataset: Dataset for training.
        :param test_dataset: Dataset for testing.
        :param lr: Learning rate.
        :param momentum: Momentum for SGD.
        :param batch_size: Mini-batch size.
        :param task_type: Task type.
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.task_type = task_type

    def evaluate(self, num_epochs):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

        self.model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # print(f"Training {self.model.__class__.__name__} on {self.device}")

        for _ in tqdm(range(num_epochs), desc="Training"):
            self.model.train()
            for data in train_loader:
                X, Y = data
                if X.size(0) != self.batch_size:
                    continue

                if hasattr(self.model, "init_zero_hidden"):
                    hidden = self.model.init_zero_hidden(self.batch_size)
                    if isinstance(hidden, tuple):  # Для LSTM
                        hidden = tuple(h.to(self.device) for h in hidden)
                    else:  # Для RNN
                        hidden = hidden.to(self.device)

                X, Y = X.to(self.device), Y.to(self.device)
                optimizer.zero_grad()

                if hasattr(self.model, "init_zero_hidden"):  # RNN/LSTM
                    outputs = []
                    targets = []
                    for c in range(X.size(1)):
                        step_input = X[:, c].unsqueeze(1)
                        out, hidden = self.model(step_input, hidden)
                        outputs.append(out)
                        targets.append(Y[:, c].long())

                    outputs = torch.cat(outputs, dim=0)
                    targets = torch.cat(targets, dim=0)
                else:  # Others
                    outputs = self.model(X)
                    targets = Y

                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                optimizer.step()

        # Model evaluation
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                X, Y = data
                X, Y = X.to(self.device), Y.to(self.device)

                if hasattr(self.model, "init_zero_hidden"):  # RNN/LSTM
                    outputs = []
                    targets = []
                    for c in range(X.size(1)):
                        step_input = X[:, c].unsqueeze(1)
                        out, hidden = self.model(step_input, hidden)
                        outputs.append(out)
                        targets.append(Y[:, c].long())

                    outputs = torch.cat(outputs, dim=0)
                    targets = torch.cat(targets, dim=0)
                else:  # Others
                    outputs = self.model(X)
                    targets = Y

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        # print(f"Accuracy: {accuracy:.4f}")
        return accuracy


class DatasetLoader:
    @staticmethod
    def load_dataset(task, **kwargs):
        """
        Universal method for loading datasets for various tasks.
        :param task: Task type ('image_classification', 'text_generation', etc.).
        :param kwargs: Additional parameters specific to each task.
        :return: A tuple containing train_dataset, test_dataset, and other necessary data.
        """
        if task == 'image_classification':
            transform = kwargs.get('transform', transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]))
            train_set = torchvision.datasets.CIFAR10(root='data', train=True, transform=transform, download=False)
            test_set = torchvision.datasets.CIFAR10(root='data', train=False, transform=transform, download=False)
            return train_set, test_set

        elif task == 'text_generation':
            dataset = load_dataset(kwargs.get('dataset_name', "Salesforce/wikitext"), kwargs.get('config', "wikitext-2-raw-v1"))
            data = "\n".join(dataset["train"]["text"]).lower()
            seq_length = kwargs.get('seq_length', 100)
            text_dataset = TextDatasetPreparation(data, seq_length)
            return text_dataset, text_dataset  # The same dataset is used for both training and testing

        else:
            raise ValueError(f"Unsupported task type: {task}")


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


def main(task, model_name, n_epochs, n_optuna_trials=100, dataset_params=None):
    if dataset_params is None:
        dataset_params = {}

    # Loading datasets
    train_set, test_set = DatasetLoader.load_dataset(task, **dataset_params)

    # Configure Optuna
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1, log=False)
        momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])

        print(f"Initializing ModelEvaluator with lr = {lr}, momentum = {momentum}, batch_size = {batch_size}")

        if task == 'image_classification':
            from Dataset.DenseNet.code import Net as CVNet
            model = CVNet()
            trainer = TrainModel(
                model=model,
                train_dataset=train_set,
                test_dataset=test_set,
                lr=lr,
                momentum=momentum,
                batch_size=batch_size,
                task_type=task
            )
        elif task == 'text_generation':
            if model_name == 'RNN':
                from Dataset.RNN.code import Net as RNNNet
                model = RNNNet(1, 256, len(train_set.chars), batch_size)
            elif model_name == 'LSTM':
                from Dataset.LSTM.code import Net as LSTMNet
                model = LSTMNet(1, 256, len(train_set.chars), batch_size, num_layers=2)
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            trainer = TrainModel(
                model=model,
                train_dataset=train_set,
                test_dataset=test_set,
                lr=lr,
                momentum=momentum,
                batch_size=batch_size,
                task_type=task
            )
        else:
            raise ValueError(f"Unsupported task type: {task}")

        return trainer.evaluate(n_epochs)

    # Launch Optuna
    study_name = f"{model_name}_study"
    study = optuna.create_study(study_name=study_name, direction='maximize')
    study.optimize(objective, n_trials=n_optuna_trials)

    # Save the best result
    best_trial = {
        "accuracy": study.best_trial.value,
        "batch_size": study.best_trial.params["batch_size"],
        "lr": study.best_trial.params["lr"],
        "momentum": study.best_trial.params["momentum"]
    }

    model_dir = f"./Dataset/{model_name}/{task}/{n_epochs}/"
    ensure_directory_exists(model_dir)

    with open(f"{model_dir}/best_trial.json", "w") as f:
        json.dump(best_trial, f, indent=4)

    # Save all results
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

    print(f"Trials for {model_name} saved")



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training parameters
    task = 'image_classification'
    # task = 'text_generation'
    model_name = "MobileNetV2"
    num_epochs = 1
    n_optuna_trials = 1

    # Configure dataset parameters
    if task == 'image_classification':
        dataset_params = {
            'transform': transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        }
    elif task == 'text_generation':
        dataset_params = {
            'dataset_name': "Salesforce/wikitext",
            'config': "wikitext-2-raw-v1",
            'seq_length': 100
        }
    else:
        raise ValueError(f"Unsupported task: {task}")

    main(task, model_name, num_epochs, n_optuna_trials, dataset_params)