import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import optuna
import os
import json
import time
from tqdm import tqdm
from Dataset.LSTM.code import Net, DatasetPreparation




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")


def ensure_directory_exists(model_dir):
    directory = os.path.dirname(model_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)



def train(model: Net, data: DataLoader, number_of_epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
    train_losses = {}
    model.to(device)

    model.train()
    # print("=> Starting training")
    total_start_time = time.time()
    for epoch in tqdm(range(number_of_epochs), desc="Training Progress", unit="epoch"):
        epoch_start_time = time.time()
        total_correct = 0
        total_count = 0
        # epoch_losses = list()
        for X, Y in data:
            # skip batch if it doesn't match with the batch_size
            if X.shape[0] != model.batch_size:
                continue
            # Инициализируем скрытое состояние
            hidden = model.init_zero_hidden(batch_size=model.batch_size)
            hidden = tuple(h.to(device) for h in hidden)  # Переносим каждый тензор в кортеже на устройство

            # send tensors to device
            X, Y = X.to(device), Y.to(device)

            # 2. clear gradients
            model.zero_grad()

            loss = 0
            correct = 0
            for c in range(X.shape[1]):
                out, hidden = model(X[:, c].reshape(X.shape[0], 1, -1), hidden)
                l = loss_fn(out[:, -1, :], Y[:, c].long())
                loss += l
                pred = out[:, -1, :].argmax(dim=1)
                correct += (pred == Y[:, c].long()).sum().item()

            # 4. Compute gradients gradients
            loss.backward()

            # 5. Adjust learnable parameters
            # clip as well to avoid vanishing and exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            total_correct += correct
            total_count += X.shape[1] * X.shape[0]

        accuracy = total_correct / total_count
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch + 1} finished in: {epoch_time}')
        # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')

    total_time = time.time() - total_start_time
    print(f'Trial finished in: {total_time}')

    return accuracy



def main(model: str, number_of_epochs: int):
    """The main function for loading data, setting up the model and training"""

    # Set dataset path
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    data = "\n".join(dataset["train"]["text"])
    data = data.lower()

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1, log=False)
        momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
        batch_size = trial.suggest_categorical('batch_size', [4, 5, 8, 16, 32, 64])

        print(f"Initializing TGModelEvaluator with lr = {lr}, momentum = {momentum}, batch_size = {batch_size}")

        # Data size variables
        seq_length = 100
        hidden_size = 256

        text_dataset = DatasetPreparation(data, seq_length=seq_length)
        text_dataloader = DataLoader(text_dataset, batch_size)

        # Model
        lstm_model = Net(1, hidden_size, len(text_dataset.chars), batch_size)  # 1 because we enter a single number/letter per step.

        # loss = torch.nn.CrossEntropyLoss().to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(lstm_model.parameters(), lr=lr, momentum=momentum)

        accuracy = train(lstm_model, text_dataloader, number_of_epochs, optimizer, loss)
        return accuracy

    n_trials = 2
    study_name = f"{model}_study"
    study = optuna.create_study(study_name=study_name, direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_trial = {
        "accuracy": study.best_trial.value,
        "batch_size": study.best_trial.params["batch_size"],
        "lr": study.best_trial.params["lr"],
        "momentum": study.best_trial.params["momentum"]
    }

    model_dir = f"./Dataset/{model}/{task}/{number_of_epochs}/"
    ensure_directory_exists(model_dir)

    with open(f"{model_dir}/best_trial.json", "w") as f:
        json.dump(best_trial, f, indent=4)

    trials_df = study.trials_dataframe()
    filtered_trials = trials_df[["value", "params_batch_size", "params_lr", "params_momentum"]]

    filtered_trials = filtered_trials.rename(columns={
        "value": "accuracy",
        "params_batch_size": "batch_size",
        "params_lr": "lr",
        "params_momentum": "momentum"
    })

    trials_dict = filtered_trials.astype(str).to_dict(orient='records')
    with open(f"{model_dir}/optuna_{n_trials}.json", "w") as f:
        json.dump(trials_dict, f, indent=4)

    print(f"Trials for {model} saved")



if __name__ == "__main__":

    task = 'text_generation'
    model = "LSTM"
    number_of_epochs = 2

    main(model, number_of_epochs)