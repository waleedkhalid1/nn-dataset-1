import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from datasets import load_dataset
import optuna
import os
import json
import time
from tqdm import tqdm
from Dataset.Llama.code import Net, ModelArgs, DatasetPreparation




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using '{device}' device")

def ensure_directory_exists(model_dir):
    directory = os.path.dirname(model_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

def train(model: Net, datas: DataLoader, number_of_epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
    model.to(device)
    model.train()
    total_start_time = time.time()

    for epoch in tqdm(range(number_of_epochs), desc="Training Progress", unit="epoch"):
        epoch_start_time = time.time()
        total_correct = 0
        total_count = 0

        for data in datas:
            X, y = data
            X = X.to(device)
            Y = y.to(device)
            optimizer.zero_grad()
            start_pos = 0

            # feedforward
            output = model(X, start_pos)
            loss = loss_fn(output.view(-1, model.params.vocab_size), Y.view(-1))

            loss.requires_grad = True

            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            else:
                print(loss.backward())
                print(optimizer.step())
                print("Error: Loss tensor does not require gradients to be calculated")

            # Accuracy calculation
            pred = output.argmax(dim=-1)
            correct = (pred == Y).sum().item()
            total_correct += correct
            total_count += Y.numel()

        accuracy = total_correct / total_count
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch + 1} finished in: {epoch_time:.2f} seconds')

    total_time = time.time() - total_start_time
    print(f'Training finished in: {total_time:.2f} seconds')

    return accuracy

def main(model: str, number_of_epochs: int):
    """The main function for loading data, setting up the model and training"""

    # Set dataset path
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    data = "\n".join(dataset["train"]["text"]).lower()

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1, log=False)
        momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
        batch_size = trial.suggest_categorical('batch_size', [4, 5, 8, 16, 32, 64])

        print(f"Initializing TGModelEvaluator with lr = {lr}, momentum = {momentum}, batch_size = {batch_size}")

        # Model parameters
        model_args = ModelArgs(
            dim=2048,
            n_layers=32,
            n_heads=16,
            vocab_size=len(set(data))
        )
        llama_model = Net(model_args)

        # Data preparation
        seq_length = 128
        text_dataset = DatasetPreparation(data, seq_length=seq_length)
        text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and loss function
        optimizer = optim.SGD(llama_model.parameters(), lr=lr, momentum=momentum)
        loss_fn = nn.CrossEntropyLoss()

        # Model training
        accuracy = train(llama_model, text_dataloader, number_of_epochs, optimizer, loss_fn)
        return accuracy

    n_trials = 100
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
    model = "Llama"
    number_of_epochs = 2

    main(model, number_of_epochs)