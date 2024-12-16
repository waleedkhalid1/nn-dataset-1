import json
import os

import optuna
from ab.nn.util.TrainModel import TrainModel
from ab.nn.util.DatasetLoader import DatasetLoader


def parse_model_config(directory_name):
    """
    Parse the model configuration to extract task, dataset, and optional transformation.
    :param directory_name: Name of the directory (e.g., "img_classification-cifar10-cifar10_norm-AlexNet").
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


def save_results(config_model_name, study, n_epochs):
    """
    Save Optuna study results for a given model in JSON-format.
    :param config_model_name: Config (Task, Dataset, Normalization) and Model name.
    :param study: Optuna study object.
    :param n_epochs: Number of epochs.
    """

    model_dir = f"stat/{config_model_name}/{n_epochs}/"
    ensure_directory_exists(model_dir)

    # Save all trials as trials.json
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
    path = f"{model_dir}/trials.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            previous_trials = json.load(f)
            trials_dict = previous_trials + trials_dict

    trials_dict = sorted(trials_dict, key=lambda x: x['accuracy'], reverse=True)
    # Save trials.json
    with open(path, "w") as f:
        json.dump(trials_dict, f, indent=4)

    # Save best_trial.json
    with open(f"{model_dir}/best_trial.json", "w") as f:
        json.dump(trials_dict[0], f, indent=4)

    print(f"Trials for {config_model_name} saved at {model_dir}")


def trials_left(config_model_name, model_name, n_epochs, n_optuna_trials):
    model_dir = f"stat/{config_model_name}/{n_epochs}/"
    path = f"{model_dir}/trials.json"
    n_passed_trials = 0
    if os.path.exists(path):
        with open(path, "r") as f:
            trials = json.load(f)
            n_passed_trials = len(trials)
    n_trials_left = max(0, n_optuna_trials - n_passed_trials)
    if n_passed_trials > 0:
        print(f"The {model_name} passed {n_passed_trials} trials, {n_trials_left} left.")
    return n_trials_left


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
    elif len(config.split('-')) == 3:  # Partial configuration, e.g., 'img_classification-cifar10-cifar10_norm'
        # Collect models matching the given configuration prefix
        config_prefix = config + '-'
        sub_configs = [
            sub_config
            for sub_config in os.listdir("stat")
            if sub_config.startswith(config_prefix) and os.path.isdir(os.path.join("stat", sub_config))
        ]
    else:  # Specific configuration, e.g., 'img_classification-cifar10-cifar10_norm-AlexNet'
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

            n_optuna_trials_left = trials_left(config_name, model_name, n_epochs, n_optuna_trials)
            if n_optuna_trials_left == 0:
                print(f"The model {model_name} has already passed all trials for task: {task}, dataset: {dataset_name}, transform: {transform_name}, epochs: {n_epochs}")
            else:
                print(f"\nStarting training for the model: {model_name}, task: {task}, dataset: {dataset_name}, transform: {transform_name}, epochs: {n_epochs}")
                if task == "img_segmentation":
                    import ab.nn.loader.cocos as cocos
                    dataset_params['class_list'] = cocos.class_list()
                    dataset_params['path'] = "./data/cocos"
                # Paths for loader and transform
                loader_path = f"loader.{dataset_name}.loader"
                transform_path = f"transform.{transform_name}.transform" if transform_name else None

                # Load dataset
                try:
                    output_dimension, train_set, test_set = DatasetLoader.load_dataset(loader_path, transform_path,
                                                                                  **dataset_params)
                except Exception as e:
                    print(f"Skipping model '{model_name}': failed to load dataset. Error: {e}")
                    continue

                # Configure Optuna for the current model
                def objective(trial):
                    if task == 'img_segmentation':
                        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=False)
                        momentum = trial.suggest_float('momentum', 0.8, 0.99, log=True)
                        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
                    else:
                        lr = trial.suggest_float('lr', 1e-4, 1, log=False)
                        momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
                        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])

                    print(f"Initialize training with lr = {lr}, momentum = {momentum}, batch_size = {batch_size}")

                    if task == 'txt_generation':
                        # Dynamically import RNN or LSTM model
                        if model_name.lower() == 'rnn':
                            from dataset.RNN import Net as RNNNet
                            model = RNNNet(1, 256, len(train_set.chars), batch_size)
                        elif model_name.lower() == 'lstm':
                            from ab.nn.dataset.LSTM import Net as LSTMNet
                            model = LSTMNet(1, 256, len(train_set.chars), batch_size, num_layers=2)
                        else:
                            raise ValueError(f"Unsupported text generation model: {model_name}")

                    trainer = TrainModel(
                            model_source_package=f"dataset.{model_name}",
                            train_dataset=train_set,
                            test_dataset=test_set,
                            output_dimension=output_dimension,
                            lr=lr,
                            momentum=momentum,
                            batch_size=batch_size,
                            task_type=task,
                            manual_args=manual_args.get(model_name) if manual_args else None
                        )
                    return trainer.evaluate(n_epochs)

                # Launch Optuna for the current model
                study_name = f"{model_name}_study"
                study = optuna.create_study(study_name=study_name, direction='maximize')
                study.optimize(objective, n_trials=n_optuna_trials_left)

                # Save results
                save_results(sub_config, study, n_epochs)


if __name__ == "__main__":
    # Config examples
    config ='all' # For all configurations
    # config = 'img_classification-cifar10-cifar10_norm' # For a particular configuration for all models
    # config = 'img_classification-cifar10-cifar10_complex-ComplexNet'  # For a particular configuration and model

    # Detects and saves performance metric values for a varying number of epochs
    for epochs in [1, 2, 5]:
        # Run training with Optuna
        main(config, epochs, 100)
