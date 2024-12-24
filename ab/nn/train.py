import os

import optuna
from torch.cuda import OutOfMemoryError

from ab.nn.util.Loader import Loader
from ab.nn.util.Stat import *
from ab.nn.util.Train import Train


def save_results(model_dir, study, config_model_name):
    """
    Save Optuna study results for a given model in JSON-format.
    :param model_dir: Directory for the model statistics.
    :param study: Optuna study object.
    :param config_model_name: Config (Task, Dataset, Normalization) and Model name.
    """
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

    i = 0
    while i < len(trials_dict):
        dic = trials_dict[i]
        acc =  dic['accuracy']
        if math.isnan(acc) or math.isinf(acc):
            dic['accuracy'] = 0.0
            trials_dict[i] = dic
        i += 1

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


def main(config: str | tuple = default_config, n_epochs: int | tuple = default_epochs,
         n_optuna_trials: int | str = default_trials, max_batch_binary_power: int = default_max_batch_power):
    """
    Main function for training models using Optuna optimization.
    :param config: Configuration specifying the model training pipelines. The default value for all configurations.
    :param n_epochs: Number or tuple of numbers of training epochs.
    :param n_optuna_trials: Number of Optuna trials.
    :param max_batch_binary_power: Maximum binary power for batch size: for a value of 6, the batch size is 2**6 = 64
    """

    # Parameters specific to dataset loading.
    define_global_paths()
    # Determine configurations based on the provided config
    sub_configs = provide_all_configs(config)

    if not isinstance(n_epochs, tuple):
        n_epochs = (n_epochs,)
    for epoch in n_epochs:
        print(f"Training configurations ({epoch} epochs):")
        for idx, sub_config in enumerate(sub_configs, start=1):
            print(f"{idx}. {sub_config}")
        for sub_config in sub_configs:
                task, dataset_name, metric, transform_name, model_name = conf_to_names(sub_config)
                model_dir: str = os.path.join(Const.stat_dir_global, sub_config, str(epoch))
                trials_file = os.path.join(model_dir, 'trials.json')

                n_optuna_trials_left = count_trials_left(trials_file, model_name, n_optuna_trials)
                if n_optuna_trials_left == 0:
                    print(f"The model {model_name} has already passed all trials for task: {task}, dataset: {dataset_name},"
                          f" metric: {metric}, transform: {transform_name}, epochs: {epoch}")
                else:
                    print(f"\nStarting training for the model: {model_name}, task: {task}, dataset: {dataset_name},"
                          f" metric: {metric}, transform: {transform_name}, epochs: {epoch}")
                    # Paths for loader and transform
                    loader_path = f"loader.{dataset_name}.loader"
                    transform_path = f"transform.{transform_name}.transform" if transform_name else None

                    # Load dataset
                    try:
                        output_dimension, train_set, test_set = Loader.load_dataset(loader_path, transform_path)
                    except Exception as e:
                        print(f"Skipping model '{model_name}': failed to load dataset. Error: {e}")
                        continue

                    continue_study = True
                    max_batch_binary_power_local = max_batch_binary_power
                    while continue_study and max_batch_binary_power_local > -1:
                        try:
                            # Configure Optuna for the current model
                            def objective(trial):
                                try:
                                    if task == 'img_segmentation':
                                            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=False)
                                            momentum = trial.suggest_float('momentum', 0.8, 0.99, log=True)
                                    else:
                                            lr = trial.suggest_float('lr', 1e-4, 1, log=False)
                                            momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
                                    batch_size = trial.suggest_categorical('batch_size', [max_batch(x) for x in range(max_batch_binary_power_local + 1)])
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

                                    trainer = Train(
                                        model_source_package=f"dataset.{model_name}",
                                        task_type=task,
                                        train_dataset=train_set,
                                        test_dataset=test_set,
                                        metric=metric,
                                        output_dimension=output_dimension,
                                        lr=lr,
                                        momentum=momentum,
                                        batch_size=batch_size)
                                    return trainer.evaluate(epoch)
                                except Exception as e:
                                    if isinstance(e, OutOfMemoryError):
                                        if max_batch_binary_power_local <= 0:
                                            return 0.0
                                        else:
                                            raise CudaOutOfMemory(batch_size)
                                    else:
                                        print(f"error '{model_name}': failed to train. Error: {e}")
                                        return 0.0
                            # Launch Optuna for the current model
                            study = optuna.create_study(study_name=model_name, direction='maximize')
                            study.optimize(objective, n_trials=n_optuna_trials_left)

                            # Save results
                            save_results(model_dir, study, sub_config)
                            continue_study = False
                        except CudaOutOfMemory as e:
                            max_batch_binary_power_local = e.batch_size_power() - 1
                            print(f"Max batch is decreased to {max_batch(max_batch_binary_power_local)} due to a CUDA Out of Memory Exception for model '{model_name}'")


if __name__ == "__main__":
    a = args()
    main(a.config, a.epochs, a.trials, a.max_batch_binary_power)
