import json
import os

import optuna

from ab.nn.util.Const import stat_dir
from ab.nn.util.Loader import Loader
from ab.nn.util.Train import Train
from ab.nn.util.Util import args, ensure_directory_exists, count_trials_left


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


def extract_all_configs(config) -> list[str]:
    """
    Collect models matching the given configuration prefix
    """
    l = list(set([sub_config
                  for sub_config in os.listdir(stat_dir)
                  if sub_config.startswith(config) and os.path.isdir(os.path.join(stat_dir, sub_config))]))
    l.sort()
    return l


# todo: move to the ab.nn.util.Start and request information from database
def provide_all_configs(config) -> tuple[str]:
    if not isinstance(config, tuple):
        config = (config,)
    all_configs = []
    for c in config:
        all_configs = all_configs + extract_all_configs(c)
    return tuple(all_configs)


def main(config: str | tuple, n_epochs: int | tuple, n_optuna_trials: int | str, max_batch_binary_power: int = 6):
    """
    Main function for training models using Optuna optimization.
    :param config: Configuration specifying the model training pipelines. The default value for all configurations.
    :param n_epochs: Number or tuple of numbers of training epochs.
    :param n_optuna_trials: Number of Optuna trials.
    :param max_batch_binary_power: Maximum binary power for batch size: for a value of 6, the batch size is 2^6 = 64
    """

    # Parameters specific to dataset loading.
    dataset_params = {}

    # Determine configurations based on the provided config
    sub_configs = provide_all_configs(config)

    if not isinstance(n_epochs, tuple):
        n_epochs = (n_epochs,)
    for epoch in n_epochs:
        print(f"Configurations found for training for {epoch} epochs:")
        for idx, sub_config in enumerate(sub_configs, start=1):
            print(f"{idx}. {sub_config}")
        for sub_config in sub_configs:
                task, dataset_name, metric, transform_name, model_name = sub_config.split('-')
                model_dir : str = os.path.join(stat_dir, sub_config, str(epoch))
                trials_file = os.path.join(model_dir, 'trials.json')

                n_optuna_trials_left = count_trials_left(trials_file, model_name, n_optuna_trials)
                if n_optuna_trials_left == 0:
                    print(f"The model {model_name} has already passed all trials for task: {task}, dataset: {dataset_name},"
                          f" metric: {metric}, transform: {transform_name}, epochs: {epoch}")
                else:
                    print(f"\nStarting training for the model: {model_name}, task: {task}, dataset: {dataset_name},"
                          f" metric: {metric}, transform: {transform_name}, epochs: {epoch}")
                    if task == "img_segmentation":
                        import ab.nn.loader.coco as coco
                        dataset_params['class_list'] = coco.class_list()
                        dataset_params['path'] = "./data/coco"
                    # Paths for loader and transform
                    loader_path = f"loader.{dataset_name}.loader"
                    transform_path = f"transform.{transform_name}.transform" if transform_name else None

                    # Load dataset
                    try:
                        output_dimension, train_set, test_set = Loader.load_dataset(loader_path, transform_path, **dataset_params)
                    except Exception as e:
                        print(f"Skipping model '{model_name}': failed to load dataset. Error: {e}")
                        continue

                    # Configure Optuna for the current model
                    def objective(trial):
                        if task == 'img_segmentation':
                            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=False)
                            momentum = trial.suggest_float('momentum', 0.8, 0.99, log=True)
                        else:
                            lr = trial.suggest_float('lr', 1e-4, 1, log=False)
                            momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
                        batch_size = trial.suggest_categorical('batch_size', [2 ** x for x in range(max_batch_binary_power + 1)])
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

                    # Launch Optuna for the current model
                    study_name = f"{model_name}_study"
                    study = optuna.create_study(study_name=study_name, direction='maximize')
                    study.optimize(objective, n_trials=n_optuna_trials_left)

                    # Save results
                    save_results(model_dir, study, sub_config)

if __name__ == "__main__":
    a = args()
    main(a.config, a.epochs, a.trials, a.max_batch_binary_power)
