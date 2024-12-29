import optuna
from torch.cuda import OutOfMemoryError

from ab.nn.util.Loader import Loader
from ab.nn.util.Stat import *
from ab.nn.util.Train import Train


def main(config: str | tuple = default_config, n_epochs: int = default_epochs,
         n_optuna_trials: int | str = default_trials,
         min_batch_binary_power: int = default_min_batch_power, max_batch_binary_power: int = default_max_batch_power,
         min_learning_rate: float = default_min_lr, max_learning_rate: float = default_max_lr,
         min_momentum: float = default_min_momentum, max_momentum: float = default_max_momentum,
         transform: str = None, nn_fail_attempts: int = default_nn_fail_attempts):
    """
    Main function for training models using Optuna optimization.
    :param config: Configuration specifying the model training pipelines. The default value for all configurations.
    :param n_epochs: Number of training epochs.
    :param n_optuna_trials: Number of Optuna trials.
    :param min_batch_binary_power: Minimum power of two for batch size. E.g., with a value of 0, batch size equals 2**0 = 1.
    :param max_batch_binary_power: Maximum power of two for batch size. E.g., with a value of 12, batch size equals 2**12 = 4096.
    :param min_learning_rate: Minimum value of learning rate.
    :param max_learning_rate: Maximum value of learning rate.
    :param min_momentum: Minimum value of momentum.
    :param max_momentum: Maximum value of momentum.
    :param transform: The transformation algorithm name. If None (default), all available algorithms are used by Optuna.
    :param nn_fail_attempts: Number of attempts if the neural network model throws exceptions.
    """

    if min_batch_binary_power > max_batch_binary_power: raise Exception(f"min_batch_binary_power {min_batch_binary_power} > max_batch_binary_power {max_batch_binary_power}")
    if min_learning_rate > max_learning_rate: raise Exception(f"min_learning_rate {min_learning_rate} > max_learning_rate {max_learning_rate}")
    if min_momentum > max_momentum: raise Exception(f"min_momentum {min_momentum} > max_momentum {max_momentum}")


    # Parameters specific to dataset loading.
    define_global_paths()
    # Initialize the SQLite database
    initialize_database()
    # Determine configurations based on the provided config
    sub_configs = get_configs(config)

    print(f"Training configurations ({n_epochs} epochs):")
    for idx, sub_config in enumerate(sub_configs, start=1):
        print(f"{idx}. {sub_config}")
    for sub_config in sub_configs:
            task, dataset_name, metric, model_name = conf_to_names(sub_config)
            model_stat_dir: str = join(Const.stat_dir_global, sub_config)
            trials_file = join(model_stat_dir, str(n_epochs), 'trials.json')
            n_optuna_trials_left = count_trials_left(trials_file, model_name, n_optuna_trials)
            if n_optuna_trials_left == 0:
                print(f"The model {model_name} has already passed all trials for task: {task}, dataset: {dataset_name},"
                      f" metric: {metric}, epochs: {n_epochs}")
            else:
                print(f"\nStarting training for the model: {model_name}, task: {task}, dataset: {dataset_name},"
                      f" metric: {metric}, epochs: {n_epochs}")
                fail_iterations = nn_fail_attempts
                continue_study = True
                max_batch_binary_power_local = max_batch_binary_power
                while continue_study and max_batch_binary_power_local > -1 and fail_iterations > -1:
                    try:
                        # Configure Optuna for the current model
                        def objective(trial):
                            try:
                                # Suggest hyperparameters
                                transform_name = trial.suggest_categorical('transform', [transform] if transform is not None else ['cifar10_complex', 'cifar10_norm', 'cifar10_norm_32', 'echo'])
                                lr = trial.suggest_float('lr', min_learning_rate, max_learning_rate, log=True)
                                momentum = trial.suggest_float('momentum', min_momentum, max_momentum, log=False)
                                batch = trial.suggest_categorical('batch', [max_batch(x) for x in range(min_batch_binary_power, max_batch_binary_power_local + 1)])
                                print(f"Initialize training with lr: {lr}, momentum: {momentum}, batch: {batch}, transform: {transform_name}")
                             # Load dataset with the chosen transformation
                                loader_path = f"loader.{dataset_name}.loader"
                                transform_path = f"transform.{transform_name}.transform"
                                # Load dataset
                                try:
                                    output_dimension, train_set, test_set = Loader.load_dataset(loader_path, transform_path)
                                except Exception as e:
                                    print(f"Skipping model '{model_name}': failed to load dataset. Error: {e}")
                                    return 0.0

                                # Initialize model and trainer
                                if task == 'txt_generation':
                                    # Dynamically import RNN or LSTM model
                                    if model_name.lower() == 'rnn':
                                        from ab.nn.dataset.RNN import Net as RNNNet
                                        model = RNNNet(1, 256, len(train_set.chars), batch)
                                    elif model_name.lower() == 'lstm':
                                        from ab.nn.dataset.LSTM import Net as LSTMNet
                                        model = LSTMNet(1, 256, len(train_set.chars), batch, num_layers=2)
                                    else:
                                        raise ValueError(f"Unsupported text generation model: {model_name}")
                                trainer = Train(config=sub_config, model_source_package=f"dataset.{model_name}",
                                                model_stat_dir=model_stat_dir, task=task, train_dataset=train_set,
                                                test_dataset=test_set, metric=metric, output_dimension=output_dimension, lr=lr,
                                                momentum=momentum, batch=batch, transform=transform_name)
                                return trainer.train_n_eval(n_epochs)
                            except Exception as e:
                                if isinstance(e, OutOfMemoryError):
                                    if max_batch_binary_power_local <= 0:
                                        return 0.0
                                    else:
                                        raise CudaOutOfMemory(batch)
                                else:
                                    print(f"error '{model_name}': failed to train. Error: {e}")
                                    if fail_iterations < 0:
                                        return 0.0
                                    else:
                                        raise ModelException()
                        # Launch Optuna for the current NN model
                        study = optuna.create_study(study_name=model_name, direction='maximize')
                        study.optimize(objective, n_trials=n_optuna_trials_left)
                        continue_study = False
                    except ModelException:
                        fail_iterations -= 1
                    except CudaOutOfMemory as e:
                        max_batch_binary_power_local = e.batch_size_power() - 1
                        print(f"Max batch is decreased to {max_batch(max_batch_binary_power_local)} due to a CUDA Out of Memory Exception for model '{model_name}'")


if __name__ == "__main__":
    a = args()
    main(a.config, a.epochs, a.trials, a.min_batch_binary_power, a.max_batch_binary_power,
         a.min_learning_rate, a.max_learning_rate, a.min_momentum, a.max_momentum, a.transform, a.nn_fail_attempts)
