import optuna
from torch.cuda import OutOfMemoryError

from ab.nn.util.Loader import Loader
from ab.nn.util.Stat import *
from ab.nn.util.Train import Train


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
    # Initialize the SQLite database
    initialize_database()
    # Determine configurations based on the provided config
    sub_configs = provide_all_configs(config)

    if not isinstance(n_epochs, tuple):
        n_epochs = (n_epochs,)
    for epoch in n_epochs:
        print(f"Training configurations ({epoch} epochs):")
        for idx, sub_config in enumerate(sub_configs, start=1):
            print(f"{idx}. {sub_config}")
        for sub_config in sub_configs:
                task, dataset_name, metric, model_name = conf_to_names(sub_config)
                model_dir: str = os.path.join(Const.stat_dir_global, sub_config, str(epoch))
                trials_file = os.path.join(model_dir, 'trials.json')

                n_optuna_trials_left = count_trials_left(trials_file, model_name, n_optuna_trials)
                if n_optuna_trials_left == 0:
                    print(f"The model {model_name} has already passed all trials for task: {task}, dataset: {dataset_name},"
                          f" metric: {metric}, epochs: {epoch}")
                else:
                    print(f"\nStarting training for the model: {model_name}, task: {task}, dataset: {dataset_name},"
                          f" metric: {metric},  epochs: {epoch}")

                    continue_study = True
                    max_batch_binary_power_local = max_batch_binary_power
                    while continue_study and max_batch_binary_power_local > -1:
                        try:
                            # Configure Optuna for the current model
                            def objective(trial):
                                try:
                                    # Suggest hyperparameters
                                    transform_name = trial.suggest_categorical('transform', ['cifar10_complex', 'cifar10_norm', 'cifar10_norm_32', 'echo'])
                                    if task == 'img_segmentation':
                                            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=False)
                                            momentum = trial.suggest_float('momentum', 0.8, 0.99, log=True)
                                    else:
                                            lr = trial.suggest_float('lr', 1e-4, 1, log=False)
                                            momentum = trial.suggest_float('momentum', 0.01, 0.99, log=True)
                                    batch_size = trial.suggest_categorical('batch_size', [max_batch(x) for x in range(max_batch_binary_power_local + 1)])
                                    print(f"Initialize training with lr = {lr}, momentum = {momentum}, batch_size = {batch_size}, transform: {transform_name}")

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
                            save_results(model_dir, study, sub_config, epoch)
                            continue_study = False
                        except CudaOutOfMemory as e:
                            max_batch_binary_power_local = e.batch_size_power() - 1
                            print(f"Max batch is decreased to {max_batch(max_batch_binary_power_local)} due to a CUDA Out of Memory Exception for model '{model_name}'")


if __name__ == "__main__":
    a = args()
    main(a.config, a.epochs, a.trials, a.max_batch_binary_power)
