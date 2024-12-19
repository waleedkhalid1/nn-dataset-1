from ab.nn import train

if __name__ == "__main__":
    # NN pipeline configuration examples
    conf = ''  # For all configurations
    # conf = 'img_classification' # For all image classification configurations
    # conf = 'img_classification-cifar10-acc-cifar10_norm' # For a particular configuration for all models
    # conf = 'img_classification-cifar10-acc-cifar10_norm-GoogLeNet'  # For a particular configuration and model
    # conf = ('img_classification', 'img_segmentation')  # For all image classification and segmentation configurations

    # Number of Optuna trial examples.
    optuna_trials = 100  # 100 trials
    # optuna_trials = "+1"  # Try once more. For quick verification of model training after code modifications.
    # optuna_trials = "+5"  # Try 5 more times. To thoroughly verify model training after code modifications.

    ''' !!! Commit updated statistics whenever it's generated !!! '''
    # Run training with Optuna: detects and saves performance metric values for a varying number of epochs
    train.main(conf, (1, 2, 5), optuna_trials)

