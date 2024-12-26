from ab.nn import train
from ab.nn.util.Util import args

if __name__ == "__main__":
    a = args()

    # NN pipeline configuration examples
    conf = a.config  # From the command line argument --config
    # conf = ''  # For all configurations
    # conf = 'img_classification' # For all image classification configurations
    # conf = 'img_classification-cifar10-acc-cifar10_norm' # For a particular configuration for all models
    # conf = 'img_classification-cifar10-acc-cifar10_norm-GoogLeNet'  # For a particular configuration and model
    # conf = ('img_classification', 'img_segmentation')  # For all image classification and segmentation configurations

    # Number of Optuna trial examples.
    optuna_trials = a.trials  # From the command line argument --trials
    # optuna_trials = 100  # 100 trials
    # optuna_trials = "+1"  # Try once more: for quick verification of model training process after code modifications.
    # optuna_trials = "+5"  # Try 5 more times: to thoroughly verify model training after code modifications.

    ''' !!! Please commit updated statistics whenever it's generated !!! '''
    # Run training with Optuna: detects and saves performance metric values
    train.main(conf, a.epochs, optuna_trials, a.max_batch_binary_power, a.transform)

