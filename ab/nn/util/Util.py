import argparse
import json
import os

from ab.nn.util.Const import nn_module


def nn_mod(*nms):
    return ".".join((nn_module,) + nms)


def ensure_directory_exists(model_dir):
    """
    Ensures that the directory for the given path exists.
    :param model_dir: Path to the target directory or file.
    :return: Creates the directory if it does not exist.
    """
    directory = os.path.dirname(model_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)


def count_trials_left(trial_file, model_name, n_optuna_trials):
    """
    Calculates the remaining Optuna trials based on the completed ones. Checks for a "trials.json" file in the
    specified directory to determine how many trials have been completed, and returns the number of trials left.
    :param trial_file: Trial file path
    :param model_name: Name of the model.
    :param n_optuna_trials: Either the total number of Optuna trials, or if the value is negative or a string, it is considered the number of additional Optuna trials.
    :return: n_trials_left: Remaining trials.
    """
    n_passed_trials = 0
    if os.path.exists(trial_file):
        with open(trial_file, "r") as f:
            trials = json.load(f)
            n_passed_trials = len(trials)
    if isinstance(n_optuna_trials, str):
        n_optuna_trials = - int(n_optuna_trials)
    n_trials_left = abs(n_optuna_trials) if n_optuna_trials < 0 else max(0, n_optuna_trials - n_passed_trials)
    if n_passed_trials > 0:
        print(f"The {model_name} passed {n_passed_trials} trials, {n_trials_left} left.")
    return n_trials_left

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='',
        help="Configuration specifying the model training pipelines. The default value for all configurations.")
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Numbers of training epochs",
        default=(1, 2, 5))
    parser.add_argument(
        '-t',
        '--trials',
        type=int,
        help="Number of Optuna trials",
        default=100)
    parser.add_argument(
        '-b',
        '--max_batch_binary_power',
        type=int,
        help="Maximum binary power for batch size: for a value of 6, the batch size is 2^6 = 64",
        default=6)
    return parser.parse_args()


