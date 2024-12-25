import argparse
import json
import math
from os import makedirs
from os.path import exists, join
from pathlib import Path

import ab.nn.util.Const as Const
from ab.nn.util.Const import *


def nn_mod(*nms):
    return ".".join(to_nn + nms)

def get_attr (mod, f):
    return getattr(__import__(nn_mod(mod), fromlist=[f]), f)


def conf_to_names(c: str) -> list[str]:
    return c.split('-')


def is_full_config(s: str):
    l = conf_to_names(s)
    return 4 == len(l) and exists(join(Const.dataset_dir_global, l[-1] + '.py'))


def ensure_directory_exists(model_dir):
    """
    Ensures that the directory for the given path exists.
    :param model_dir: Path to the target directory or file.
    :return: Creates the directory if it does not exist.
    """
    if not exists(model_dir):
        makedirs(model_dir)


def define_global_paths():
    """
    Defines project paths from current directory.
    """
    stat_dir = 'stat'

    import ab.nn.__init__ as init_file
    pref = Path(init_file.__file__).parent.absolute()
    Const.stat_dir_global = join(pref, stat_dir)
    Const.dataset_dir_global = join(pref, 'dataset')

    data_dir = 'data'
    dataset_file = 'ab.nn.stat.db'
    Const.data_dir_global = data_dir
    Const.db_dir_global = dataset_file
    if exists(stat_dir):
        project_root = ['..'] * len(to_nn)
        Const.data_dir_global = join(*project_root, data_dir)
        Const.db_dir_global = join(*project_root, dataset_file)

def max_batch (binary_power):
    return 2 ** binary_power


class CudaOutOfMemory(Exception):
    def __init__(self, batch):
        self.batch_power = int(math.log2(batch))

    def batch_size_power(self):
        return self.batch_power

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
    if exists(trial_file):
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
        default=default_config,
        help="Configuration specifying the model training pipelines. The default value for all configurations.")
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Numbers of training epochs",
        default=default_epochs)
    parser.add_argument(
        '-t',
        '--trials',
        type=int,
        help="Number of Optuna trials",
        default=default_trials)
    parser.add_argument(
        '-b',
        '--max_batch_binary_power',
        type=int,
        help="Maximum binary power for batch size: for a value of 6, the batch size is 2**6 = 64",
        default=default_max_batch_power)
    return parser.parse_args()


