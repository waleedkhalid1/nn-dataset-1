import json
import os
import sqlite3
from os import listdir, makedirs

import pandas as pd

from ab.nn.util.Util import *


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

def extract_all_configs(config) -> list[str]:
    """
    Collect models matching the given configuration prefix
    """
    l = [c for c in listdir(Const.stat_dir_global) if c.startswith(config)]
    if not l and is_full_config(config):
        makedirs(join(Const.stat_dir_global, config))
        l = [config]
    return l


# todo: Request information from database 
# once the database is loaded, the function will be updated
def get_configs(config) -> tuple[str]:
    if not isinstance(config, tuple):
        config = (config,)
    all_configs = []
    for c in config:
        all_configs = all_configs + extract_all_configs(c)
    all_configs: list = list(set(all_configs))
    all_configs.sort()
    return tuple(all_configs)


def initialize_database():
    """
    Initialize the SQLite database and create the `stat` table if it doesn't exist.
    """
    makedirs(Path(Const.db_dir_global).parent.absolute(), exist_ok=True)
    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_model_name TEXT,
        accuracy REAL,
        batch INTEGER,
        lr REAL,
        momentum REAL,
        transform TEXT,
        epoch INTEGER
    )
    """)

    conn.commit()
    conn.close()


def save_results(config: str, model_stat_file: str, prm: dict):
    """
    Save Optuna study results for a given model in JSON-format.
    :param config: Config (Task, Dataset, Metric, and Model name).
    :param model_stat_file: File for the model statistics.
    :param prm: Dictionary of all saved parameters.
    """
    trials_dict = [prm]
    trials_dict_all = trials_dict

    if os.path.exists(model_stat_file):
        with open(model_stat_file, "r") as f:
            previous_trials = json.load(f)
            trials_dict_all = previous_trials + trials_dict_all

    trials_dict_all = sorted(trials_dict_all, key=lambda x: x['accuracy'], reverse=True)
    # Save trials.json
    with open(model_stat_file, "w") as f:
        json.dump(trials_dict_all, f, indent=4)

    print(f"Trial (accuracy {prm['accuracy']}) for {config} saved at {model_stat_file}")

    # Save results to SQLite DB
    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Insert each trial into the database with epoch
    for trial in trials_dict:
        cursor.execute("""
        INSERT INTO stat (config_model_name, accuracy, batch, lr, momentum, transform, epoch)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (config, trial['accuracy'], trial['batch'], trial['lr'],
              trial['momentum'], trial['transform'], trial['epoch']))

    conn.commit()
    conn.close()

def get_pandas_df_all() -> pd.DataFrame:
    """
    Get the full dataset including its default full statistics as a pandas dataframe
    Returns:
    pd.Dataframe with columns ["structure", "code", "epochs", "accuracy", "batch", "lr", "momentum", "transform"]
    todo: update this to load from the db file
    """
    out = pd.DataFrame(columns=["structure", "code", "epochs", "accuracy", "batch", "lr", "momentum", "transform"])
    pwd = str(Path(__file__).parent.resolve())
    for stat_dir in listdir(pwd + "/../stat/"):
        structure = stat_dir.split('-')[-1]

        with open(pwd + "/../dataset/" + structure + ".py", "r") as code_file:
            code = str(code_file.read())

        for epochs in listdir(pwd + "/../stat/" + stat_dir + "/"):
            with open(pwd + "/../stat/" + stat_dir + "/" + epochs + ".json", "r") as json_file:
                content = json.load(json_file)

            for stat in content:
                next_row = [
                    structure,
                    code,
                    epochs,
                    stat["accuracy"],
                    stat["batch"],
                    stat["lr"],
                    stat["momentum"],
                    stat["transform"]
                ]
                out.loc[len(out)] = next_row
    return out

def get_pandas_df_best() -> pd.DataFrame:
    """
        Get the full dataset including its best default statistics as a pandas dataframe
        Returns:
        pd.Dataframe with columns ["structure", "code", "epochs", "accuracy", "batch", "lr", "momentum", "transform"]
        todo: update this to load from the db file
    """
    out = pd.DataFrame(columns=["structure", "code", "epochs", "accuracy", "batch", "lr", "momentum", "transform"])
    pwd = str(Path(__file__).parent.resolve())
    for stat_dir in listdir(pwd + "/../stat/"):
        structure = stat_dir.split('-')[-1]

        with open(pwd + "/../dataset/" + structure + ".py", "r") as code_file:
            code = str(code_file.read())

        for epochs in listdir(pwd + "/../stat/" + stat_dir + "/"):
            with open(pwd + "/../stat/" + stat_dir + "/" + epochs + "/best_trial.json", "r") as json_file:
                stat = json.load(json_file)

            next_row = [
                structure,
                code,
                epochs,
                stat["accuracy"],
                stat["batch"],
                stat["lr"],
                stat["momentum"],
                stat["transform"]
            ]
            out.loc[len(out)] = next_row
    return out
