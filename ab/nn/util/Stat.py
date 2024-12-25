import os
import sqlite3
from os import listdir

from ab.nn.util.Util import *


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
def provide_all_configs(config) -> tuple[str]:
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
    Initialize the SQLite database and create the `results` table if it doesn't exist.
    """
    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_model_name TEXT,
        accuracy REAL,
        batch_size INTEGER,
        lr REAL,
        momentum REAL,
        transform TEXT,
        epoch INTEGER
    )
    """)

    conn.commit()
    conn.close()


def save_results(model_dir, study, config_model_name, epoch):
    """
    Save Optuna study results for a given model in JSON-format.
    :param model_dir: Directory for the model statistics.
    :param study: Optuna study object.
    :param config_model_name: Config (Task, Dataset, Normalization) and Model name.
    """
    ensure_directory_exists(model_dir)

    # Save all trials as trials.json
    trials_df = study.trials_dataframe()
    filtered_trials = trials_df[['value', 'params_batch_size', 'params_lr', 'params_momentum', 'params_transform']]

    filtered_trials = filtered_trials.rename(columns={
        'value': 'accuracy',
        'params_batch_size': 'batch_size',
        'params_lr': 'lr',
        'params_momentum': 'momentum',
        'params_transform': 'transform'
    })

    filtered_trials = filtered_trials.astype({
        'accuracy': float,
        'batch_size': int,
        'lr': float,
        'momentum': float,
        'transform': str
    })

    trials_dict = filtered_trials.to_dict(orient='records')

    i = 0
    while i < len(trials_dict):
        dic = trials_dict[i]
        acc = dic['accuracy']
        if math.isnan(acc) or math.isinf(acc):
            dic['accuracy'] = 0.0
            trials_dict[i] = dic
        i += 1

    trials_dict_all = trials_dict
    path = f"{model_dir}/trials.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            previous_trials = json.load(f)
            trials_dict_all = previous_trials + trials_dict_all

    trials_dict_all = sorted(trials_dict_all, key=lambda x: x['accuracy'], reverse=True)
    # Save trials.json
    with open(path, "w") as f:
        json.dump(trials_dict_all, f, indent=4)

    # Save best_trial.json
    with open(f"{model_dir}/best_trial.json", "w") as f:
        json.dump(trials_dict_all[0], f, indent=4)

    print(f"Trials for {config_model_name} saved at {model_dir}")

    # Save results to SQLite DB
    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Insert each trial into the database with epoch
    for trial in trials_dict:
        cursor.execute("""
        INSERT INTO results (config_model_name, accuracy, batch_size, lr, momentum, transform, epoch)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (config_model_name, trial['accuracy'], trial['batch_size'], trial['lr'],
              trial['momentum'], trial['transform'], epoch))

    conn.commit()
    conn.close()
