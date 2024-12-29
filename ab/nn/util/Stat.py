import json
import os
import random
import sqlite3
from os import listdir, makedirs

import pandas as pd

from ab.nn.util.Util import *
import uuid


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
def get_configs(config: str, random_config_order: bool) -> tuple[str]:
    if not isinstance(config, tuple):
        config = (config,)
    all_configs = []
    for c in config:
        all_configs = all_configs + extract_all_configs(c)
    all_configs: list = list(set(all_configs))
    if random_config_order:
        random.shuffle(all_configs)
    else:
        all_configs.sort()
    return tuple(all_configs)


def initialize_database():
    """
    Initialize the SQLite database and create the `stat` and `nn` table if it doesn't exist.
    """
    makedirs(Path(Const.db_dir_global).parent.absolute(), exist_ok=True)
    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Create `nn` table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS nn (
        id TEXT PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        code TEXT
    )
    """)

    # Create `stat` table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stat (
        id TEXT PRIMARY KEY,
        task TEXT NOT NULL,
        dataset TEXT NOT NULL,
        metric TEXT NOT NULL,
        nn_id INTEGER NOT NULL,
        accuracy REAL,
        batch INTEGER,
        lr REAL,
        momentum REAL,
        transform TEXT,
        epoch INTEGER,
        time REAL,
        FOREIGN KEY (nn_id) REFERENCES nn (id) ON DELETE CASCADE
    )
    """)

    populate_nn_table(conn)
    conn.commit()
    conn.close()


def populate_nn_table(conn):
    """
    Populate the `nn` table with models from the dataset directory.
    """
    nn_directory = Path(Const.dataset_dir_global)
    nn_files = [
        Path(f) for f in nn_directory.iterdir() if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"
    ]

    cursor = conn.cursor()
    for nn_file in nn_files:
        print(f"Adding NN model {nn_file} to the `nn` table.")
        nn_id = str(uuid.uuid4())
        nn_name = nn_file.stem

        with open(nn_file, 'r') as file:
            model_code = file.read()
         # Check if the model exists in the database
        cursor.execute("SELECT id, code FROM nn WHERE name = ?", (nn_name,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # If model exists, update the code if it has changed
            nn_id, existing_code = existing_entry
            if existing_code != model_code:
                print(f"Updating code for model: {nn_name}")
                cursor.execute("UPDATE nn SET code = ? WHERE id = ?", (model_code, nn_id))
        else:
            # If model does not exist, insert it with a new UUID
            nn_id = str(uuid.uuid4())
            print(f"Adding new NN model {nn_name} to the `nn` table with UUID: {nn_id}")
            cursor.execute("INSERT INTO nn (id, name, code) VALUES (?, ?, ?)", (nn_id, nn_name, model_code))

    print(f"NN models added to the `nn` table: {nn_files}")


def clear_and_reload_database():
    """
    Clear the database and reload all NN models and statistics.
    """
    makedirs(Path(Const.db_dir_global).parent.absolute(), exist_ok=True)

    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS stat")
    cursor.execute("DROP TABLE IF EXISTS nn")
    conn.commit()

    # Reinitialize the database
    initialize_database()

    # Reload statistics
    load_all_statistics_from_json_to_db(conn)
    conn.close()

def load_all_statistics_from_json_to_db(conn):
    """
    Reload all statistics into the database for all subconfigs and epochs.
    """
    stat_base_path = Path(Const.stat_dir_global)
    sub_configs = [d.name for d in stat_base_path.iterdir() if d.is_dir()]
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    for epoch in epochs:

        for sub_config in sub_configs:
            model_dir = Path(Const.stat_dir_global) / sub_config / str(epoch)
            trials_file = model_dir / 'trials.json'

            if not trials_file.exists():
                print(f"Skipping {sub_config}: trials.json not found for epoch {epoch}.")
                continue

            with open(trials_file, 'r') as f:
                trials = json.load(f)

            # Insert statistics into the database
            for trial in trials:
                task, dataset, metric, nn_name = conf_to_names(sub_config)
                cursor = conn.cursor()

                # Get or add NN model ID
                cursor.execute("SELECT id FROM nn WHERE name = ?", (nn_name,))
                nn_id = cursor.fetchone()

                if not nn_id:
                    print(f"Model {nn_name} not found in `nn` table. Checking for new models in the dataset directory.")
                    nn_directory = Path(Const.dataset_dir_global)
                    nn_file = nn_directory / f"{nn_name}.py"

                    if nn_file.exists():
                        print(f"New NN model {nn_name} found. Adding to the `nn` table.")
                        with nn_file.open('r', encoding='utf-8') as file:
                            model_code = file.read()
                        nn_id = str(uuid.uuid4())  # Generate a new UUID for the model
                        cursor.execute("INSERT OR IGNORE INTO nn (id, name, code) VALUES (?, ?, ?)",
                                    (nn_id, nn_name, model_code))
                        conn.commit()
                    else:
                        print(f"NN model {nn_name} not found in the dataset directory. Skipping database save.")
                        continue
                else:
                    nn_id = nn_id[0]  # Extract the actual ID

                # Insert statistics into the `stat` table
                try:
                    trial_time = trial.get('time', None)  # Default to None if 'time' is missing
                    trial_uuid = str(uuid.uuid4())  # Generate a unique UUID for the trial

                    conn.execute("""
                    INSERT INTO stat (id, task, dataset, metric, nn_id, accuracy, batch, lr, momentum, transform, epoch, time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trial_uuid,  # UUID for the stat record
                        task, dataset, metric, nn_id, trial["accuracy"], trial["batch"],
                        trial["lr"], trial["momentum"], trial["transform"], epoch, trial_time
                    ))
                except Exception as e:
                    print(f"Error inserting trial for {sub_config}: {e}")

    conn.commit()
    print("All statistics reloaded successfully.")



def save_results(config: str, model_stat_file: str, prm: dict):
    """
    Save Optuna study results for a given model in JSON-format.
    :param config: Config (Task, Dataset, Metric, and Model name).
    :param model_stat_file: File for the model statistics.
    :param prm: Dictionary of all saved parameters.
    """

    # Extract task, dataset, metric, and nn_name from the config
    try:
        task, dataset, metric, nn_name = config.split('-')
    except ValueError:
        print(f"Invalid config format: {config}.")
        return
    
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

    # Get or add NN model ID
    cursor.execute("SELECT id FROM nn WHERE name = ?", (nn_name,))
    nn_id = cursor.fetchone()
    nn_id = nn_id[0] 

    # Insert each trial into the database with epoch
    for trial in trials_dict:
        stat_id = str(uuid.uuid4())
        trial_time = trial.get('time', None)  # Default to None if 'time' is missing
        cursor.execute("""
        INSERT INTO stat (id, task, dataset, metric, nn_id, accuracy, batch, lr, momentum, transform, epoch, time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (stat_id, task, dataset, metric, nn_id, trial['accuracy'], trial['batch'], trial['lr'],
              trial['momentum'], trial['transform'], trial['epoch'], trial_time))

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
