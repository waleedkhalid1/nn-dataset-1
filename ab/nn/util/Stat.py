import json
import os
import sqlite3
import uuid
from os import listdir, makedirs

import pandas as pd

from ab.nn.util.Util import *


def count_trials_left(trial_file, model_name, n_optuna_trials):
    """
    Calculates the remaining Optuna trials based on the completed ones. Checks for a "trials.json" file in the
    specified directory to determine how many trials have been completed, and returns the number of trials left.
    :param trial_file: Trial file path
    :param model_name: Name of the model.
    :param n_optuna_trials: The total number of Optuna trials the model should have. If negative, its absolute value represents the number of additional trials.
    :return: n_trials_left: Remaining trials.
    """
    n_passed_trials = 0
    if exists(trial_file):
        with open(trial_file, 'r') as f:
            trials = json.load(f)
            n_passed_trials = len(trials)
    n_trials_left = abs(n_optuna_trials) if n_optuna_trials < 0 else max(0, n_optuna_trials - n_passed_trials)
    if n_passed_trials > 0:
        print(f"The {model_name} passed {n_passed_trials} trials, {n_trials_left} left.")
    return n_trials_left


# todo: Request from the database unique names of all configures corresponding to config-patterns
# once the database is loaded, the function will be updated
def unique_configs(patterns) -> list[str]:
    """
    Collect models matching the given configuration prefix
    """
    all_configs = []
    for pattern in patterns:
        l = [c for c in listdir(Const.stat_dir_global) if c.startswith(pattern)]
        if not l and is_full_config(pattern):
            makedirs(join(Const.stat_dir_global, patterns))
        all_configs = all_configs + l
    all_configs: list = list(set(all_configs))
    return all_configs


def initialize_database():
    """
    Initialize the SQLite database, create tables, and add indexes for optimized reads.
    """
    makedirs(Path(Const.db_dir_global).parent.absolute(), exist_ok=True)
    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Create `nn` table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS nn (
        id TEXT PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        code TEXT NOT NULL
    )
    """)

    # Create `transform` table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transform (
        id TEXT PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        code TEXT NOT NULL
    )
    """)

    # Create `metric` table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metric (
        id TEXT PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        code TEXT NOT NULL
    )
    """)

    # Create `stat` table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stat (
        id TEXT PRIMARY KEY,
        task TEXT NOT NULL,
        dataset TEXT NOT NULL,
        metric_id TEXT NOT NULL,
        nn_id TEXT NOT NULL,
        transform_id TEXT NOT NULL,
        accuracy REAL,
        batch INTEGER,
        lr REAL,
        momentum REAL,
        epoch INTEGER,
        time INTEGER,
        FOREIGN KEY (metric_id) REFERENCES metric (id) ON DELETE CASCADE,
        FOREIGN KEY (nn_id) REFERENCES nn (id) ON DELETE CASCADE,
        FOREIGN KEY (transform_id) REFERENCES transform (id) ON DELETE CASCADE
    )
    """)

    # Add indexes for optimized reads
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_task ON stat (task);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dataset ON stat (dataset);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metric_id ON stat (metric_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nn_id ON stat (nn_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_transform_id ON stat (transform_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accuracy_desc ON stat (accuracy DESC);")

    # Populate `nn`, `transform`, and `metric` tables
    populate_nn_table(conn)
    populate_transform_table(conn)
    populate_metric_table(conn)

    conn.commit()
    conn.close()
    print(f"Database initialized at {Const.db_dir_global}")



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

def populate_transform_table(conn):
    """
    Populate the `transform` table with transform names and code.
    """
    transform_directory = Path("ab/nn/transform")
    print(f"Populating `transform` table with transforms from {transform_directory}.")
    
    transform_files = [
        f for f in transform_directory.iterdir() if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"
    ]

    cursor = conn.cursor()
    for transform_file in transform_files:
        transform_name = transform_file.stem
        with open(transform_file, "r") as f:
            code = f.read()

        # Check if the transform exists in the database
        cursor.execute("SELECT id, code FROM transform WHERE name = ?", (transform_name,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # If transform exists, update the code if it has changed
            transform_id, existing_code = existing_entry
            if existing_code != code:
                print(f"Updating code for transform: {transform_name}")
                cursor.execute("UPDATE transform SET code = ? WHERE id = ?", (code, transform_id))
        else:
            # If transform does not exist, insert it with a new UUID
            transform_id = str(uuid.uuid4())
            print(f"Adding new transform {transform_name} to the `transform` table with UUID: {transform_id}")
            cursor.execute("INSERT INTO transform (id, name, code) VALUES (?, ?, ?)",
                           (transform_id, transform_name, code))

    conn.commit()
    print(f"Transforms added/updated in the `transform` table: {[f.stem for f in transform_files]}")


def populate_metric_table(conn):
    """
    Populate the `metric` table with names and code from the metric directory.
    """
    metric_directory = Path("ab/nn/metric")
    print(f"Populating `metric` table with metrics from {metric_directory}.")
    
    metric_files = [
        f for f in metric_directory.iterdir() if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"
    ]

    cursor = conn.cursor()
    for metric_file in metric_files:
        metric_name = metric_file.stem
        with open(metric_file, "r") as f:
            metric_code = f.read()

        # Check if the metric exists in the database
        cursor.execute("SELECT id, code FROM metric WHERE name = ?", (metric_name,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # If metric exists, update the code if it has changed
            metric_id, existing_code = existing_entry
            if existing_code != metric_code:
                print(f"Updating code for metric: {metric_name}")
                cursor.execute("UPDATE metric SET code = ? WHERE id = ?", (metric_code, metric_id))
        else:
            # If metric does not exist, insert it with a new UUID
            metric_id = str(uuid.uuid4())
            print(f"Adding new metric {metric_name} to the `metric` table with UUID: {metric_id}")
            cursor.execute("INSERT INTO metric (id, name, code) VALUES (?, ?, ?)",
                           (metric_id, metric_name, metric_code))

    conn.commit()
    print(f"Metrics added/updated in the `metric` table: {[f.stem for f in metric_files]}")


def clear_and_reload_database():
    """
    Clear the database and reload all NN models and statistics.
    """
    makedirs(Path(Const.db_dir_global).parent.absolute(), exist_ok=True)
    print(f"Clearing and reloading database at {Const.db_dir_global}")
    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS stat")
    cursor.execute("DROP TABLE IF EXISTS nn")
    cursor.execute("DROP TABLE IF EXISTS transform")
    cursor.execute("DROP TABLE IF EXISTS metric")
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

    for sub_config in sub_configs:
        model_stat_dir = stat_base_path / sub_config

        for epoch_file in Path(model_stat_dir).iterdir():
            model_stat_file = model_stat_dir / epoch_file
            epoch = int(epoch_file[: epoch_file.index('.')])

            with open(model_stat_file, 'r') as f:
                trials = json.load(f)

            for trial in trials:
                task, dataset, metric, nn_name = conf_to_names(sub_config)
                cursor = conn.cursor()

                # Get or add NN model ID
                cursor.execute("SELECT id FROM nn WHERE name = ?", (nn_name,))
                nn_id = cursor.fetchone()
                if not nn_id:
                    print(f"Model {nn_name} not found in `nn` table. Adding it.")
                    nn_directory = Path(Const.dataset_dir_global)
                    nn_file = nn_directory / f"{nn_name}.py"

                    if nn_file.exists():
                        with nn_file.open('r', encoding='utf-8') as file:
                            model_code = file.read()
                        nn_id = str(uuid.uuid4())
                        cursor.execute("INSERT INTO nn (id, name, code) VALUES (?, ?, ?)",
                                       (nn_id, nn_name, model_code))
                        conn.commit()
                    else:
                        print(f"NN model {nn_name} not found in dataset directory. Skipping database save.")
                        continue
                else:
                    nn_id = nn_id[0]  # Extract ID

                # Get or add Metric ID
                cursor.execute("SELECT id FROM metric WHERE name = ?", (metric,))
                metric_id = cursor.fetchone()
                if not metric_id:
                    metric_id = str(uuid.uuid4())
                    metric_code = f"ab.nn.metric.{metric}"
                    cursor.execute("INSERT INTO metric (id, name, code) VALUES (?, ?, ?)",
                                   (metric_id, metric, metric_code))
                    conn.commit()
                else:
                    metric_id = metric_id[0]

                # Get or add Transform ID
                transform_name = trial.get("transform")
                cursor.execute("SELECT id FROM transform WHERE name = ?", (transform_name,))
                transform_id = cursor.fetchone()
                if not transform_id:
                    transform_id = str(uuid.uuid4())
                    transform_code = f"ab.nn.transform.{transform_name}"
                    cursor.execute("INSERT INTO transform (id, name, code) VALUES (?, ?, ?)",
                                   (transform_id, transform_name, transform_code))
                    conn.commit()
                else:
                    transform_id = transform_id[0]

                # Insert into stat table
                try:
                    trial_uuid = str(uuid.uuid4())
                    trial_time = trial.get('time', None)
                    conn.execute("""
                    INSERT INTO stat (id, task, dataset, metric_id, nn_id, transform_id, accuracy, batch, lr, momentum, epoch, time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trial_uuid, task, dataset, metric_id, nn_id, transform_id, trial["accuracy"],
                        trial["batch"], trial["lr"], trial["momentum"], epoch, trial_time
                    ))
                except Exception as e:
                    print(f"Error inserting trial for {sub_config}, epoch {epoch}: {e}")

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
        task, dataset, metric, nn_name = conf_to_names(config)
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

    return # todo Change to align with the new functionality

    # Save results to SQLite DB
    conn = sqlite3.connect(Const.db_dir_global)
    cursor = conn.cursor()

    # Get IDs from `nn`, `transform`, and `metric` tables
    cursor.execute("SELECT id FROM nn WHERE name = ?", (nn_name,))
    nn_id = cursor.fetchone()
    if not nn_id:
        print(f"NN model {nn_name} not found. Skipping save.")
        return
    nn_id = nn_id[0]

    cursor.execute("SELECT id FROM transform WHERE name = ?", (prm["transform"],))
    transform_id = cursor.fetchone()
    if not transform_id:
        print(f"Transform {prm['transform']} not found. Skipping save.")
        return
    transform_id = transform_id[0]

    cursor.execute("SELECT id FROM metric WHERE name = ?", (metric,))
    metric_id = cursor.fetchone()
    if not metric_id:
        print(f"Metric {metric} not found. Skipping save.")
        return
    metric_id = metric_id[0]


    # Insert each trial into the database with epoch
    for trial in trials_dict:
        stat_id = str(uuid.uuid4())
        trial_time = trial.get('time', None)  # Default to None if 'time' is missing
        cursor.execute("""
        INSERT INTO stat (id, task, dataset, metric_id, nn_id, transform_id, accuracy, batch, lr, momentum, epoch, time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (stat_id, task, dataset, metric_id, nn_id, transform_id, trial['accuracy'], trial['batch'],
              trial['lr'], trial['momentum'], trial['epoch'], trial_time))

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
