from os import listdir

from ab.nn.util.Util import *
import sqlite3


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


def initialize_database(db_path="results.db"):
    """
    Initialize the SQLite database and create the `results` table if it doesn't exist.
    :param db_path: Path to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
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
