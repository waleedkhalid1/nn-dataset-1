import sqlite3
from os import makedirs
from pathlib import Path

from ab.nn.util.Const import db_file, db_dir, main_tables, code_tables, dependent_columns, all_tables, index_colum


def create_code_table(name, cursor):
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {name} (
        name TEXT PRIMARY KEY,
        code TEXT NOT NULL)""")


def create_param_table(name, cursor):
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {name} (
        uid TEXT NOT NULL,
        name TEXT NOT NULL,
        value TEXT NOT NULL,
        type TEXT NOT NULL)""")


def sql_conn():
    conn = sqlite3.connect(db_file)
    return conn, conn.cursor()


def close_conn(conn):
    conn.commit()
    conn.close()


def init_db():
    """
    Initialize the SQLite database, create tables, and add indexes for optimized reads.
    """
    makedirs(Path(db_dir).absolute(), exist_ok=True)
    conn, cursor = sql_conn()

    # Create all tables with code
    for name in code_tables:
        create_code_table(name, cursor)

    create_param_table('prm', cursor)

    # Create main stat tables
    for nm in main_tables:
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {nm} (
            id TEXT PRIMARY KEY,
            accuracy REAL,
            epoch INTEGER,
            duration INTEGER,
            {', '.join(index_colum)},         
        """ + ',\n'.join([f"FOREIGN KEY ({nm}) REFERENCES {nm} (name) ON DELETE CASCADE" for nm in dependent_columns]) + ')')

    # Add indexes for optimized reads
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accuracy_desc ON stat (accuracy DESC);")
    for nm in index_colum:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{nm} ON stat ({nm});")
    close_conn(conn)
    print(f"Database initialized at {db_file}")


def reset_db():
    """
    Clear the database and reload all NN models and statistics.
    """
    makedirs(Path(db_dir).absolute(), exist_ok=True)
    print(f"Clearing and reloading database at {db_file}")
    conn, cursor = sql_conn()

    # Drop existing tables
    for nm in all_tables:
        cursor.execute(f"DROP TABLE IF EXISTS {nm}")
    close_conn(conn)
    init_db()
