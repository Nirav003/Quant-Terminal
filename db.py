
import sqlite3
import pandas as pd

DB_NAME = "btcusd_data.db"

def create_connection():
    return sqlite3.connect(DB_NAME)

def save_dataframe_to_db(df, table_name):
    with create_connection() as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=True)

def query_db(sql):
    with create_connection() as conn:
        try:
            df = pd.read_sql_query(sql, conn)
            return df
        except Exception as e:
            return str(e)
