import yfinance as yf
import pandas as pd
import sqlite3
import json

# Load config
with open("config.json") as f:
    config = json.load(f)

DB_NAME = "data.db"
START_DATE = config.get("start_date", "2015-01-01")

def fetch_data(ticker, start):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start, interval="1d")
    df.reset_index(inplace=True)
    return df

def store_data_in_db(df, table_name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume REAL
        )
    """)

    df.columns = [col.lower() for col in df.columns]
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    print(f"✅ Stored {len(df)} rows in '{table_name}'.")

def export_to_json(table_name):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY date ASC", conn)
    conn.close()
    df.to_json(f"{table_name}.json", orient="records", date_format="iso")
    print(f"📁 Exported data to {table_name}.json")


for asset in config["assets"]:
    ...
    store_data_in_db(df, table_name)
    export_to_json(table_name)

def main():
    for asset in config["assets"]:
        ticker = asset["ticker"]
        table_name = asset["table_name"]
        df = fetch_data(ticker, START_DATE)
        store_data_in_db(df, table_name)

if _name_ == "_main_":
    main()