import sqlite3
import pandas as pd

# Replace with your actual database file name if different
DB_FILE = "btcusd_data.db"  # e.g., "btc_data.db"

def preview_btc_data():
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_FILE)

    try:
        # Load the entire btc_usd table into a DataFrame
        df = pd.read_sql_query("SELECT * FROM btc_usd", conn)

        print("\n📌 First 5 entries:")
        print(df.head())

        print("\n📌 Last 5 entries:")
        print(df.tail())

    except Exception as e:
        print(f"❌ Error reading data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    preview_btc_data()
