import yfinance as yf
import pandas as pd
import numpy as np
from db import save_dataframe_to_db

def fetch_btc_data(start="2020-03-01", end="2025-06-14"):
    print(f"Fetching BTC data from {start} to {end}")
    btc = yf.download("BTC-USD", start=start, end=end, interval="1d")

    if btc.empty:
        raise Exception("BTC data fetch failed or returned empty.")

    print("Available columns before filtering:", btc.columns.tolist())
    
    # Safe column selection
    columns_to_use = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in btc.columns]
    btc = btc[columns_to_use]

    # Add calculated features
    btc["Return (%)"] = btc["Close"].pct_change() * 100
    btc["Log Return"] = np.log(btc["Close"] / btc["Close"].shift(1)).replace([np.inf, -np.inf], np.nan)
    btc.dropna(inplace=True)

    # Reset index and rename 'Date' column
    btc = btc.reset_index()

    # Flatten MultiIndex if present
    btc.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in btc.columns]

    print("Flattened columns:", btc.columns.tolist())
    return btc

def update_database():
    data = fetch_btc_data()
    save_dataframe_to_db(data, "btc_usd")
    print("BTC data saved to database.")

if __name__ == "__main__":
    update_database()