from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from db import query_db
import yfinance as yf
import numpy as np
import time
import re

app = Flask(__name__)

COINS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "dogecoin": "DOGE",
    "litecoin": "LTC",
    "solana": "SOL",
    "pepe": "PEPE",
    "binancecoin": "BNB"
}

FOREX = {
    "eur": "EUR",
    "jpy": "JPY",
    "usd": "USD"
}

# ✅ Summary
def summary():
    btc_df = query_db("SELECT * FROM btc_usd")

    # Technical indicators
    btc_df["EMA 20"] = btc_df["Close BTC-USD"].ewm(span=20).mean()
    btc_df["SMA 50"] = btc_df["Close BTC-USD"].rolling(window=50).mean()
    btc_df["Volatility 7d"] = btc_df["Return (%)"].rolling(window=7).std()

    delta = btc_df["Close BTC-USD"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    btc_df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = btc_df["Close BTC-USD"].ewm(span=12, adjust=False).mean()
    exp2 = btc_df["Close BTC-USD"].ewm(span=26, adjust=False).mean()
    btc_df["MACD"] = exp1 - exp2
    btc_df["MACD Signal"] = btc_df["MACD"].ewm(span=9, adjust=False).mean()

    trend = lambda df, m: "Bullish" if df["Close BTC-USD"].tail(m).pct_change().sum() > 0 else "Bearish"

    summary = {
        "volatility": round(btc_df["Volatility 7d"].iloc[-1], 2),
        "sharpe": round(btc_df["Return (%)"].mean() / btc_df["Return (%)"].std(), 2),
        "return": round(btc_df["Return (%)"].iloc[-1], 2),
        "volume": int(btc_df["Volume BTC-USD"].iloc[-1]),
        "trend_1m": trend(btc_df, 1),
        "trend_5m": trend(btc_df, 5),
        "trend_15m": trend(btc_df, 15),
        "rsi": round(btc_df["RSI"].iloc[-1], 2),
        "macd": round(btc_df["MACD"].iloc[-1], 2),
        "macd_signal": round(btc_df["MACD Signal"].iloc[-1], 2),
    }

    return summary

def fetch_crypto_news():
    try:
        url = "https://en.wikipedia.org/wiki/Portal:Cryptocurrency"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        html = res.text

        # Match article headlines inside <a> tags with class name that includes "card-title"
        headlines = re.findall(r'<a[^>]+class="[^"]*card-title[^"]*"[^>]*>(.*?)</a>', html)
        headlines = [re.sub("<[^<]+?>", "", h).strip() for h in headlines if h.strip()]
        
        top_headlines = headlines[:5]
        news = "   •   ".join(top_headlines) if top_headlines else "No headlines found."
        return news
    
    except Exception as e:
        return f"⚠️ Error fetching news: {e}"

@app.route("/automate")
def automate():
    return render_template("automate.html", summary=summary())
    # Add MTF code here

@app.route("/backtest")
def backtest():
    return render_template("backtest.html", summary=summary())
    # Create a backtesting page with MTF code here

@app.route("/logs")
def logs():
    return render_template("logs.html", summary=summary())
    # Create a logs with SQL query here

@app.route("/settings")  # Replace with your actual third template if different
def settings():
    return render_template("settings.html", summary=summary())
    # 

@app.route("/analysis")
def analysis():
    return render_template("analysis.html", summary=summary())
    # 

# ✅ Main route
@app.route("/", methods=["GET"])
def index():
    symbol = request.args.get("symbol", default="BINANCE:BTCUSDT")


    try:
        coin_ids = ",".join(COINS.keys())
        forex_ids = ",".join(FOREX.keys())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_ids}&vs_currencies={forex_ids}"
        res = requests.get(url)
        res.raise_for_status()
        prices = res.json()

        ticker_items = []
        for coin_id, symbol in COINS.items():
            price = prices.get(coin_id, {}).get('usd', 'N/A')
            ticker_items.append(f"{symbol}/USD: ${price:,}")
        ticker_text = "   •   ".join(ticker_items)

    except Exception as e:
        ticker_text = f"Error fetching prices: {e}"

    # News from CoinGecko (Status Updates)
    try:
        news_url = "https://api.coingecko.com/api/v1/status_updates"
        news_res = requests.get(news_url)
        news_res.raise_for_status()
        news_data = news_res.json()

        # Extract top 5 news titles
        news_titles = [item["title"] for item in news_data.get("status_updates", [])[:5]]
        news_marquee_text = "   •   ".join(news_titles)

    except Exception as e:
        news_marquee_text = f"Error fetching news: {e}"


    return render_template(
    "index.html",
    summary=summary(),
    symbol=symbol,
    ticker_text=ticker_text,
    news_marquee_text=fetch_crypto_news()
)
    
# ✅ Run app
if __name__ == "__main__":
    app.run(debug=True)
