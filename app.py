from flask import Flask, render_template, request, Blueprint, jsonify
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import yfinance as yf
import numpy as np
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
import time
import re
import feedparser
from db import query_db
from difflib import get_close_matches

app = Flask(__name__)
SYMBOL_MAP = {
    "xauusd": "OANDA:XAUUSD",
    "btcusd": "BINANCE:BTCUSD",
    "ethusd": "BINANCE:ETHUSD",
    "eurjpy": "OANDA:EURJPY",
    "usoil": "TVC:USOIL",
    "solana": "BINANCE:SOLUSD",
    "eurusd": "OANDA:EURUSD",
}


# ✅ Finance news feed
RSS_FEED = "https://www.fxstreet.com/rss/news"
def index():
    selected_symbol = "BINANCE:BTCUSD"  # default

    if request.method == "POST":
        symbol = request.form.get("symbol")
        custom = request.form.get("custom_symbol", "").strip().lower()

        if custom:
            match = get_close_matches(custom.replace("/", ""), SYMBOL_MAP.keys(), n=1, cutoff=0.4)
            if match:
                selected_symbol = SYMBOL_MAP[match[0]]
            else:
                selected_symbol = "BINANCE:BTCUSD"  # fallback if no match
        elif symbol:
            selected_symbol = symbol
def get_price(symbol):
    symbol = symbol.upper()

    PRICE_MAP = {
        "XAUUSD": ("OANDA", "xauusd"),
        "BTCUSD": ("BINANCE", "bitcoin"),
        "ETHUSD": ("BINANCE", "ethereum"),
    }

    try:
        if symbol in PRICE_MAP:
            _, coin_id = PRICE_MAP[symbol]
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
            res = requests.get(url)
            res.raise_for_status()
            price = res.json()[coin_id]['usd']
            return jsonify({"symbol": symbol, "price": f"${price:,.2f}"})
        else:
            return jsonify({"error": "Symbol not supported"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_finance_news():
    feed = feedparser.parse(RSS_FEED)
    if feed.bozo:
        return [f"⚠️ Error parsing RSS: {feed.bozo_exception}"]
    titles = [entry.title for entry in feed.entries[:5]]
    return titles if titles else ["No news found."]

# ✅ Coin & Forex config
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

# ✅ Summary calculation
def summary():
    btc_df = query_db("SELECT * FROM btc_usd")

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

    return {
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

# ✅ Routes
@app.route("/", methods=["GET", "POST"])
def index():
    default_symbol = "BINANCE:BTCUSD"

    # Get selected symbol from form
    symbol = request.form.get("symbol", default_symbol)

    try:
        coin_ids = ",".join(COINS.keys())
        forex_ids = ",".join(FOREX.keys())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_ids}&vs_currencies={forex_ids}"
        res = requests.get(url)
        res.raise_for_status()
        prices = res.json()

        ticker_items = []
        for coin_id, sym in COINS.items():
            price = prices.get(coin_id, {}).get('usd', 'N/A')
            ticker_items.append(f"{sym}/USD: ${price:,}")
        ticker_text = "   |   ".join(ticker_items)

    except Exception as e:
        ticker_text = f"Error fetching prices: {e}"

    # ✅ Use Yahoo Finance RSS instead of Wikipedia if preferred:
    rss_headlines = get_finance_news()
    news_marquee_text = "   |   ".join(rss_headlines)

    return render_template(
        "index.html",
        summary=summary(),
        symbol=symbol,
        ticker_text=ticker_text,
        news_marquee_text=news_marquee_text  # Or: fetch_crypto_news()
    )
# === Strategy Settings ===
fast_length = 9
slow_length = 21
avg_length = 200
atr_period = 14
sl_multiplier = 0.69
tp_multiplier = 1.25
lot_size = 0.02

# === Global Trade State ===
active_trade = None
trade_announced = False
last_update = {'status': 'Waiting for trade ...'}

def get_data():
    url = 'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=200'
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

def apply_indicators(df):
    df['fast_ema'] = EMAIndicator(df['close'], window=fast_length).ema_indicator()
    df['slow_ema'] = EMAIndicator(df['close'], window=slow_length).ema_indicator()
    df['avg_ema'] = EMAIndicator(df['close'], window=avg_length).ema_indicator()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=atr_period).average_true_range()
    return df

def get_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if prev['fast_ema'] < prev['slow_ema'] and last['fast_ema'] > last['slow_ema']:
        return 'buy'
    elif prev['fast_ema'] > prev['slow_ema'] and last['fast_ema'] < last['slow_ema']:
        return 'sell'
    return None

def calculate_trade(signal, price, atr):
    if signal == 'buy':
        sl = price - atr * sl_multiplier
        tp = price + atr * tp_multiplier
    else:
        sl = price + atr * sl_multiplier
        tp = price - atr * tp_multiplier
    return {'type': signal, 'entry': price, 'sl': sl, 'tp': tp}

def monitor_trade(trade, price):
    if trade['type'] == 'buy':
        if price >= trade['tp']:
            return 'TP'
        elif price <= trade['sl']:
            return 'SL'
    elif trade['type'] == 'sell':
        if price <= trade['tp']:
            return 'TP'
        elif price >= trade['sl']:
            return 'SL'
    return None

def calculate_pnl(entry, exit_price):
    pips = abs(exit_price - entry) * 10
    pnl = pips * lot_size
    return round(pips, 1), round(pnl, 2)

@app.route('/')
def home():
    return render_template('automate.html')

@app.route('/data')
def signal_data():
    global active_trade, trade_announced, last_update

    try:
        df = get_data()
        df = apply_indicators(df)
        signal = get_signal(df)
        latest = df.iloc[-1]
        current_price = latest['close']
        atr = latest['atr']

        if not active_trade:
            if signal:
                active_trade = calculate_trade(signal, current_price, atr)
                trade_announced = True
                last_update = {
                    'status': f"{signal.upper()} SIGNAL",
                    'entry': round(active_trade['entry'], 2),
                    'tp': round(active_trade['tp'], 2),
                    'sl': round(active_trade['sl'], 2),
                    'current': round(current_price, 2),
                    'pips': '-',
                    'pnl': '-'
                }
            else:
                last_update = {'status': 'Waiting for trade ...'}

        else:
            result = monitor_trade(active_trade, current_price)
            if result:
                pips, pnl = calculate_pnl(active_trade['entry'], current_price)
                last_update = {
                    'status': f"{result} HIT",
                    'entry': round(active_trade['entry'], 2),
                    'tp': round(active_trade['tp'], 2),
                    'sl': round(active_trade['sl'], 2),
                    'current': round(current_price, 2),
                    'pips': pips,
                    'pnl': f"${pnl}"
                }
                active_trade = None
            else:
                last_update['current'] = round(current_price, 2)

    except Exception as e:
        last_update = {'status': f"Error: {e}"}

    return jsonify(last_update)


@app.route("/backtest")
def backtest():
    return render_template("backtest.html", summary=summary())

@app.route("/logs")
def logs():
    return render_template("logs.html", summary=summary())

@app.route("/settings")
def settings():
    return render_template("settings.html", summary=summary())

@app.route("/analysis")
def analysis():
    return render_template("analysis.html", summary=summary())

@app.route('/automate')
def automate_page():
    return render_template('automate.html')

# ✅ Run
if __name__ == "__main__":
    app.run(debug=True)
