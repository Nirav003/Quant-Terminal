import requests
import time
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# === Strategy Config ===
FAST_LENGTH = 9
SLOW_LENGTH = 21
AVG_LENGTH = 200
ATR_PERIOD = 14
SL_MULTIPLIER = 0.69
TP_MULTIPLIER = 1.25

LOT_SIZE = 0.02  # For PnL calculation (Exness micro lot)

def fetch_btc_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100"
    try:
        response = requests.get(url)
        raw = response.json()
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_", "_", "_", "_", "_", "_"
        ])
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print("Fetch error:", e)
        return pd.DataFrame()

def get_signal(df):
    if df.empty or len(df) < max(FAST_LENGTH, SLOW_LENGTH, AVG_LENGTH, ATR_PERIOD):
        return None

    df["fast_ema"] = EMAIndicator(df["close"], window=FAST_LENGTH).ema_indicator()
    df["slow_ema"] = EMAIndicator(df["close"], window=SLOW_LENGTH).ema_indicator()
    df["avg_ema"] = EMAIndicator(df["close"], window=AVG_LENGTH).ema_indicator()
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=ATR_PERIOD).average_true_range()

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    buy_signal = prev["fast_ema"] < prev["slow_ema"] and latest["fast_ema"] > latest["slow_ema"]
    sell_signal = prev["fast_ema"] > prev["slow_ema"] and latest["fast_ema"] < latest["slow_ema"]

    signal = "BUY" if buy_signal else "SELL" if sell_signal else None

    if signal:
        entry = latest["close"]
        atr = latest["atr"]
        if signal == "BUY":
            sl = entry - atr * SL_MULTIPLIER
            tp = entry + atr * TP_MULTIPLIER
        else:
            sl = entry + atr * SL_MULTIPLIER
            tp = entry - atr * TP_MULTIPLIER

        pip_value = 1  # 1 pip = $1 for BTCUSD per lot
        pnl = round(abs(tp - entry) * pip_value * 100000 * LOT_SIZE / 10000, 2)

        return {
            "signal": signal,
            "entry": round(entry, 2),
            "tp": round(tp, 2),
            "sl": round(sl, 2),
            "pnl": pnl
        }

    return None
