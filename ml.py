# ml.py
import numpy as np
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from sqlalchemy import create_engine

DB_URL = "sqlite:///market_data.db"
engine = create_engine(DB_URL)

# === Data Fetch & Save ===
def fetch_data(symbol="BTCUSDT", interval="1m", limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
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
    df.to_sql("klines", engine, if_exists="replace")  # save latest
    return df

# === Indicators ===
def compute_indicators(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    return df.dropna()

# === Q-Learning ===
class QTrader:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1, bins=10):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.bins = bins
        self.q_table = {}
        self.actions = [0, 1, 2]  # 0 = hold, 1 = buy, 2 = sell

    def discretize(self, df_row):
        return (
            int(df_row['rsi']//(100/self.bins)),
            int((df_row['macd_hist']+5)//(10/self.bins)),
            int(((df_row['ema9']-df_row['ema21'])/df_row['ema21']*100)//2),
            int((df_row['volume']/df_row['vol_ma']*10)//1)
        )

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.get_q(state))

    def update(self, state, action, reward, next_state):
        q_current = self.get_q(state)[action]
        q_future = np.max(self.get_q(next_state))
        self.q_table[state][action] = q_current + self.alpha * (reward + self.gamma * q_future - q_current)

def train_q_agent(df, episodes=10):
    agent = QTrader()
    for _ in range(episodes):
        position = 0
        entry_price = 0
        pnl = 0
        for i in range(len(df)-1):
            state = agent.discretize(df.iloc[i])
            action = agent.choose_action(state)

            reward = 0
            if action == 1 and position == 0:  # buy
                position = 1
                entry_price = df['close'].iloc[i]
            elif action == 2 and position == 1:  # sell
                pnl = df['close'].iloc[i] - entry_price
                reward = pnl
                position = 0

            next_state = agent.discretize(df.iloc[i+1])
            agent.update(state, action, reward, next_state)
    return agent

def backtest(agent, df):
    position = 0
    entry_price = 0
    trades = []
    pnl = 0
    for i in range(len(df)-1):
        state = agent.discretize(df.iloc[i])
        action = np.argmax(agent.get_q(state))
        if action == 1 and position == 0:  # buy
            position = 1
            entry_price = df['close'].iloc[i]
            trades.append((df.index[i], "BUY", entry_price))
        elif action == 2 and position == 1:  # sell
            exit_price = df['close'].iloc[i]
            pnl += exit_price - entry_price
            trades.append((df.index[i], "SELL", exit_price))
            position = 0
    return {"trades": trades, "pnl": pnl}
