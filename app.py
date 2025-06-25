from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from db import query_db

app = Flask(__name__)

# Helper: Get BTC news
def get_btc_news_from_wikipedia(n=5):
    url = "https://en.wikipedia.org/wiki/Portal:Current_events"
    response = requests.get(url)
    if response.status_code != 200:
        return ["⚠️ Failed to fetch news. Try again later."]
    html = response.text
    items = html.split("<li>")
    btc_news = []
    for item in items[1:]:
        text = item.split("</li>")[0]
        while '<' in text and '>' in text:
            start = text.find('<')
            end = text.find('>', start)
            if start != -1 and end != -1:
                text = text[:start] + text[end + 1:]
            else:
                break
        clean_text = text.strip()
        if any(k in clean_text.lower() for k in ['bitcoin', 'btc', 'crypto']):
            btc_news.append(clean_text)
        if len(btc_news) >= n:
            break
    return btc_news or ["No recent BTC news found."]

# Helper: Chart rendering
def generate_price_chart(btc_df, chart_type):
    if chart_type == "Line":
        fig = px.line(btc_df, x="Date", y="Close BTC-USD", title="BTC/USD Close Price", template="plotly_dark")
        fig.add_scatter(x=btc_df["Date"], y=btc_df["EMA 20"], mode="lines", name="EMA 20", line=dict(color="cyan"))
        fig.add_scatter(x=btc_df["Date"], y=btc_df["SMA 50"], mode="lines", name="SMA 50", line=dict(color="orange"))
    else:
        fig = go.Figure(data=[go.Candlestick(
            x=btc_df["Date"],
            open=btc_df["Open BTC-USD"], high=btc_df["High BTC-USD"],
            low=btc_df["Low BTC-USD"], close=btc_df["Close BTC-USD"],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        fig.add_trace(go.Scatter(x=btc_df["Date"], y=btc_df["EMA 20"], mode='lines', name='EMA 20', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=btc_df["Date"], y=btc_df["SMA 50"], mode='lines', name='SMA 50', line=dict(color='orange')))
        fig.update_layout(xaxis_rangeslider_visible=False)
    return fig.to_html(full_html=False)

# --- Main Route ---
@app.route("/", methods=["GET"])
def index():
    symbol = request.args.get("symbol", default="BINANCE:BTCUSDT")

    btc_df = query_db("SELECT * FROM btc_usd")

    if isinstance(btc_df, str) or btc_df is None or btc_df.empty:
        return render_template("index.html", error="⚠️ Failed to load BTC data.", summary=None, news=[], chart=None, symbol=symbol)

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

    symbol = request.args.get("symbol", default="BINANCE:BTCUSDT")

    chart_html = generate_price_chart(btc_df, chart_type="Line")
    news = get_btc_news_from_wikipedia()

    return render_template("index.html", chart=chart_html, summary=summary, news=news, symbol=symbol)

if __name__ == "__main__":
    app.run(debug=True)
