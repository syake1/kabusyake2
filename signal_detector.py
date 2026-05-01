import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("株シグナル（安定版）")

# 銘柄入力
ticker = st.text_input("銘柄コード（例: 7203.T）", "7203.T")

# データ取得
@st.cache_data(ttl=300)
def load_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    return df

df = load_data(ticker)

if df.empty:
    st.error("データ取得失敗")
    st.stop()

# 指標
df['MA25'] = df['Close'].rolling(25).mean()
df['MA75'] = df['Close'].rolling(75).mean()

# MACD
exp1 = df['Close'].ewm(span=12).mean()
exp2 = df['Close'].ewm(span=26).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df['MACD'].ewm(span=9).mean()

# RSI
delta = df['Close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# ===== シグナル（絶対出る安定版） =====
df['Buy'] = False
df['Sell'] = False

df.loc[
    (df['MACD'] > df['Signal']) &
    (df['MACD'].shift(1) <= df['Signal'].shift(1)),
    'Buy'
] = True

df.loc[
    (df['MACD'] < df['Signal']) &
    (df['MACD'].shift(1) >= df['Signal'].shift(1)),
    'Sell'
] = True

# ===== チャート =====
fig = go.Figure()

# ローソク
fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="価格"
))

# MA
fig.add_trace(go.Scatter(x=df.index, y=df['MA25'], name="MA25"))
fig.add_trace(go.Scatter(x=df.index, y=df['MA75'], name="MA75"))

# シグナル
buy = df[df['Buy'] == True]
sell = df[df['Sell'] == True]

fig.add_trace(go.Scatter(
    x=buy.index,
    y=buy['Low'] * 0.98,
    mode='markers',
    marker=dict(size=12),
    name="買い"
))

fig.add_trace(go.Scatter(
    x=sell.index,
    y=sell['High'] * 1.02,
    mode='markers',
    marker=dict(size=12),
    name="売り"
))

st.plotly_chart(fig, use_container_width=True)
