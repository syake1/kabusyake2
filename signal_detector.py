import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(layout="wide")

st.title("日本株スイングスキャナー")

# ===== 銘柄リスト（増やせる） =====
tickers = {
    "トヨタ": "7203.T",
    "ソニー": "6758.T",
    "三菱UFJ": "8306.T",
    "任天堂": "7974.T",
    "ソフトバンク": "9984.T",
    "キーエンス": "6861.T",
    "東京エレクトロン": "8035.T"
}

# ===== データ取得 =====
@st.cache_data(ttl=300)
def load_data(ticker):
    return yf.download(ticker, period="3mo", interval="1d", progress=False)

# ===== シグナル判定（軽量＆確実） =====
def get_signal(df):
    if len(df) < 30:
        return "なし"

    macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()

    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
        return "買い"
    elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
        return "売り"
    else:
        return "なし"

# ===== スキャン =====
results = []

for name, code in tickers.items():
    try:
        df = load_data(code)
        if df.empty:
            continue

        price = df['Close'].iloc[-1]
        change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        signal = get_signal(df)

        results.append({
            "銘柄": name,
            "コード": code,
            "価格": round(price, 1),
            "騰落率%": round(change, 2),
            "シグナル": signal
        })
    except:
        continue

df_result = pd.DataFrame(results)

# ===== ランキング =====
df_result = df_result.sort_values(by="騰落率%", ascending=False)

st.subheader("ランキング（強い順）")
st.dataframe(df_result, use_container_width=True)

# ===== 買い銘柄だけ =====
st.subheader("今の買い候補")
buy_df = df_result[df_result["シグナル"] == "買い"]

if buy_df.empty:
    st.write("なし")
else:
    st.dataframe(buy_df, use_container_width=True)
