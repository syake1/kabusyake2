import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import os

# ファイルパスの設定
TICKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tickers.json")
if os.path.exists(TICKER_FILE):
    with open(TICKER_FILE, "r", encoding="utf-8") as f:
        global_tickers = json.load(f)
else:
    global_tickers = {
        "半導体": {"東京エレクトロン": "8035.T"}
    }

def jump_to_ticker_action(name, code):
    target_sec = "カスタム（直接入力）"
    target_name = name
    for sec, t_dict in global_tickers.items():
        for t_name, t_code in t_dict.items():
            if t_code == code:
                target_sec = sec
                target_name = t_name
                break
    
    st.session_state['sector_choice'] = target_sec
    if target_sec != "カスタム（直接入力）":
        st.session_state['ticker_choice'] = target_name
    else:
        st.session_state['custom_ticker_input'] = code

# --- 指標計算ロジック ---
def calculate_ma(df, windows=[5, 25, 75, 200]):
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_bollinger_bands(df, window=20, num_std=2.0):
    df['BB_Mid'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * num_std)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * num_std)
    return df

def calculate_macd(df, fast=12, slow=26, signal=9):
    df['MACD_Fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['MACD_Slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['MACD_Fast'] - df['MACD_Slow']
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, min_periods=window).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/window, min_periods=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_ichimoku(df):
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['Tenkan'] = (high_9 + low_9) / 2
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['Kijun'] = (high_26 + low_26) / 2
    df['Senkou_A'] = ((df['Tenkan'] + df['Kijun']) / 2).shift(26)
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
    df['Chikou'] = df['Close'].shift(-26)
    return df

def calculate_stochastic(df, window=14, smooth_k=3):
    low_min = df['Low'].rolling(window=window).min()
    high_max = df['High'].rolling(window=window).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=smooth_k).mean()
    return df

def calculate_dmi(df, window=14):
    high_diff = df['High'] - df['High'].shift(1)
    low_diff = df['Low'].shift(1) - df['Low']
    plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=df.index)
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=df.index)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    smooth_tr = tr.ewm(alpha=1/window, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=1/window, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=1/window, adjust=False).mean()
    df['Plus_DI'] = 100 * (smooth_plus_dm / smooth_tr)
    df['Minus_DI'] = 100 * (smooth_minus_dm / smooth_tr)
    dx = 100 * (df['Plus_DI'] - df['Minus_DI']).abs() / (df['Plus_DI'] + df['Minus_DI'])
    df['ADX'] = dx.ewm(alpha=1/window, adjust=False).mean()
    return df

def add_all_indicators(df, bb_std=2.0):
    df = calculate_ma(df)
    df = calculate_bollinger_bands(df, num_std=bb_std)
    df = calculate_macd(df)
    df = calculate_rsi(df)
    df = calculate_ichimoku(df)
    df = calculate_stochastic(df)
    df = calculate_dmi(df)
    return df

# --- 共通データ取得・計算用ユーティリティ ---
@st.cache_data(ttl=300)
def load_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.dropna(how='all', inplace=True)
            if pd.api.types.is_datetime64_any_dtype(df.index):
                if df.index.tz is None:
                    df.index = df.index.tz_localize('Asia/Tokyo')
                else:
                    df.index = df.index.tz_convert('Asia/Tokyo')
        return df
    except Exception:
        return pd.DataFrame()

def get_latest_signal_info(ticker, period, timeframe, bb_std, rsi_overbought, rsi_oversold, sensitivity, trend_filter, dmi_filter):
    df_mini = load_data(ticker, period, timeframe)
    if not df_mini.empty and len(df_mini) >= 50:
        df_mini = add_all_indicators(df_mini, bb_std=bb_std)
        df_mini = detect_signals(df_mini, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold, 
                                 sensitivity=sensitivity, trend_filter=trend_filter, dmi_filter=dmi_filter)
        latest = df_mini.iloc[-1]
        prev = df_mini.iloc[-2]
        change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        signal = "➖"
        if latest['Buy_Signal']: signal = "🟢 買い"
        elif latest['Sell_Signal']: signal = "🔴 売り"
        return {
            "価格": f"{latest['Close']:,.1f}",
            "騰落率": f"{change:+.2f}%",
            "RSI": f"{latest['RSI']:.1f}",
            "シグナル": signal
        }
    return None

# --- シグナル判定ロジック ---
def detect_signals(df, rsi_overbought=70, rsi_oversold=30, sensitivity="標準", trend_filter=True, dmi_filter=False):
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    
    if dmi_filter:
        dmi_uptrend = (df['Plus_DI'] > df['Minus_DI']) & (df['ADX'] > 20)
        dmi_downtrend = (df['Minus_DI'] > df['Plus_DI']) & (df['ADX'] > 20)
    else:
        dmi_uptrend = pd.Series(True, index=df.index)
        dmi_downtrend = pd.Series(True, index=df.index)

    if len(df) < 50: 
        return df

    prev_macd = df['MACD'].shift(1)
    prev_signal = df['MACD_Signal'].shift(1)
    prev_rsi = df['RSI'].shift(1)
    prev_hist = df['MACD_Hist'].shift(1)
    prev_stoch_k = df['Stoch_K'].shift(1)
    prev_stoch_d = df['Stoch_D'].shift(1)

    trend_col = 'MA_75' if not df['MA_75'].isna().all() else 'MA_25'
    if trend_filter:
        uptrend = df['Close'] > df[trend_col]
    else:
        uptrend = pd.Series(True, index=df.index)

    macd_gc = (prev_macd <= prev_signal) & (df['MACD'] > df['MACD_Signal'])
    macd_dc = (prev_macd >= prev_signal) & (df['MACD'] < df['MACD_Signal'])
    macd_improving = df['MACD_Hist'] > prev_hist
    macd_worsening = df['MACD_Hist'] < prev_hist
    stoch_gc = (prev_stoch_k <= prev_stoch_d) & (df['Stoch_K'] > df['Stoch_D'])
    stoch_dc = (prev_stoch_k >= prev_stoch_d) & (df['Stoch_K'] < df['Stoch_D'])
    rsi_rebound = (prev_rsi <= rsi_oversold + 10) & (df['RSI'] > prev_rsi)
    rsi_drop = (prev_rsi >= rsi_overbought - 10) & (df['RSI'] < prev_rsi)
    
    if sensitivity == "敏感 (シグナル多)":
        not_overbought = (df['Stoch_K'] < 70) & (df['RSI'] < 65)
        df.loc[uptrend & dmi_uptrend & (macd_gc | (stoch_gc & (df['Stoch_K'] < 50)) | rsi_rebound) & ~macd_dc & not_overbought, 'Buy_Signal'] = True
        df.loc[dmi_downtrend & (macd_dc | (stoch_dc & (df['Stoch_K'] > 50)) | rsi_drop) & ~macd_gc & macd_worsening, 'Sell_Signal'] = True
    else:
        near_25ma = (df['Close'] - df['MA_25']).abs() / df['MA_25'] < 0.08
        not_overbought = (df['Stoch_K'] < 70) & (df['RSI'] < 65)
        df.loc[uptrend & dmi_uptrend & near_25ma & (macd_gc | (stoch_gc & (df['Stoch_K'] < 40))) & ~macd_dc & macd_improving & not_overbought, 'Buy_Signal'] = True
        touch_upper_bb = df['High'] >= df['BB_Upper']
        df.loc[dmi_downtrend & ((df['RSI'] > rsi_overbought) | touch_upper_bb | (stoch_dc & (df['Stoch_K'] > 70))) & ~macd_gc & macd_worsening, 'Sell_Signal'] = True

    return df

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Signal Detector Pro", layout="wide")

def check_password():
    if "password" not in st.secrets:
        return True
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.warning("🔒 このアプリは保護されています。")
        st.text_input("🔑 パスワードを入力してEnterを押してください", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.warning("🔒 このアプリは保護されています。")
        st.text_input("🔑 パスワードを入力してEnterを押してください", type="password", on_change=password_entered, key="password")
        st.error("😕 パスワードが間違っています。")
        return False
    return True

if not check_password():
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
banner_path = os.path.join(BASE_DIR, "banner.jpg")
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)

col1, col2 = st.columns([1, 15])
with col1:
    logo_path = os.path.join(BASE_DIR, "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
with col2:
    st.title("日本株・為替 シグナル判定＆可視化システム")

# サイドバー設定
st.sidebar.header("1. 銘柄設定")
tickers = global_tickers
sectors = list(tickers.keys()) + ["カスタム（直接入力）"]
if 'sector_choice' not in st.session_state:
    st.session_state['sector_choice'] = sectors[0]
sector = st.sidebar.selectbox("セクター/カテゴリ", sectors, key="sector_choice")

if sector == "カスタム（直接入力）":
    if 'custom_ticker_input' not in st.session_state:
        st.session_state['custom_ticker_input'] = "9984.T"
    ticker_symbol = st.sidebar.text_input("銘柄コードを入力 (例: 9984.T)", key="custom_ticker_input")
    ticker_name = "カスタム銘柄"
else:
    valid_tickers = list(tickers[sector].keys())
    if st.session_state.get('ticker_choice') not in valid_tickers:
        st.session_state['ticker_choice'] = valid_tickers[0]
    ticker_name = st.sidebar.selectbox("銘柄", valid_tickers, key="ticker_choice")
    ticker_symbol = tickers[sector][ticker_name]

st.sidebar.header("2. 期間・時間軸設定")
timeframe_label = st.sidebar.selectbox("時間軸", ["日足 (1d)", "週足 (1wk)", "1時間足 (1h)", "5分足 (5m)"])
tf_map = {"日足 (1d)": "1d", "週足 (1wk)": "1wk", "1時間足 (1h)": "1h", "5分足 (5m)": "5m"}
timeframe = tf_map[timeframe_label]

if timeframe == "1d":
    period_options = {"半年": "6mo", "1年": "1y", "2年": "2y", "3年": "3y", "5年": "5y"}
    idx = 1
elif timeframe == "1wk":
    period_options = {"1年": "1y", "3年": "3y", "5年": "5y", "10年": "10y"}
    idx = 1
elif timeframe == "1h":
    period_options = {"1週間": "5d", "1ヶ月": "1mo", "3ヶ月": "3mo", "1年": "1y", "2年": "2y"}
    idx = 2
elif timeframe == "5m":
    period_options = {"1日": "1d", "5日": "5d", "1ヶ月": "1mo", "60日": "60d"}
    idx = 1

period_label = st.sidebar.selectbox("表示期間", options=list(period_options.keys()), index=idx)
period = period_options[period_label]

st.sidebar.header("3. チャート表示オプション")
show_ma = st.sidebar.checkbox("移動平均線を表示", value=True)
show_bb = st.sidebar.checkbox("ボリンジャーバンドを表示", value=True)
show_ichimoku = st.sidebar.checkbox("一目均衡表を表示", value=False)

st.sidebar.header("4. シグナル・指標パラメータ調整")
sensitivity = st.sidebar.radio("シグナル発生の感度", ["標準", "敏感 (シグナル多)"])
trend_filter = st.sidebar.checkbox("上昇トレンド時のみ買いシグナルを出す (順張り)", value=True)
dmi_filter = st.sidebar.checkbox("DMIフィルター (ADXでトレンドの強さを確認)", value=False)
rsi_overbought = st.sidebar.slider("RSI 買われすぎ水準 (売りシグナル)", 60, 90, 70)
rsi_oversold = st.sidebar.slider("RSI 売られすぎ水準 (買いシグナル)", 10, 40, 30)
bb_std = st.sidebar.slider("ボリンジャーバンド ±σ (売りシグナル用)", 1.0, 3.0, 2.0, 0.1)

# --- サイドバー設定は完了 ---

# --- セクター状況サマリー ---
if sector != "カスタム（直接入力）":
    with st.expander(f"📊 {sector} セクターの全銘柄ステータス一覧", expanded=True):
        sector_tickers = tickers[sector]
        summary_data = []
        cols = st.columns(len(sector_tickers))
        
        # セクター内の各銘柄の状態を並列的に取得して表示（簡易版）
        for idx, (t_name, t_code) in enumerate(sector_tickers.items()):
            info = get_latest_signal_info(t_code, period, timeframe, bb_std, rsi_overbought, rsi_oversold, sensitivity, trend_filter, dmi_filter)
            if info:
                summary_data.append({
                    "銘柄名": t_name,
                    "コード": t_code,
                    "価格": info["価格"],
                    "騰落率": info["騰落率"],
                    "RSI": info["RSI"],
                    "判定": info["シグナル"]
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            st.caption("※上の表から銘柄名を確認し、サイドバーで切り替えると詳細チャートが表示されます。")

st.markdown("---")

with st.spinner('メインデータを取得・計算中...'):
    df = load_data(ticker_symbol, period, timeframe)

if df.empty:
    st.error("データの取得に失敗しました。銘柄コードまたは期間の組み合わせを確認してください。")
else:
    df = add_all_indicators(df, bb_std=bb_std)
    df = detect_signals(df, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold, sensitivity=sensitivity, trend_filter=trend_filter, dmi_filter=dmi_filter)
    
    latest = df.iloc[-1]
    st.markdown(f"### 📢 {ticker_name} ({ticker_symbol}) の現在（直近）の判定結果")
    if latest['Buy_Signal']:
        st.success("✨ **現在、「買いシグナル」が点灯しています！**")
    elif latest['Sell_Signal']:
        st.error("⚠️ **現在、「売りシグナル」が点灯しています！**")
    else:
        st.info("⏸️ 現在、新しいシグナルは出ていません。（様子見）")

    # --- チャート作成・表示 ---
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.35, 0.12, 0.12, 0.16, 0.25],
                        subplot_titles=(f"{ticker_name} ({ticker_symbol}) - 価格 & トレンド指標", "MACD", "RSI", "ストキャスティクス", "DMI / ADX"))

    # --- Row 1: 価格チャート ---
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ローソク足'), row=1, col=1)
    
    if show_ma:
        colors = ['orange', 'blue', 'green', 'red']
        for ma, color in zip([5, 25, 75, 200], colors):
            fig.add_trace(go.Scatter(x=df.index, y=df[f'MA_{ma}'], line=dict(color=color, width=1), name=f'{ma}MA'), row=1, col=1)
            
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name=f'+{bb_std}σ'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name=f'-{bb_std}σ', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    if show_ichimoku:
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], line=dict(color='#E53935', width=1), name='転換線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun'], line=dict(color='#1E88E5', width=1), name='基準線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'], line=dict(color='rgba(0,0,0,0)', width=0), name='先行スパンA', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'], line=dict(color='rgba(0,0,0,0)', width=0), fill='tonexty', fillcolor='rgba(76, 175, 80, 0.2)', name='一目均衡表 雲'), row=1, col=1)

    # シグナルマーカー
    buy_signals = df[df['Buy_Signal']]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'] * 0.98, mode='markers', marker=dict(symbol='triangle-up', color='magenta', size=14, line=dict(width=1, color='DarkSlateGrey')), name='買い'), row=1, col=1)

    sell_signals = df[df['Sell_Signal']]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'] * 1.02, mode='markers', marker=dict(symbol='triangle-down', color='white', size=15, line=dict(width=2, color='red')), name='売り'), row=1, col=1)

    # --- Row 2-5: テクニカル指標 ---
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='orange', width=1), name='Signal'), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=['green' if v>=0 else 'red' for v in df['MACD_Hist']], name='Hist'), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=3, col=1)
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='blue', width=1), name='%K'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='orange', width=1), name='%D'), row=4, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['Plus_DI'], line=dict(color='green', width=1), name='+DI'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Minus_DI'], line=dict(color='red', width=1), name='-DI'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='blue', width=2), name='ADX'), row=5, col=1)

    fig.update_layout(height=1400, xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- 全銘柄シグナルスキャナー ---
    st.markdown("### 📡 全銘柄シグナルスキャナー")
    st.write("登録されている全銘柄を一括チェックし、現在シグナルが点灯している銘柄を表示します。")
    if st.button("🔄 今すぐ全銘柄をスキャンする"):
        with st.spinner("全銘柄のデータを解析中...（しばらくお待ちください）"):
            ticker_to_name = {}
            for sec, t_dict in tickers.items():
                for name, code in t_dict.items():
                    ticker_to_name[code] = name
            
            buy_list, sell_list = [], []
            for code, name in ticker_to_name.items():
                try:
                    df_scan = load_data(code, period, timeframe)
                    if not df_scan.empty and len(df_scan) >= 50:
                        df_scan = add_all_indicators(df_scan, bb_std=bb_std)
                        df_scan = detect_signals(df_scan, rsi_overbought=rsi_overbought, rsi_oversold=rsi_oversold, sensitivity=sensitivity, trend_filter=trend_filter, dmi_filter=dmi_filter)
                        if df_scan.iloc[-1]['Buy_Signal']:
                            buy_list.append(f"{name} ({code})")
                        elif df_scan.iloc[-1]['Sell_Signal']:
                            sell_list.append(f"{name} ({code})")
                except: continue
            st.session_state['scan_results'] = {"buy": buy_list, "sell": sell_list}

    if 'scan_results' in st.session_state:
        res = st.session_state['scan_results']
        
        # サマリー表示
        sc_m1, sc_m2 = st.columns(2)
        sc_m1.metric("買いシグナル発見数", f"{len(res['buy'])} 銘柄", delta=len(res['buy']) if res['buy'] else 0, delta_color="normal")
        sc_m2.metric("売りシグナル発見数", f"{len(res['sell'])} 銘柄", delta=-len(res['sell']) if res['sell'] else 0, delta_color="inverse")

        sc1, sc2 = st.columns(2)
        
        with sc1:
            st.success("#### 🟢 買いサイン点灯中")
            if not res['buy']:
                st.write("現在、買いサインが出ている銘柄はありません。")
            else:
                for t in res['buy']:
                    try:
                        name, code = t.rsplit(" (", 1)
                        code = code.rstrip(")")
                        # エクスパンダーで簡易チャートを見れるようにする
                        with st.expander(f"📈 {name} ({code})"):
                            st.write(f"この銘柄のメインチャートへ切り替える場合は下のボタンを押してください。")
                            if st.button(f"メインチャートで表示: {code}", key=f"jump_buy_{code}"):
                                jump_to_ticker_action(name, code)
                                st.rerun()
                            # 簡易的な価格情報を表示
                            st.info(f"この銘柄で買いシグナルが検出されました。トレンド条件、MACD、ストキャスティクス等の条件を満たしています。")
                    except Exception as e:
                        st.write(f"- {t}")
                        
        with sc2:
            st.error("#### 🔴 売りサイン点灯中")
            if not res['sell']:
                st.write("現在、売りサインが出ている銘柄はありません。")
            else:
                for t in res['sell']:
                    try:
                        name, code = t.rsplit(" (", 1)
                        code = code.rstrip(")")
                        with st.expander(f"📉 {name} ({code})"):
                            st.write(f"この銘柄のメインチャートへ切り替える場合は下のボタンを押してください。")
                            if st.button(f"メインチャートで表示: {code}", key=f"jump_sell_{code}"):
                                jump_to_ticker_action(name, code)
                                st.rerun()
                            st.warning(f"この銘柄で売りシグナルが検出されました。買われすぎ水準、または反転の兆候があります。")
                    except Exception as e:
                        st.write(f"- {t}")
            
    st.markdown("---")

    # --- シグナル判定ルールのわかりやすい解説 ---
    with st.expander("💡 「なぜシグナルが出るの？」現在の判定ルールのやさしい解説を見る", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟢 買いシグナルが出る条件")
            buy_rules = []
            if trend_filter:
                buy_rules.append("📈 **株価のトレンド**: 75日線より株価が上にあり、全体が上昇傾向であること")
            else:
                buy_rules.append("🌊 **株価のトレンド**: 不問（下落中の「底拾い」も狙います）")
                
            if dmi_filter:
                buy_rules.append("🔥 **トレンドの勢い**: ADXが20以上で、緑の線(+DI)が赤の線(-DI)より上")
                
            if sensitivity == "敏感 (シグナル多)":
                buy_rules.append("⚡ **買うタイミング (以下のどれか1つでも満たせばOK)**:\n  - **MACD**: ゴールデンクロスした\n  - **ストキャス**: 50以下で上向きにクロスした\n  - **RSI**: 売られすぎ水準から少しでも反発した")
            else:
                buy_rules.append("🎯 **買うタイミング (以下のすべてを満たす)**:\n  - 株価が25日線付近（±8%以内）まで少し下がった時\n  - かつ、**MACD**がクロス、または**ストキャス**が40以下でクロスした時")

            buy_rules.append("🛡️ **ダマシ防止 (以下の時は絶対に買いません)**:\n  - MACDの棒グラフが下向きの時（落ちるナイフ状態）\n  - ストキャスが70以上、またはRSIが65以上の時（すでに高値圏の時）")
            st.info("\n\n".join(buy_rules))

        with col2:
            st.markdown("#### ⬛ 売りシグナルが出る条件")
            sell_rules = []
            if dmi_filter:
                sell_rules.append("🔥 **トレンドの勢い**: ADXが20以上で、赤の線(-DI)が緑の線(+DI)より上")
                
            if sensitivity == "敏感 (シグナル多)":
                sell_rules.append("⚡ **売るタイミング (以下のどれか1つでも満たせば即売り)**:\n  - **MACD**: デッドクロスした\n  - **ストキャス**: 50以上で下向きにクロスした\n  - **RSI**: 買われすぎ水準から少しでも下落した")
            else:
                sell_rules.append("🎯 **売るタイミング (以下のどれか1つを満たせば売り)**:\n  - ボリンジャーバンドの上の線(+σ)にタッチした\n  - **ストキャス**が70以上でデッドクロスした\n  - **RSI**が買われすぎ水準を超えた")
                
            sell_rules.append("🛡️ **ダマシ防止**:\n  - MACDの棒グラフが上向きの時は売りません（まだ上がる勢いがあるため）")
            st.warning("\n\n".join(sell_rules))

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.35, 0.12, 0.12, 0.16, 0.25],
                        subplot_titles=(f"{ticker_name} ({ticker_symbol}) - 価格 & トレンド指標", "MACD", "RSI", "ストキャスティクス", "DMI / ADX"))

    # --- Row 1: 価格チャート ---
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'],
                                 name='ローソク足'), row=1, col=1)
    
    if show_ma:
        colors = ['orange', 'blue', 'green', 'red']
        for ma, color in zip([5, 25, 75, 200], colors):
            fig.add_trace(go.Scatter(x=df.index, y=df[f'MA_{ma}'], line=dict(color=color, width=1), name=f'{ma}MA'), row=1, col=1)
            
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', width=1, dash='dash'), name=f'+{bb_std}σ'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', width=1, dash='dash'), name=f'-{bb_std}σ', fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

    if show_ichimoku:
        fig.add_trace(go.Scatter(x=df.index, y=df['Tenkan'], line=dict(color='#E53935', width=1), name='転換線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Kijun'], line=dict(color='#1E88E5', width=1), name='基準線'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_A'], line=dict(color='rgba(0,0,0,0)', width=0), name='先行スパンA', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Senkou_B'], line=dict(color='rgba(0,0,0,0)', width=0), fill='tonexty', fillcolor='rgba(76, 175, 80, 0.2)', name='一目均衡表 雲'), row=1, col=1)

    # シグナルマーカー
    buy_signals = df[df['Buy_Signal']]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'] * 0.98, 
                                 mode='markers', marker=dict(symbol='triangle-up', color='magenta', size=14, line=dict(width=1, color='DarkSlateGrey')), 
                                 name='買いシグナル'), row=1, col=1)

    sell_signals = df[df['Sell_Signal']]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'] * 1.02, 
                                 mode='markers', marker=dict(symbol='triangle-down', color='white', size=15, line=dict(width=2, color='red')), 
                                 name='売りシグナル'), row=1, col=1)

    # --- Row 2: MACD ---
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='orange', width=1), name='Signal'), row=2, col=1)
    colors_macd = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors_macd, name='Histogram'), row=2, col=1)

    # --- Row 3: RSI ---
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI'), row=3, col=1)
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=3, col=1)

    # --- Row 4: ストキャスティクス ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='blue', width=1), name='%K'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='orange', width=1), name='%D'), row=4, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=4, col=1)

    # --- Row 5: DMI / ADX ---
    fig.add_trace(go.Scatter(x=df.index, y=df['Plus_DI'], line=dict(color='green', width=1), name='+DI (買いの強さ)'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Minus_DI'], line=dict(color='red', width=1), name='-DI (売りの強さ)'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='blue', width=2), name='ADX (トレンド全体の勢い)'), row=5, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="gray", row=5, col=1)

    # --- レイアウトと休場非表示設定 ---
    layout_update = dict(
        height=1400, 
        xaxis_rangeslider_visible=False, 
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified"
    )
    fig.update_layout(**layout_update)

    if timeframe in ['1d', '1wk']:
        hover_fmt = "%Y年%m月%d日"
    elif timeframe == '1h':
        hover_fmt = "%m月%d日 %H時"
    elif timeframe == '5m':
        hover_fmt = "%m月%d日 %H:%M"
    else:
        hover_fmt = "%Y-%m-%d %H:%M"

    fig.update_xaxes(hoverformat=hover_fmt)

    # 休場や夜間のギャップを消す処理
    if ticker_symbol.endswith('.T'):
        # 日本株の場合：夜間と昼休みを非表示
        if timeframe in ['1h', '5m']:
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]), # 土日非表示
                    dict(bounds=[15, 9], pattern="hour"), # 15:00~09:00を非表示
                    dict(bounds=[11.5, 12.5], pattern="hour") # 昼休み(11:30~12:30)非表示
                ]
            )
        else:
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]) # 日足以上は土日のみ非表示
                ]
            )
    else:
        # 為替(FX)などの場合：土日のみ非表示（24時間取引のため夜間を消さない）
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"])
            ]
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- バックテスト（勝率）の計算と表示 ---
    st.subheader(f"📊 【{ticker_name} / {timeframe}】 今回のルールの過去勝率テスト結果")
    st.write("現在表示されている期間内で、**「買いサインが出たら買い、次に売りサインが出たら売る」**というルールで自動売買したシミュレーション結果です。")
    
    trades = []
    position = None
    
    for idx, row in df.iterrows():
        if row['Buy_Signal'] and position is None:
            position = {'buy_price': row['Close'], 'buy_time': idx}
        elif row['Sell_Signal'] and position is not None:
            profit_pct = (row['Close'] - position['buy_price']) / position['buy_price'] * 100
            trades.append({
                'buy_time': position['buy_time'],
                'sell_time': idx,
                'profit_pct': profit_pct,
                'is_win': profit_pct > 0
            })
            position = None
            
    total_trades = len(trades)
    if total_trades > 0:
        wins = sum(1 for t in trades if t['is_win'])
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100
        avg_profit = sum([t['profit_pct'] for t in trades]) / total_trades
        total_profit = sum([t['profit_pct'] for t in trades])
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("トレード回数", f"{total_trades} 回")
        m2.metric("勝率", f"{win_rate:.1f} %")
        m3.metric("1回あたりの平均利益", f"{avg_profit:.2f} %")
        m4.metric("合計利益 (累積)", f"{total_profit:.2f} %")
        
        # 履歴を少しだけ表示
        with st.expander("詳細なトレード履歴を見る"):
            trade_df = pd.DataFrame(trades)
            # 見やすくフォーマット
            trade_df['勝敗'] = trade_df['is_win'].apply(lambda x: '🟢 勝ち' if x else '🔴 負け')
            trade_df['利益率'] = trade_df['profit_pct'].apply(lambda x: f"{x:+.2f}%")
            trade_df = trade_df[['buy_time', 'sell_time', '勝敗', '利益率']]
            trade_df.columns = ['買った日時', '売った日時', '勝敗', '利益率(%)']
            st.dataframe(trade_df, use_container_width=True)
            
    else:
        st.info("※ この期間内では、買いから売りまで完了したトレードはありませんでした。")
