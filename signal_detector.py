def detect_signals(df, rsi_overbought=70, rsi_oversold=30, sensitivity="標準", trend_filter=True, dmi_filter=False):
    df = df.copy()

    # --- 初期化（ここ重要）
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False

    if len(df) < 30:
        return df

    # --- 前データ
    prev_macd = df['MACD'].shift(1)
    prev_signal = df['MACD_Signal'].shift(1)
    prev_rsi = df['RSI'].shift(1)
    prev_hist = df['MACD_Hist'].shift(1)

    # --- 基本シグナル（絶対出る）
    macd_gc = (prev_macd <= prev_signal) & (df['MACD'] > df['MACD_Signal'])
    macd_dc = (prev_macd >= prev_signal) & (df['MACD'] < df['MACD_Signal'])

    rsi_rebound = (df['RSI'] < 45) & (df['RSI'] > prev_rsi)
    rsi_drop = (df['RSI'] > 65) & (df['RSI'] < prev_rsi)

    # --- トレンド（ゆるく）
    if trend_filter:
        uptrend = (df['MA_25'] >= df['MA_75']) | (df['MA_75'].isna())
    else:
        uptrend = pd.Series(True, index=df.index)

    # --- DMI（あってもなくてもOKにする）
    if dmi_filter:
        dmi_uptrend = (df['Plus_DI'] > df['Minus_DI']) | (df['ADX'] > 15)
        dmi_downtrend = (df['Minus_DI'] > df['Plus_DI']) | (df['ADX'] > 15)
    else:
        dmi_uptrend = pd.Series(True, index=df.index)
        dmi_downtrend = pd.Series(True, index=df.index)

    # --- 改良ロジック（でも緩め）
    early_entry = (
        (df['MACD_Hist'] > prev_hist) &
        (df['RSI'] > 40)
    )

    pullback_entry = (
        (df['Close'] > df['MA_25'] * 0.97) &
        (df['RSI'] > 35)
    )

    # --- 出来高（無視OKにする）
    volume_filter = (df['Volume'] > df['Volume'].rolling(20).mean()) | (df['Volume'].isna())

    # --- 買い（とにかく出す）
    df.loc[
        (
            macd_gc |
            rsi_rebound |
            early_entry |
            pullback_entry
        ) &
        uptrend &
        dmi_uptrend &
        volume_filter,
        'Buy_Signal'
    ] = True

    # --- 売り
    df.loc[
        (
            macd_dc |
            rsi_drop |
            (df['RSI'] > rsi_overbought)
        ) &
        dmi_downtrend,
        'Sell_Signal'
    ] = True

    # --- 型強制（表示バグ対策）
    df['Buy_Signal'] = df['Buy_Signal'].fillna(False).astype(bool)
    df['Sell_Signal'] = df['Sell_Signal'].fillna(False).astype(bool)

    return df
