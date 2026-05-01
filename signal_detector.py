def detect_signals(df, rsi_overbought=70, rsi_oversold=30, sensitivity="標準", trend_filter=True, dmi_filter=False):
    df = df.copy()
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False

    if len(df) < 30:
        return df

    # --- 基本 ---
    prev_macd = df['MACD'].shift(1)
    prev_signal = df['MACD_Signal'].shift(1)

    # ★ 超シンプル（まず出す）
    macd_gc = (prev_macd <= prev_signal) & (df['MACD'] > df['MACD_Signal'])
    macd_dc = (prev_macd >= prev_signal) & (df['MACD'] < df['MACD_Signal'])

    # ★ RSIだけでも拾う（これが重要）
    rsi_rebound = (df['RSI'] < 40) & (df['RSI'] > df['RSI'].shift(1))
    rsi_drop = (df['RSI'] > 70) & (df['RSI'] < df['RSI'].shift(1))

    # ★ とにかく出す
    df.loc[
        macd_gc | rsi_rebound,
        'Buy_Signal'
    ] = True

    df.loc[
        macd_dc | rsi_drop,
        'Sell_Signal'
    ] = True

    return df
