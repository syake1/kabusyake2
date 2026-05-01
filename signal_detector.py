def detect_signals(df, rsi_overbought=70, rsi_oversold=30, sensitivity="標準", trend_filter=True, dmi_filter=False):
    df['Buy_Signal'] = False
    df['Sell_Signal'] = False
    
    if len(df) < 50:
        return df

    # --- 前データ ---
    prev_macd = df['MACD'].shift(1)
    prev_signal = df['MACD_Signal'].shift(1)
    prev_rsi = df['RSI'].shift(1)
    prev_hist = df['MACD_Hist'].shift(1)

    # --- トレンド ---
    if trend_filter:
        uptrend = df['MA_25'] > df['MA_75']
    else:
        uptrend = pd.Series(True, index=df.index)

    # --- DMI ---
    if dmi_filter:
        dmi_uptrend = (df['Plus_DI'] > df['Minus_DI']) & (df['ADX'] > 20)
        dmi_downtrend = (df['Minus_DI'] > df['Plus_DI']) & (df['ADX'] > 20)
    else:
        dmi_uptrend = pd.Series(True, index=df.index)
        dmi_downtrend = pd.Series(True, index=df.index)

    # --- 基本 ---
    macd_gc = (prev_macd <= prev_signal) & (df['MACD'] > df['MACD_Signal'])
    macd_dc = (prev_macd >= prev_signal) & (df['MACD'] < df['MACD_Signal'])

    # ===== 改良ポイントここから =====

    # ① 初動（早いエントリー）
    early_entry = (
        (df['MACD_Hist'] > prev_hist) &
        (prev_hist < df['MACD_Hist'].shift(2)) &
        (df['RSI'] > 45) &
        (df['Close'] > df['MA_25'])
    )

    # ② 押し目（←あなたの赤四角）
    pullback_entry = (
        (df['Close'] > df['MA_25']) &
        (df['Low'] <= df['MA_25'] * 0.98) &
        (df['Close'] > df['MA_25']) &
        (df['RSI'] > 40)
    )

    # ③ ボリンジャー逆張り
    bb_rebound = (
        (df['Close'] < df['BB_Lower']) &
        (df['RSI'] < 40)
    )

    # ④ 出来高フィルター（重要）
    volume_filter = df['Volume'] > df['Volume'].rolling(20).mean()

    # ⑤ 買われすぎ防止
    not_overbought = (df['RSI'] < 65)

    # ===== 買いシグナル =====
    df.loc[
        uptrend &
        dmi_uptrend &
        (
            macd_gc |
            early_entry |
            pullback_entry |
            bb_rebound
        ) &
        volume_filter &
        not_overbought,
        'Buy_Signal'
    ] = True

    # ===== 売りシグナル =====
    df.loc[
        dmi_downtrend &
        (
            (df['RSI'] > rsi_overbought) |
            (df['High'] >= df['BB_Upper']) |
            macd_dc
        ),
        'Sell_Signal'
    ] = True

    return df
