import numpy as np
import pandas as pd


def createAlphaOneFeatures(df, rolling_window=50, rsi_window=14):
    df["dir"] = np.where(df["returns"] > 0, 1, 0)
    df["dir"] = df["dir"].shift(-1)

    df = calculate_macd(df)
    df = calculate_obv(df)
    df = calculate_ADLine(df)
    df = calculate_stochastic(df, period=14, smooth_k=3, smooth_d=3)
    df = calculate_aroon(df, period=14)
    df = calculate_adx(df, period=14)
    df = calculate_rsi(df, window=14)

    df = calculate_sma(df, window=rolling_window)
    df = calculate_ema(df, window=rolling_window)
    df = calculate_bollinger_bands(df, window=rolling_window, no_of_std=2)
    df = calculate_roc(df, window=12)

    df.dropna(inplace=True)

    features = ["dir", "macd", "macd_signal", "sma",
                "rsi", "ema", "blngr_h", "blngr_l"]

    return df, features


def calculate_ADLine(data):
    # Calculate the Money Flow Multiplier (MFM)
    df = data.copy()
    df["MFM"] = ((data['close'] - data['low']) -
                 (data['high'] - data['close'])) / (data['high'] - data['low'])

    # Calculate the Money Flow volume (MFV)
    df["MFV"] = df["MFM"] * data['volume']

    # Calculate the Accumulation/Distribution Line (A/D Line)
    data['adl'] = df["MFV"].cumsum()

    return data


def calculate_stochastic(data, period=14, smooth_k=3, smooth_d=3):
    # Calculate Stochastic %K
    df = data.copy()
    df["lowest_low"] = data['low'].rolling(window=period).min()
    df["highest_high"] = data['high'].rolling(window=period).max()
    df["K"] = (data['close'] - df["lowest_low"]) / \
        (df["highest_high"] - df["lowest_low"]) * 100

    # Smooth %K
    df["K"] = df["K"].rolling(window=smooth_k).mean()

    # Calculate Stochastic %D
    data['stochastic'] = df["K"].rolling(window=smooth_d).mean()

    return data


def calculate_aroon(data, period=14):
    # Calculate Aroon Up and Aroon Down
    data['aaron_up'] = data['high'].rolling(window=period).apply(
        lambda x: ((period - x[::-1].argmax()) / period) * 100, raw=True)
    data['aaron_down'] = data['low'].rolling(window=period).apply(
        lambda x: ((period - x[::-1].argmin()) / period) * 100, raw=True)

    return data


def calculate_obv(data):
    obv = [0]
    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i-1]:
            obv.append(obv[-1] + data['volume'].iloc[i])
        elif data['close'].iloc[i] < data['close'].iloc[i-1]:
            obv.append(obv[-1] - data['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['obv'] = obv
    return data


def calculate_adx(data, period=14):
    # Calculate the true range (TR)
    tr = data[['high', 'low', 'close']].apply(lambda row: max(
        row['high'] - row['low'], abs(row['high'] - row['close']), abs(row['low'] - row['close'])), axis=1)

    # Calculate the +DM and -DM
    dm_plus = data['high'].diff()
    dm_minus = -data['low'].diff()
    dm_plus = dm_plus.where((dm_plus
                             > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus
                               > dm_plus) & (dm_minus > 0), 0)

    # Calculate smoothed TR, +DM, and -DM
    tr_smooth = tr.rolling(window=period).mean()
    dm_smooth_plus = dm_plus.rolling(window=period).mean()
    dm_smooth_minus = dm_minus.rolling(window=period).mean()

    # Calculate +DI and -DI
    di_plus = 100 * (dm_smooth_plus / tr_smooth)
    di_minus = 100 * (dm_smooth_minus / tr_smooth)

    # Calculate the DX
    dx = 100 * (abs(di_plus - di_minus) /
                (di_plus + di_minus))

    # Calculate the ADX
    data['adx'] = dx.rolling(window=period).mean()

    return data


def calculate_rsi(data, window):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    return data

# Calculate Simple Moving Average (SMA)


def calculate_sma(data, window):
    data['sma'] = data['close'].rolling(window=window).mean()
    return data

# Calculate Exponential Moving Average (EMA)


def calculate_ema(data, window):
    data['ema'] = data['close'].ewm(span=window, adjust=False).mean()
    return data

# Calculate Relative Strength Index (RSI)


def calculate_bollinger_bands(data, window, no_of_std):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    data['blngr_h'] = rolling_mean + (rolling_std * no_of_std)
    data['blngr_l'] = rolling_mean - (rolling_std * no_of_std)
    return data


# Calculate Rate of Change (ROC)


def calculate_roc(data, window):
    data['roc'] = data['close'].pct_change(periods=window) * 100
    return data


def calculate_macd(data):
    data["macd"] = data["close"].ewm(span=12, adjust=False).mean(
    ) - data["close"].ewm(span=26, adjust=False).mean()
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    return data
