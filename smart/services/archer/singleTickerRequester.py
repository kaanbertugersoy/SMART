from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf


def get_single_ticker_data(ticker=None,
                           start_date=None,
                           end_date=datetime.now(),
                           day_count=729,
                           interval="1d"):
    if ticker is None:
        raise ValueError("ticker cannot be None")

    if start_date is None:
        start_date = end_date - \
            timedelta(days=day_count)  # 2 years of data
        start_date.strftime(
            '%Y-%m-%d')

    ticker = yf.Ticker(ticker)
    df = ticker.history(start=start_date, end=end_date.strftime(
        '%Y-%m-%d'), interval=interval)
    if df.empty:
        raise ValueError("No data found for the given ticker")

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df


def compute_and_add_log_returns(data, columns=None, suffix='LogReturn'):

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if columns is None:
        columns = data.columns  # If no columns specified, use all columns

    for col in columns:
        log_return_col = f'{col}{suffix}'
        data[log_return_col] = np.log(data[col])

    data.dropna(inplace=True)

    return data


def add_direction(data, column=None, suffix='_direction'):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    data[f'{column}{suffix}'] = np.where(data[column] > 0, 1, 0)
    data[f'{column}{suffix}'] = data[f'{column}{suffix}'].shift(-1)

    return data
