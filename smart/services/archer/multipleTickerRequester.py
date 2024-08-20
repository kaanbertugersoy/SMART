from datetime import datetime, timedelta
import numpy as np
import yfinance as yf


def get_multiple_ticker_data(tickers=[],
                             start_date=None,
                             end_date=datetime.now(),
                             day_count=729,
                             interval="1d"):
    if tickers is []:
        raise ValueError("tickers array cannot be empty")

    if start_date is None:
        start_date = end_date - \
            timedelta(days=day_count)  # 2 years of data
        start_date.strftime(
            '%Y-%m-%d')

    df = yf.download(tickers=tickers, start=start_date, end=end_date.strftime(
        '%Y-%m-%d'), interval=interval)
    if df.empty:
        raise ValueError("No data found for the given tickers")

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df


def add_returns_and_direction(data):
    data["returns"] = np.log(data.close / data.close.shift(1))
    data["returns"] = data["returns"].shift(-1)
    data.dropna(inplace=True)
    # data["dir"] = np.where(data["returns"] > 0, 1, 0)
    # shift the dir column to predict the next day's direction [IMPORTANT!!!]
    # data["dir"] = data["dir"].shift(-1)  # !!!!!!!!!
    return data


if __name__ == "__main__":

    data = get_multiple_ticker_data(
        ["AAPL", "MSFT", "GOOGL", "AMZN"], day_count=60, interval="15m")

    df_swapped = data.swaplevel(0, 1, axis=1).sort_index(axis=1)
    data_array = df_swapped.to_numpy()

    timesteps = df_swapped.shape[0]
    number_of_features = len(df_swapped.columns.levels[1])
    number_of_tickers = len(df_swapped.columns.levels[0])

    reshaped_data = data_array.reshape(
        timesteps, number_of_features, number_of_tickers)

    print(reshaped_data.shape)
