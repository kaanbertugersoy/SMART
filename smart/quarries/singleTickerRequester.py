from datetime import datetime, timedelta
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
    data = ticker.history(start=start_date, end=end_date.strftime(
        '%Y-%m-%d'), interval=interval)
    if data.empty:
        raise ValueError("No data found for the given ticker")

    data = data[['Close', 'Volume', 'High', 'Low', 'Open']]
    data.rename(columns={"Close": "close", "Volume": "volume",
                "High": "high", "Low": "low", "Open": "open"}, inplace=True)  # Rename columns for pandas_ta

    data["returns"] = np.log(data.close / data.close.shift(1))
    data["dir"] = np.where(data["returns"] > 0, 1, 0)
    # shift the dir column to predict the next day's direction [IMPORTANT!!!]
    data["dir"] = data["dir"].shift(-1)  # !!!!!!!!!

    # Already localized
    data.index = data.index.tz_localize(None)
    return data
