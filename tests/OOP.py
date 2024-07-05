import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
plt.style.use("seaborn-v0_8")


class FinancialInstrumentBase:
    """
    A base class representing a financial instrument.

    Attributes:
        _ticker (str): The ticker symbol of the financial instrument.
        start (str): The start date for data retrieval.
        end (str): The end date for data retrieval.
        data (pandas.DataFrame): The historical data of the financial instrument.

    Methods:
        __init__(self, ticker, start, end): Initializes a new instance of the FinancialInstrumentBase class.
        __repr__(self): Returns a string representation of the FinancialInstrumentBase object.
        get_data(self): Retrieves the historical data of the financial instrument.
        log_returns(self): Calculates the logarithmic returns of the financial instrument.
        plot_prices(self): Plots the price chart of the financial instrument.
        plot_returns(self, kind="ts"): Plots the returns chart or the frequency histogram of the financial instrument.
        set_ticker(self, ticker=None): Sets the ticker symbol of the financial instrument.

    """

    def __init__(self, ticker, start, end):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.get_data()
        self.log_returns()

    def __repr__(self):
        return f"FinancialInstrument({self.ticker}, {self.start}, {self.end})"

    def get_data(self):
        raw = yf.download(self._ticker, start=self.start,
                          end=self.end).Close.to_frame()
        raw.rename(columns={"Close": "Price"}, inplace=True)
        self.data = raw

    def log_returns(self):
        self.data["Log_returns"] = np.log(
            self.data.Price / self.data.Price.shift(1))

    def plot_prices(self):
        self.data.Price.plot(
            figsize=(12, 8), title=f"Price Chart: {self._ticker}", fontsize=15)
        plt.show()

    def plot_returns(self, kind="ts"):
        if kind == "ts":
            self.data.Log_returns.plot(
                figsize=(12, 8), title=f"Returns Chart: {self._ticker}", fontsize=15)
            plt.show()
        elif kind == "hist":
            self.data.Log_returns.hist(figsize=(12, 8), bins=int(np.sqrt(
                len(self.data))), title=f"Frequency of Returns: {self.ticker}", fontsize=15)
            plt.show()

    def set_ticker(self, ticker=None):
        if ticker is not None:
            self._ticker = ticker
            self.get_data()
            self.log_returns()


class RiskAndReturn(FinancialInstrumentBase):
    """
    A class representing a financial instrument for calculating risk and return.

    Attributes:
        ticker (str): The ticker symbol of the financial instrument.
        start (datetime): The start date of the data.
        end (datetime): The end date of the data.
        freq (str, optional): The frequency of resampling the data. Defaults to None.
    """

    def __init__(self, ticker, start, end, freq=None):
        super().__init__(ticker, start, end)
        self.freq = freq

    def __repr__(self):
        return f"RiskAndReturn({self.ticker}, {self.start}, {self.end})"

    def mean_returns(self):
        """
        Calculate the mean returns of the financial instrument.

        Returns:
            float: The mean returns.
        """
        if self.freq is None:
            return self.data.Log_returns.mean()
        else:
            freq_resampled = self.data.Price.resample(self.freq).last()
            resampled_returns = np.log(
                freq_resampled / freq_resampled.shift(1))
            return resampled_returns.mean()

    def std_returns(self):
        """
        Calculate the standard deviation of returns of the financial instrument.

        Returns:
            float: The standard deviation of returns.
        """
        if self.freq is None:
            return self.data.Log_returns.std()
        else:
            freq_resampled = self.data.Price.resample(self.freq).last()
            resampled_returns = np.log(
                freq_resampled / freq_resampled.shift(1))
            return resampled_returns.std()

    def annualized_returns(self):
        """
        Calculate the annualized returns and risk of the financial instrument.

        Returns:
            tuple: A tuple containing the mean return and risk.
        """
        mean_return = round(self.mean_returns() * 252, 3)
        risk = round(self.std_returns() * np.sqrt(252), 3)
        return mean_return, risk


stock = RiskAndReturn(ticker="AAPL", start="2010-01-01", end="2020-01-01")

stock.plot_prices()
