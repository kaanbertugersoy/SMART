from itertools import product
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SMABacktester:
    """
    A class for backtesting a simple moving average (SMA) trading strategy.
    
    Attributes:
        ticker (str): The ticker symbol of the financial instrument.
        sma_s (int): The short-term SMA period.
        sma_l (int): The long-term SMA period.
        start (str or datetime): The start date of the backtesting period.
        end (str or datetime): The end date of the backtesting period.
        results (pd.DataFrame): The backtesting results.
        data (pd.DataFrame): The historical price and returns data.
    
    Methods:
        __init__(self, ticker, sma_s, sma_l, start, end): Initializes the SMABacktester object.
        __repr__(self): Returns a string representation of the SMABacktester object.
        get_data(self): Downloads and prepares the historical price and returns data.
        prepare_data(self): Calculates the SMA values based on the given periods.
        set_sma_params(self, sma_s=None, sma_l=None): Sets the SMA parameters and recalculates the SMA values.
        test_strategy(self): Tests the SMA trading strategy and returns the performance metrics.
        plot_results(self): Plots the cumulative returns of the strategy.
        optimize_sma_params(self, sma_s_range, sma_l_range): Optimizes the SMA parameters and returns the best parameters and performance.
    """
    
    def __init__ (self, ticker, sma_s, sma_l, start, end):
        self._ticker = ticker
        self.sma_s = sma_s
        self.sma_l = sma_l
        self.start = start
        self.end = end
        self.results = None
        self.data = None
        self.get_data()
        self.prepare_data()

    def __repr__(self):
        return "SMABacktester({},{},{},{},{})".format(self._ticker, self.sma_s, self.sma_l, self.start, self.end)

    def get_data(self):
        raw = yf.download(self._ticker, start = self.start, end = self.end)
        raw = raw.Close.to_frame()
        raw.rename(columns = {"Close": "Price"}, inplace = True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        self.data = raw

    def prepare_data(self):
        data = self.data.copy()
        data["SMA_S"] = data.Price.rolling(self.sma_s).mean()
        data["SMA_L"] = data.Price.rolling(self.sma_l).mean()
        self.data = data

    def set_sma_params(self, sma_s = None, sma_l = None):
        if sma_s is not None:
            self.sma_s = sma_s
            self.data["SMA_S"] = self.data.Price.rolling(self.sma_s).mean()
        if sma_l is not None:
            self.sma_l = sma_l
            self.data["SMA_L"] = self.data.Price.rolling(self.sma_l).mean()

    def test_strategy(self):
        data = self.data.copy().dropna()

        # Determine the position (1 for long, -1 for short) // that is the core strategy of the SMA
        data["Position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)

        # Other part is same for all strategies (calculating the strategy returns and cumulative returns)
        data["Strategy"] = data["Position"].shift(1) * data["Returns"]
        data.dropna()
        data["Creturns"] = data["Returns"].cumsum().apply(np.exp)
        data["Cstrategy"] = data["Strategy"].cumsum().apply(np.exp)
        self.results = data

        perf = data["Cstrategy"].iloc[-1]
        outperf = perf - data["Creturns"].iloc[-1]  # To calculate the profit of the strategy subtract the profit of buy and hold
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        if self.results is None:
            print("No results to plot")
        else:
            title = "{} | SMA_S = {} | SMA_L = {}".format(self._ticker, self.sma_s, self.sma_l)
            self.results[["Creturns", "Cstrategy"]].plot(title = title, figsize = (12, 8))
            plt.show()


    def optimize_sma_params(self, sma_s_range, sma_l_range):
        """
        Optimize the parameters for the Simple Moving Average (SMA) strategy.

        Args:
            sma_s_range (tuple): A tuple representing the range of values for the short SMA parameter.
            sma_l_range (tuple): A tuple representing the range of values for the long SMA parameter.

        Returns:
            tuple: A tuple containing the best parameters (short SMA, long SMA) and the corresponding performance.

        """
        combinations = list(product(range(*sma_s_range), range(*sma_l_range)))
        
        results = []
        for combination in combinations:
            self.set_sma_params(combination[0], combination[1])
            results.append(self.test_strategy()[0])

        best_perf = np.max(results)
        best_params = combinations[np.argmax(results)]

        self.set_sma_params(best_params[0], best_params[1])
        self.test_strategy()

        results_df = pd.DataFrame(data=combinations, columns=["SMA_S", "SMA_L"])
        results_df["Performance"] = results
        self.results_overview = results_df

        return best_params, best_perf

tester = SMABacktester("AAPL", 50, 200, "2010-01-01", "2019-12-31")
tester.optimize_sma_params((10, 50, 1), (100, 252, 1))
tester.test_strategy()
tester.plot_results()
