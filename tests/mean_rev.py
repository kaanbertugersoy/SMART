from itertools import product
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MeanRevBacktester:

    def __init__ (self, ticker, sma, dev, start, end, transaction_cost):
        self._ticker = ticker
        self.sma = sma
        self.dev = dev
        self.start = start
        self.end = end
        self.transaction_cost = transaction_cost
        self.results = None
        self.data = None
        self.get_data()
        self.prepare_data()

    def __repr__(self):
        return "MeanRevBacktester({},{},{},{},{})".format(self._ticker, self.sma, self.dev, self.start, self.end)

    def get_data(self):
        raw = yf.download(self._ticker, start = self.start, end = self.end)
        raw = raw.Close.to_frame()
        raw.rename(columns = {"Close": "Price"}, inplace = True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        self.data = raw

    def prepare_data(self):
        data = self.data.copy()
        data["SMA"] = data.Price.rolling(self.sma).mean()
        data["Lower"] = data.SMA - data.Price.rolling(self.sma).std() * self.dev
        data["Upper"] = data.SMA + data.Price.rolling(self.sma).std() * self.dev
        self.data = data

    def set_sma_params(self, sma = None, dev = None):
        if sma is not None:
            self.sma = sma
            self.data["SMA"] =  self.data.Price.rolling(self.sma).mean()
            self.data["Lower"] =  self.data.SMA - self.data.Price.rolling(self.sma).std() * self.dev
            self.data["Upper"] =  self.data.SMA +  self.data.Price.rolling(self.sma).std() * self.dev
        if dev is not None:
            self.dev = dev
            self.data["Lower"] =  self.data.SMA - self.data.Price.rolling(self.sma).std() * self.dev
            self.data["Upper"] =  self.data.SMA +  self.data.Price.rolling(self.sma).std() * self.dev

    def test_strategy(self):
        data = self.data.copy().dropna()

        # Determine the position (1 for long, -1 for short) // that is the core strategy of the SMA
        data["Distance"] = data.Price - data.SMA
        data["Position"] = np.where(data.Price < data.Lower, 1, np.nan)
        data["Position"] = np.where(data.Price > data.Upper, -1, data.Position)
        data["Position"] = np.where(data.Distance * data.Distance.shift(1) < 0, 0, data.Position)
        data["Position"] = data.Position.ffill().fillna(0)

        # Other part is same for all strategies (calculating the strategy returns and cumulative returns)
        data["Strategy"] = data["Position"].shift(1) * data["Returns"]
        data.dropna()

        data["Trades"] = data.Position.diff().fillna(0).abs()  # Calculate the number of trades
        data["Strategy"] = data["Strategy"] - data["Trades"] * self.transaction_cost

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
            title = "{} | SMA = {} | dev = {} | TC = {}".format(self._ticker, self.sma, self.dev, self.transaction_cost)
            self.results[["Creturns", "Cstrategy"]].plot(title = title, figsize = (12, 8))
            plt.show()


    def optimize_sma_params(self, sma_range, dev_range):
        combinations = list(product(range(*sma_range), np.arange(*dev_range)))
        
        results = []
        for combination in combinations:
            self.set_sma_params(combination[0], combination[1])
            results.append(self.test_strategy()[0])

        best_perf = np.max(results)
        best_params = combinations[np.argmax(results)]

        self.set_sma_params(best_params[0], best_params[1])
        self.test_strategy()

        results_df = pd.DataFrame(data=combinations, columns=["SMA", "dev"])
        results_df["Performance"] = results
        self.results_overview = results_df

        return best_params, best_perf

tester = MeanRevBacktester("AAPL", 50, 200, "2010-01-01", "2019-12-31")
tester.optimize_sma_params((10, 50, 1), (100, 252, 1))
tester.test_strategy()
tester.plot_results()