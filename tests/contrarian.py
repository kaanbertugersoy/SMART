
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ConBacktester:
    
    def __init__ (self, ticker, start, end, transaction_cost = 0.0):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.transaction_cost = transaction_cost
        self.results = None
        self.data = None
        self.get_data()
        self.prepare_data()

    def __repr__(self):
        return "ConBacktester({},{},{},{},{})".format(self._ticker, self.start, self.end, self.transaction_cost)

    # Same for all strategies
    def get_data(self):
        raw = yf.download(self._ticker, start = self.start, end = self.end)
        raw = raw.Close.to_frame()
        raw.rename(columns = {"Close": "Price"}, inplace = True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        self.data = raw

    def test_strategy(self, window = 1):
        self.window = window

        data = self.data.copy().dropna()

        # Determine the position (1 for long, -1 for short) // that is the core of the Contrarian strategy
        data["Position"] = -np.sign(data["Returns"].rolling(self.window).mean())

        # Other part is same for all strategies (calculating the strategy returns and cumulative returns)
        data["Strategy"] = data["Position"].shift(1) * data["Returns"]
        data.dropna()

        # Subtract transaction costs from the strategy returns
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
            title = "{} | Window = {} | Transaction Cost = {}".format(self._ticker, self.window, self.transaction_cost)
            self.results[["Creturns", "Cstrategy"]].plot(title = title, figsize = (12, 8))
            plt.show()
    
    def optimize_window(self, window_range):

        windows = range(*window_range)
        
        results = []
        for window in windows:
            results.append(self.test_strategy(window)[0])

        best_perf = np.max(results)
        best_window = windows[np.argmax(results)]

        self.test_strategy(best_window)

        results_df = pd.DataFrame(data = {"Window": windows})
        results_df["Performance"] = results
        self.results_overview = results_df

        return best_window, best_perf


tester = ConBacktester("AAPL", 50, 200, "2010-01-01", "2019-12-31")
tester.test_strategy()
tester.plot_results()