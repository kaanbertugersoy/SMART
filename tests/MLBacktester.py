import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import yfinance as yf
plt.use("seaborn-v0_8")


class MLBacktester():
    def __init__(self, ticker, start, end, tc):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.tc = tc
        self.model = LogisticRegression(
            C=1e6, multi_class="ovr", max_iter=100000)
        self.results = None
        self.get_data()

    def __repr__(self):
        return "MLBacktester({},{},{},{})".format(self.ticker, self.start, self.end, self.tc)

    def get_data(self):
        raw = yf.download(self.ticker, start=self.start, end=self.end)
        raw = raw.Close.to_frame()
        raw.rename(columns={"Close": "Price"}, inplace=True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        self.data = raw

    def split_data(self, start, end):
        data = self.data[(self.data.index >= start) &
                         (self.data.index < end)].copy()
        return data

    def prepare_features(self, start, end):
        self.data_subset = self.split_data(start, end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1):
            col = f"lag{lag}"
            self.data_subset[col] = self.data_subset["Returns"].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)

    def fit_model(self, start, end):
        self.prepare_features(start, end)
        X = self.data_subset[self.feature_columns]
        y = np.sign(self.data_subset["Returns"])
        self.model.fit(X, y)

    def test_strategy(self, train_ratio=0.7, lags=5):

        self.lags = lags

        # split data
        full_data = self.data.copy().dropna()
        split_index = int(len(full_data) * train_ratio)
        split_date = full_data.index[split_index - 1]
        train_start = full_data.index[0]
        test_end = full_data.index[-1]

        # fit model
        self.fit_model(train_start, split_date)

        # prepare test set
        self.prepare_features(split_date, test_end)

        # predict and backtest
        predict = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset["pred"] = predict

        # calculate strategy returns
        self.data_subset["strategy"] = self.data_subset["pred"] * \
            self.data_subset["Returns"]

        # number of trades
        self.data_subset["trades"] = self.data_subset.pred.diff().fillna(
            0).abs()

        # subtract transaction costs from return when trade occurs
        self.data_subset.strategy -= self.data_subset.trades * self.tc

        # calculate cumulative returns
        self.data_subset["creturns"] = self.data_subset["Returns"].cumsum().apply(
            np.exp)
        self.data_subset["cstrategy"] = self.data_subset["strategy"].cumsum().apply(
            np.exp)

        perf = self.data_subset["cstrategy"].iloc[-1]
        outperf = perf - self.data_subset["creturns"].iloc[-1]

        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        if self.data_subset["cstrategy"] is None:
            print("No results to plot")
        else:
            title = "{} | tc = {}".format(self.ticker, self.tc)
            self.data_subset[["creturns", "cstrategy"]].plot(
                title=title, figsize=(12, 8))
            plt.show()

    if __name__ == "__main__":
        pass


tester = MLBacktester("AAPL", "2010-01-01", "2020-12-31", 0)

tester.test_strategy()
tester.plot_results()
