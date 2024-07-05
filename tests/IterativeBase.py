import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


class IterativeBase:
    def __init__(self, ticker, start, end, amount):
        self._ticker = ticker
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.get_data()

    def get_data(self):
        raw = yf.download(self._ticker, start=self.start, end=self.end)
        raw = raw.Close.to_frame()
        raw.rename(columns={"Close": "Price"}, inplace=True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        self.data = raw

    def plot_data(self, cols=None):
        if cols is None:
            cols = ["Price"]
        self.data[cols].plot(
            figsize=(12, 8), title=f"Price Chart: {self._ticker}", fontsize=15)
        plt.show()

    def get_values(self, bar):
        date = str(self.data.index[bar].date())
        price = round(self.data.Price.iloc[bar], 5)
        # spread = round(self.data.Spread.iloc[bar], 5)
        return date, price

    def print_current_balance(self, bar):
        date, price = self.get_values(bar)
        print(f"{date} | Current Balance: {self.current_balance:.2f}")

    def buy_instrument(self, bar, units=None, amount=None):
        date, price = self.get_values(bar)
        if amount is not None:
            units = int(amount / price)
        self.current_balance -= units * price
        self.units += units
        self.trades += 1
        print(f"{date} |  Buying {units} units at price {price}")

    def sell_instrument(self, bar, units=None, amount=None):
        date, price = self.get_values(bar)
        if amount is not None:
            units = int(amount / price)
        self.current_balance += units * price
        self.units -= units
        self.trades += 1
        print(f"{date} |  Selling {units} units at price {price}")

    def print_current_position_value(self, bar):
        date, price = self.get_values(bar)
        cpv = self.units * price
        print(f"{date} | Current Position Value: {cpv:.2f}")

    def print_current_nav(self, bar):
        date, price = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print(f"{date} | Net Asset Value: {nav:.2f}")

    def close_positions(self, bar):
        date, price = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price
        print("{} | Closing {} units at price {}".format(date, self.units, price))
        self.units = 0
        self.trades += 1
        performance = (self.current_balance -
                       self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar)
        print("{} | Net Performance: {:.2f}%".format(date, performance))
        print("{} | Trades: {}".format(date, self.trades))
        print(75 * "-")
