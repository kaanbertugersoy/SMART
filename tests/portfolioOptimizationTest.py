import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta

from smart.convexOptimization import optimal_portfolio
from smart.services.archer.singleTickerRequester import get_single_ticker_data, compute_and_add_log_returns


MAX_ASSETS = 1
MAX_WEIGHT = 1 / MAX_ASSETS
RISK_AVERSION = 1.0


def portfolioOptimizationTest():

    # Get tickers
    with open('targets.txt', 'r') as f:
        tickers = [line.strip() for line in f]

    tickers = tickers[:50]  # Limit the number of tickers for testing purposes
    successful_tickers = []
    returns_dict = {}
    cutoff_date = datetime.now().date() - timedelta(days=365)
    for ticker in tqdm(tickers, desc='Processing tickers', unit='ticker'):
        try:
            data = get_single_ticker_data(
                ticker=ticker, interval="1d", start_date="1990-01-01")
            data = compute_and_add_log_returns(data)

            last_data_date = data["CloseLogReturn"].dropna().index[-1].date()
            if last_data_date < cutoff_date:
                print(
                    f"{ticker} data is not up to date (last available date: {last_data_date}) and will be skipped.")
                continue

            first_data_date = data["CloseLogReturn"].dropna().index[0].date()
            if cutoff_date - first_data_date < timedelta(days=365*2):
                print(
                    f"{ticker} data length is not sufficient for this operation and will be skipped.")
                continue

            returns_dict[ticker] = data["CloseLogReturn"]
            successful_tickers.append(ticker)
        except:
            print(f"{ticker} is skipped.")
            continue

    tickers = successful_tickers

    # Combine the returns data into DataFrame
    returns_df = pd.concat(returns_dict.values(), axis=1,
                           keys=returns_dict.keys())
    returns_df.dropna(inplace=True)
    returns_list = returns_df.values.tolist()

    weights = optimal_portfolio(
        X=returns_list, tickers=tickers, max_weight=MAX_WEIGHT, risk_aversion=RISK_AVERSION)
    weights_series = pd.Series(weights, index=tickers)

    print(returns_df.shape)
    print(weights_series.nlargest(MAX_ASSETS+5))
    plot_portfolio_weights(weights_series)


def plot_portfolio_weights(weights_series):
    plt.figure(figsize=(10, 6))
    weights_series.plot(kind='bar', color='skyblue')
    plt.title('Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()
