import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta

from smart.convexOptimization import optimal_portfolio
from smart.services.archer.singleTickerRequester import get_single_ticker_data, compute_and_add_log_returns
from smart.utils.split_data import split_data
from smart.utils.data_scaler import scale_data
from smart.utils.set_seeds import set_seeds
from smart.learning.models.training import train_nn_model
import smart.learning.models.tsa_si_ann_v1 as Model


def plot_portfolio_weights(weights_series):
    plt.figure(figsize=(10, 6))
    weights_series.plot(kind='bar', color='skyblue')
    plt.title('Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()


# Old test code, should be refactored

def NNBasedPortfolioOptimizationTest():

    # Get tickers
    with open('targets.txt', 'r') as f:
        tickers = [line.strip() for line in f]

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

    train, test = split_data(returns_df, split_ratio=0.75)
    # train_s, test_s = scale_data(train, test)

    x_train = train[tickers].iloc[:-1].values
    y_train = train[tickers].iloc[1:].values

    set_seeds(42)
    nn = Model.build_ann_model(hl=3,
                               hu=64,
                               dropout=True,
                               rate=0.25,
                               learning_rate=0.001,
                               regularize=False,
                               input_shape=(x_train.shape[1],),
                               output_shape=y_train.shape[1])
    train_nn_model(nn=nn,
                   x=x_train,
                   y=y_train,
                   epochs=64,
                   verbose=True,
                   validation_split=None,
                   shuffle=False,
                   cw=None,
                   save_checkpoint=False)

    nn.evaluate(test[tickers].iloc[:-1].values,
                test[tickers].iloc[1:].values)
    pred = nn.predict(test[tickers].iloc[:-1].values)

    returns_list = pred[-y_train.shape[1]:].tolist()
    # returns_list = pred.iloc[-1].values.tolist()

    weights = optimal_portfolio(returns_list, tickers)
    weights_series = pd.Series(weights, index=tickers)

    print(returns_df.shape)
    print(weights_series.nlargest(10))
    plot_portfolio_weights(weights_series)
