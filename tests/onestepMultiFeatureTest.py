
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax, Nadam, RMSprop, SGD  # type: ignore
from sklearn.preprocessing import StandardScaler

from smart.utils.set_seeds import set_seeds
from smart.services.archer.singleTickerRequester import get_single_ticker_data, compute_and_add_log_returns
from smart.benchmark.monitor import monitor_train_loss
from smart.benchmark.randomWalk import random_walk_forecast_from_series
from smart.pipeline.oneStepAhead import one_step_ahead_forecast_pipeline
from smart.services.archer.features.ta import buildFeatures, buildImportantFeatures
from smart.benchmark.compareScore import report_scores

from smart.learning.models.tsa_si_lstm_v1 import NNModel as Model, model_config


TICKER = "TSLA"
n_test = 20
T = 10


def onestepMultiFeatureTest():
    set_seeds(42)
    df = get_single_ticker_data(
        ticker=TICKER, interval="1d", start_date="2020-01-01")
    df = compute_and_add_log_returns(df, columns=['Close'])
    df, cols = buildImportantFeatures(df=df)
    cols.append('CloseLogReturn')

    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train[cols])
    test_s = scaler.transform(test[cols])

    df.loc[train_idx, cols] = train_s
    df.loc[test_idx, cols] = test_s
    print(train_idx.shape)
    # ----------------------------------------------------------------

    series = df['CloseLogReturn'].dropna().to_numpy()

    # Basic ANN
    # number_of_features = series.shape[1]
    model = Model.model_build_fn((T, 1), 1)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    Ptrain, Ptest, r1, model, X_test, Y_test = one_step_ahead_forecast_pipeline(
        series, T, n_test, model, is_lstm=model_config['lstm_spec_seq'])

    # ----------------------------------------------------------------
    # One-step forecast using true targets
    train_idx[:T] = False  # first T+1 values are not predictable

    data = df.copy()
    data.loc[train_idx, 'CloseLogReturn'] = Ptrain.flatten()
    data.loc[test_idx, 'CloseLogReturn'] = Ptest.flatten()

    inv_train = pd.DataFrame(scaler.inverse_transform(
        data.iloc[train_idx][cols]), index=data.iloc[train_idx].index, columns=cols)
    inv_test = pd.DataFrame(scaler.inverse_transform(
        data.iloc[test_idx][cols]), index=data.iloc[test_idx].index, columns=cols)

    # Store predictions
    df.loc[train_idx, 'ANN Train Prediction'] = inv_train[['CloseLogReturn']]
    df.loc[test_idx, 'ANN Test Prediction'] = inv_test[['CloseLogReturn']]

    # Needed to compute un-differenced predictions
    df['ShiftCloseLogReturn'] = df['CloseLogReturn'].shift(1)
    prev = df['ShiftCloseLogReturn']

    # Last-known train value
    last_train = train.iloc[-1]['CloseLogReturn']

    # 1-step forecast
    df.loc[train_idx, 'singlestep_train'] = Ptrain
    df.loc[test_idx, 'singlestep'] = Ptest

    # ----------------------------------------------------------------

    # Our model should at least beat the random walk
    random_walk_forecast = random_walk_forecast_from_series(
        df['CloseLogReturn'])
    df.loc[test_idx, 'Random Walk'] = random_walk_forecast[test_idx]

    # Monitor
    monitor_train_loss([r1])

    # Plot the forecast
    # cols = ['CloseLogReturn',
    #         'ANN Train Prediction',
    #         'ANN Test Prediction']
    # df[cols].plot(figsize=(15, 5))
    # plt.show()

    # Plot all forecasts
    cols = ['CloseLogReturn', 'singlestep_train', 'singlestep', 'Random Walk']
    df.iloc[-100:][cols].plot(figsize=(15, 5))
    plt.show()

    report_scores(analog=df.iloc[-n_test:]['CloseLogReturn'],
                  comparison_list=[('single-step', df.loc[test_idx, 'singlestep']),
                                   ('Random Walk', df.loc[test_idx, 'Random Walk'])],
                  plot=False)
