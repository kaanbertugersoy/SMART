
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

from smart.services.archer.singleTickerRequester import get_single_ticker_data
from smart.utils.set_seeds import set_seeds
from smart.utils.data_scaler import scale_data
from smart.pipeline.multioutput import multioutput_forecast_pipeline
from smart.pipeline.multiStepAhead import multistep_forecast_pipeline
from smart.pipeline.oneStepAhead import one_step_ahead_forecast_pipeline
from smart.benchmark.randomWalk import random_walk_forecast_from_series
from smart.benchmark.compareScore import report_scores
from smart.benchmark.monitor import monitor_train_loss

from smart.learning.models.tsa_si_lstm_v1 import NNModel as Model

TICKER = "GOOGL"
n_test = 20
T = 20


def onestepMultistepMultioutputTest():

    set_seeds(42)
    data = get_single_ticker_data(
        ticker=TICKER, interval="1d", start_date="2020-01-01")

    df = data['Close'].to_frame()
    df['LogClose'] = np.log(df['Close'])
    df['DiffLogClose'] = df['LogClose'].diff()
    # df['LogVolume'] = np.log(df['volume'])

    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train[['DiffLogClose']])
    test_s = scaler.transform(test[['DiffLogClose']])

    df.loc[train_idx, 'ScaledDiffLogClose'] = train_s.flatten()
    df.loc[test_idx, 'ScaledDiffLogClose'] = test_s.flatten()

    # ----------------------------------------------------------------

    series = df['ScaledDiffLogClose'].dropna().to_numpy()

    # Basic ANN
    model = Model.model_build_fn((T, 1), 1)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    Ptrain, Ptest, r1, model, X_test, Y_test = one_step_ahead_forecast_pipeline(
        series, T, n_test, model, is_lstm=True)

    # ----------------------------------------------------------------

    Ptrain = scaler.inverse_transform(Ptrain.reshape(-1, 1)).flatten()
    Ptest = scaler.inverse_transform(Ptest.reshape(-1, 1)).flatten()

    # One-step forecast using true targets
    train_idx[:T+1] = False  # first T+1 values are not predictable

    # Store diff predictions
    df.loc[train_idx, 'Diff ANN Train Prediction'] = Ptrain
    df.loc[test_idx, 'Diff ANN Test Prediction'] = Ptest

    # Needed to compute un-differenced predictions
    df['ShiftLogClose'] = df['LogClose'].shift(1)
    prev = df['ShiftLogClose']

    # Last-known train value
    last_train = train.iloc[-1]['LogClose']

    # 1-step forecast
    df.loc[train_idx, 'singlestep_train'] = prev[train_idx] + Ptrain
    df.loc[test_idx, 'singlestep'] = prev[test_idx] + Ptest

    # ----------------------------------------------------------------
    # multi-step forecast

    multistep_preds = multistep_forecast_pipeline(
        n_test=n_test, pre_trained_model=model, X_test=X_test)

    multistep_preds = scaler.inverse_transform(
        multistep_preds.reshape(-1, 1)).flatten()

    # save multi-step forecast to dataframe
    df.loc[test_idx, 'multistep'] = last_train + \
        np.cumsum(multistep_preds)

    # ----------------------------------------------------------------
    # multi-output forecast

    Tx, Ky = T, n_test

    # multioutput forecast requires a different model because of the different output shape
    model = Model.model_build_fn((Tx, 1), Ky)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    multi_output_test_pred, r2 = multioutput_forecast_pipeline(
        T, n_test, series, model, is_lstm=True)

    multi_output_test_pred = scaler.inverse_transform(
        multi_output_test_pred.reshape(-1, 1)).flatten()

    # save multi-output forecast to dataframe
    df.loc[test_idx, 'multioutput'] = last_train + \
        np.cumsum(multi_output_test_pred)

    # ----------------------------------------------------------------

    # Our model should at least beat the random walk
    random_walk_forecast = random_walk_forecast_from_series(df['LogClose'])
    df.loc[test_idx, 'Random Walk'] = random_walk_forecast[test_idx]

    # Monitor
    monitor_train_loss([r1, r2])

    # Plot the forecast
    cols = ['DiffLogClose',
            'Diff ANN Train Prediction',
            'Diff ANN Test Prediction']
    df[cols].plot(figsize=(15, 5))
    plt.show()

    # Plot all forecasts
    cols = ['LogClose', 'singlestep_train', 'singlestep',
            'multistep', 'multioutput', 'Random Walk']
    df.iloc[-100:][cols].plot(figsize=(15, 5))
    plt.show()

    report_scores(analog=df.iloc[-n_test:]['LogClose'],
                  comparison_list=[('single-step', df.loc[test_idx, 'singlestep']),
                                   ('multi-step',
                                    df.loc[test_idx, 'multistep']),
                                   ('multi-output',
                                    df.loc[test_idx, 'multioutput']),
                                   ('Random Walk', df.loc[test_idx, 'Random Walk'])],
                  plot=False)
