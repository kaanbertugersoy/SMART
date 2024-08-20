from smart.utils.set_seeds import set_seeds

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from smart.services.archer.multipleTickerRequester import get_multiple_ticker_data
from smart.learning.models.tsa_mi_ann_v1 import NNModel as Model, model_config


def multiInputMultiOutputTest():

    tickers = ["AMZN", "GOOGL", "MSFT", "AAPL"]

    set_seeds(42)
    df = get_multiple_ticker_data(
        tickers, start_date='2014-01-01', interval="1d")

    df_swapped = df.swaplevel(0, 1, axis=1).sort_index(axis=1)

    cols = ['Close']

    Ntest = 20
    Ntrain = len(df_swapped) - Ntest
    T = len(cols)
    D = len(tickers)
    # Shape (N, T, D) => N is the timeseries length, T is the number of features, D is the number of tickers

    # Minus 1 because of the shift creates NaN values and we remove them
    Xtrain = np.zeros((Ntrain - 1, T, D))
    Xtest = np.zeros((Ntest - 1, T, D))

    Ytrain = np.zeros((Ntrain - 1, D))
    Ytest = np.zeros((Ntest - 1, D))

    df_train = df_swapped.iloc[:-Ntest]
    df_test = df_swapped.iloc[-Ntest:]

    ticker_order = []

    for d, col in enumerate(df_train.columns.get_level_values(0).unique()):
        ticker_order.append(col)
        X_shifted = df_train[col][cols].shift(1)
        Y_actual = df_train[col][cols]

        X_np = X_shifted.to_numpy()
        Y_np = Y_actual.to_numpy().flatten()

        valid_indices = ~np.isnan(X_np).any(axis=1)
        Xtrain[:, :, d] = X_np[valid_indices]
        Ytrain[:, d] = Y_np[valid_indices]

    for d, col in enumerate(df_test.columns.get_level_values(0).unique()):
        X_shifted = df_test[col][cols].shift(1)
        Y_actual = df_test[col][cols]

        X_np = X_shifted.to_numpy()
        Y_np = Y_actual.to_numpy().flatten()

        valid_indices = ~np.isnan(X_np).any(axis=1)
        Xtest[:, :, d] = X_np[valid_indices]
        Ytest[:, d] = Y_np[valid_indices]

    print(Ytrain)

    # plt.plot(Xtrain[:, :, 1])
    # plt.show()

    # ANN
    model = Model.model_build_fn(
        number_of_tails=D, tail_input_shape=(T,), output_shape=D)

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.001))

    Xtrain_split = []
    Xtest_split = []
    for d in range(D):
        Xtrain_split.append(Xtrain[:, :, d])
        Xtest_split.append(Xtest[:, :, d])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                           patience=5, verbose=1, mode='min')

    r = model.fit(
        Xtrain_split,
        Ytrain,
        epochs=50,
        validation_data=(Xtest_split, Ytest),
        callbacks=[es, lr]
    )

    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='test loss')
    plt.legend()
    plt.show()

    Ptest = model.predict(Xtest_split)
    print(Ytest)
    print(Ptest)
    # np.mean(np.argmax(Ptest, axis=1) == Ytest)

    time_points = range(Ytest.shape[0])
    for i, col in enumerate(ticker_order):
        plt.figure(figsize=(10, 4))
        plt.plot(time_points, Ytest[:, i],
                 label=f'True Series {col}', marker='o')
        plt.plot(time_points, Ptest[:, i],
                 label=f'Predicted Series {col}', marker='x')
        plt.title(f'Series {col}')
        plt.xlabel('Time Point')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    # Calculate performance metrics for each series
    for i, col in enumerate(ticker_order):
        mae = mean_absolute_error(Ytest[:, i], Ptest[:, i])
        mse = mean_squared_error(Ytest[:, i], Ptest[:, i])
        r2 = r2_score(Ytest[:, i], Ptest[:, i])
        print(f'Series {col} - MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}')
