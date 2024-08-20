import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam  # type: ignore

import smart.learning.models.tsa_si_ann_v1 as Model

from smart.utils.set_seeds import set_seeds
from smart.utils.class_weight import cw
from smart.utils.data_scaler import scale_data
from smart.utils.split_data import split_data
from smart.utils.column_lagger import lag_columns
from smart.services.archer.singleTickerRequester import get_single_ticker_data
from smart.services.archer.financialDataProvider import add_financial_params
from smart.services.archer.features.ta import buildFeatures, buildImportantFeatures
from smart.learning.evaluation.binaryClassification import binary_classification_prediction
from smart.learning.models.training import train_nn_model
from smart.learning.models.compile import compile_model

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Command-Line Script for GOAT Model

TICKER = "NVDA"

SPLIT_RATIO = 0.75

# Hyperparameters
HIDDEN_LAYERS = 3
HIDDEN_UNITS = 64
DROPOUT = True
DROPOUT_RATE = 0.25
LEARNING_RATE = 0.001
REGULARIZE = False

EPOCHS = 64
VERBOSE = True
VALIDATION_SPLIT = 0.2
SHUFFLE = True

SHORT_BOUNDARY = 0.47  # now it is hardcoded in the next versions it will be calculated
LONG_BOUNDARY = 0.53
USE_DYNAMIC_BOUNDARIES = False
PLOT_PERFORMANCE_GRAPH = True
SAVE_CHECKPOINT = False

n_test = 21


def binClassTest():

    # Prepare Data
    data = get_single_ticker_data(
        ticker=TICKER, interval="1d", start_date="2020-01-01")
    df, cols = buildFeatures(df=data)

    # df, cols = createAlphaOneFeatures(df=data)
    # df, cols = lag_columns(df=df, lag_count=6, cols=cols)
    # df, cols = add_financial_params(df, cols, ticker=TICKER)

    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]

    # train, test = split_data(df, split_ratio=SPLIT_RATIO)
    train_s, test_s = scale_data(train, test)

    train_idx = df.index <= train.index[-1]
    test_idx = df.index > train.index[-1]

    x_train = train_s[cols]
    y_train = train["returns"]
    CW = cw(train)

    print(x_train.shape)

    set_seeds(42)
    # nn = build_nbeats_model(input_shape=(x_train.shape[1],), output_shape=(1,))
    # nn.compile(optimizer='adam', loss="binary_crossentropy",
    #            metrics=['accuracy'])
    nn = Model.build_ann_model(
        input_shape=(x_train.shape[1],), output_shape=1)

    nn = compile_model(nn, optimizer_class=Adam, learning_rate=LEARNING_RATE,
                       loss="mse", metrics=['mse'])

    train_nn_model(nn=nn,
                   x=x_train,
                   y=y_train,
                   epochs=EPOCHS,
                   verbose=VERBOSE,
                   validation_split=VALIDATION_SPLIT,
                   shuffle=SHUFFLE,
                   # class weigth optimization added to reduce class imbalance but it is seen that it can be unnecessary for some datasets even harmful
                   cw=CW,
                   save_checkpoint=SAVE_CHECKPOINT,
                   callbacks=None)

    x_test = test_s[cols]
    y_test = test["returns"]

    Ptrain = nn.predict(x_train).flatten()
    Ptest = nn.predict(x_test).flatten()

    df.loc[train_idx, 'Train Prediction'] = Ptrain
    df.loc[test_idx, 'Test Prediction'] = Ptest

    # Plot the forecast
    cols = ['returns',
            'Train Prediction',
            'Test Prediction']
    df.iloc[-100:][cols].plot(figsize=(15, 5))
    plt.show()

    # binary_classification_prediction(nn=nn,
    #                                  x=x_test,
    #                                  y=y_test,
    #                                  data=test,
    #                                  short_boundary=SHORT_BOUNDARY,
    #                                  long_boundary=LONG_BOUNDARY,
    #                                  use_dynamic_boundaries=USE_DYNAMIC_BOUNDARIES,
    #                                  plot_performance_graph=PLOT_PERFORMANCE_GRAPH)
