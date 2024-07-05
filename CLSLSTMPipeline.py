import numpy as np

import smart.models.LSTMModel as LSTM

from smart.utils.set_seeds import set_seeds
from smart.utils.class_weight import cw
from smart.utils.data_scaler import scale_data
from smart.utils.split_data import split_data
from smart.utils.column_lagger import lag_columns
from smart.models.training import train_nn_model
from smart.quarries.singleTickerRequester import get_single_ticker_data
from smart.modules.financialDataProvider import add_financial_params
from smart.features.ta import buildFeatures, buildImportantFeatures
from smart.predict.binaryClassification import binary_classification_prediction

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Command-Line Script for LSTM Model

TICKER = "TSLA"

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
USE_TRAIN_DATA = False
SAVE_CHECKPOINT = False

TIME_STEPS = 10

if __name__ == "__main__":

    # Prepare Data
    data = get_single_ticker_data(
        ticker=TICKER, interval="1d", start_date="1990-01-01")
    df, cols = buildFeatures(df=data)

    df, cols = lag_columns(df=df, lag_count=6, cols=cols)
    df, cols = add_financial_params(df, cols, ticker=TICKER)

    train, test = split_data(df, split_ratio=SPLIT_RATIO)
    train_s, test_s = scale_data(train, test)

    x_train = train_s[cols].values
    y_train = train["dir"].values
    num_features = x_train.shape[1]
    cw = cw(train)

    num_samples = (x_train.shape[0] // TIME_STEPS) * TIME_STEPS
    x_train = x_train[:num_samples, :]
    y_train = y_train[:num_samples + 1]
    x_train_lstm = np.reshape(
        x_train, (x_train.shape[0] // TIME_STEPS, TIME_STEPS, num_features))

    y_train_lstm = []
    for i in range(0, len(y_train) - 1, TIME_STEPS):
        y_train_lstm.append(y_train[i + TIME_STEPS])
    y_train_lstm = np.expand_dims(y_train_lstm, axis=-1)

    set_seeds(42)
    nn = LSTM.lstm_model(layers=HIDDEN_LAYERS,
                         units=HIDDEN_UNITS,
                         dropout=DROPOUT,
                         rate=DROPOUT_RATE,
                         learning_rate=LEARNING_RATE,
                         regularize=REGULARIZE,
                         time_steps=TIME_STEPS,
                         num_features=(num_features))
    train_nn_model(nn=nn,
                   x=x_train_lstm,
                   y=y_train,
                   epochs=EPOCHS,
                   verbose=VERBOSE,
                   validation_split=VALIDATION_SPLIT,
                   shuffle=SHUFFLE,
                   cw=cw,
                   save_checkpoint=SAVE_CHECKPOINT)

    x_test = test_s[cols].values
    y_test = test["dir"].values

    num_test_samples = (x_test.shape[0] // TIME_STEPS) * TIME_STEPS
    x_test = x_test[:num_test_samples, :]
    y_test = y_test[:num_test_samples + 1]  # +1 for the last prediction

    x_test_lstm = np.reshape(
        x_test, (x_test.shape[0] // TIME_STEPS, TIME_STEPS, num_features))

    y_test_lstm = []
    for i in range(0, len(y_test) - 1, TIME_STEPS):
        y_test_lstm.append(y_test[i + TIME_STEPS])
    y_test_lstm = np.expand_dims(y_test_lstm, axis=-1)

    # THIS LINE THROWS AN ERROR !!!!!!!!!
    # ----------------------------------------------------------------
    # nn.evaluate(x=x_test_lstm, y=y_test_lstm)
    # ----------------------------------------------------------------

    pred = nn.predict(x_test_lstm)
    print(y_test_lstm.shape)
    print(pred.shape)
    # EVEN THOUGH THE SHAPES OF y_test_lstm and pred ARE SAME, BUT EVALUATION THROWS SHAPE ERROR
