
import smart.models.GOAT as GOAT

from smart.utils.set_seeds import set_seeds
from smart.utils.class_weight import cw
from smart.utils.data_scaler import scale_data
from smart.utils.split_data import split_data
from smart.utils.column_lagger import lag_columns
from smart.quarries.singleTickerRequester import get_single_ticker_data
from smart.modules.financialDataProvider import add_financial_params
from smart.features.ta import buildFeatures, buildImportantFeatures
from smart.predict.binaryClassification import binary_classification_prediction
from smart.models.training import train_nn_model

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Command-Line Script for GOAT Model

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
SAVE_CHECKPOINT = False

if __name__ == "__main__":

    # Prepare Data
    data = get_single_ticker_data(
        ticker=TICKER, interval="1d", start_date="1990-01-01")
    df, cols = buildFeatures(df=data)

    # df, cols = createAlphaOneFeatures(df=data)
    df, cols = lag_columns(df=df, lag_count=6, cols=cols)
    df, cols = add_financial_params(df, cols, ticker=TICKER)

    train, test = split_data(df, split_ratio=SPLIT_RATIO)
    train_s, test_s = scale_data(train, test)

    x_train = train_s[cols]
    y_train = train["dir"]
    cw = cw(train)

    set_seeds(42)
    nn = GOAT.model(hl=HIDDEN_LAYERS,
                    hu=HIDDEN_UNITS,
                    dropout=DROPOUT,
                    rate=DROPOUT_RATE,
                    learning_rate=LEARNING_RATE,
                    regularize=REGULARIZE,
                    shape=(x_train.shape[1],))
    train_nn_model(nn=nn,
                   x=x_train,
                   y=y_train,
                   epochs=EPOCHS,
                   verbose=VERBOSE,
                   validation_split=VALIDATION_SPLIT,
                   shuffle=SHUFFLE,
                   # class weigth optimization added to reduce class imbalance but it is seen that it can be unnecessary for some datasets even harmful
                   cw=cw,
                   save_checkpoint=SAVE_CHECKPOINT)

    x_test = test_s[cols]
    y_test = test["dir"]

    binary_classification_prediction(nn=nn,
                                     x=x_test,
                                     y=y_test,
                                     data=test,
                                     short_boundary=SHORT_BOUNDARY,
                                     long_boundary=LONG_BOUNDARY,
                                     use_dynamic_boundaries=USE_DYNAMIC_BOUNDARIES,
                                     plot_performance_graph=PLOT_PERFORMANCE_GRAPH)
