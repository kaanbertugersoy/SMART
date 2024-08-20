import smart.learning.models.hpt_si_ann_v1 as Model

from smart.utils.split_data import split_data
from smart.utils.data_scaler import scale_data
from smart.utils.class_weight import cw
from smart.utils.set_seeds import set_seeds
from smart.utils.column_lagger import lag_columns
from smart.learning.models.training import train_nn_model
from smart.learning.bayesianOptimization import maximize_bayesian_optimization
from smart.learning.evaluation.binaryClassification import binary_classification_prediction
from smart.services.archer.singleTickerRequester import get_single_ticker_data
from smart.services.archer.financialDataProvider import add_financial_params
from smart.services.archer.features.ta import buildFeatures, buildImportantFeatures

import warnings
warnings.filterwarnings("ignore")

TICKER = "TSLA"
SPLIT_RATIO = 0.75
SHUFFLE = True
VALIDATION_SPLIT = 0.2
VERBOSE = 1

SHORT_BOUNDARY = 0.48
LONG_BOUNDARY = 0.52
SAVE_CHECKPOINT = False
USE_DYNAMIC_BOUNDARIES = False
PLOT_PERFORMANCE_GRAPH = True

ITERATIONS = 100

# Old code, should be refactored for the new version


def bayesianOptimizationTest():
    raw_data = get_single_ticker_data(
        ticker=TICKER, interval="1d", start_date="1990-01-01")
    df, cols = buildFeatures(raw_data)

    df, cols = lag_columns(df=df, lag_count=6, cols=cols)
    df, cols = add_financial_params(df, cols, ticker=TICKER)

    train, test = split_data(df, split_ratio=SPLIT_RATIO)
    train_s, test_s = scale_data(train, test)

    x_train = train_s[cols]
    y_train = train["dir"]
    cw = cw(train)

    set_seeds(42)

    layers, neurons, activations, batch_norms, dropouts, dropout_rate, optimizer, batch_size, epochs = maximize_bayesian_optimization(
        x=x_train, y=y_train, cw=cw, iterations=ITERATIONS)

    # Save parameters in a file

    nn = Model.model_build_fn(layers, neurons, activations, batch_norms,
                              dropouts, dropout_rate, optimizer, shape=(x_train.shape[1],))

    train_nn_model(nn=nn,
                   x=x_train,
                   y=y_train,
                   epochs=epochs,
                   batch_size=batch_size,
                   verbose=VERBOSE,
                   validation_split=VALIDATION_SPLIT,
                   shuffle=SHUFFLE,
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
