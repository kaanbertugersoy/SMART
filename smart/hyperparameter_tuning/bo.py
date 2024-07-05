# Import packages
from smart.utils.data_scaler import scale_data
from smart.utils.split_data import split_data
from smart.features._deprecated import createAlphaOneFeatures, lagAlphaOneFeatures
from smart.quarries.singleTickerRequester import get_single_ticker_data
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from scikeras.wrappers import KerasClassifier

LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

score_acc = make_scorer(accuracy_score)

# OLD CODE - DO NOT USE
# TESTING BAYESIAN OPTIMIZATION => NOT RELIABLE STATE


raw_data = get_single_ticker_data(
    ticker="GOOGL", interval="1d", start_date="2000-01-01")
df = createAlphaOneFeatures(raw_data)
features = ["dir", "sma", "ema", "rsi"]
df_lagged, cols = lagAlphaOneFeatures(df=df, lag_n=8, features=features)
train, test = split_data(df_lagged, split_ratio=0.666)
train_s, test_s = scale_data(train, test)
input_dim = len(cols)
X_train_s = train_s[cols]

params_nn = {
    'neurons': (10, 100),
    'activation': (0, 9),
    'optimizer': (0, 7),
    'learning_rate': (0.01, 1),
    'batch_size': (200, 1000),
    'epochs': (20, 100)
}


def nn_cl_bo(neurons, activation, optimizer, learning_rate,  batch_size, epochs):
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta',
                  'Adagrad', 'Adamax', 'Nadam', 'Ftrl', 'SGD']
    optimizerD = {'Adam': tf.keras.optimizers.Adam(learning_rate=learning_rate), 'SGD': tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate), 'Adadelta': tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
                  'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate), 'Adamax': tf.keras.optimizers.Adamax(learning_rate=learning_rate),
                  'Nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate), 'Ftrl': tf.keras.optimizers.Ftrl(learning_rate=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU, 'relu']
    neurons = round(neurons)
    activation = activationL[round(activation)]
    batch_size = round(batch_size)
    epochs = round(epochs)

    def nn_cl_fun():
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        nn = tf.keras.models.Sequential()
        nn.add(tf.keras.layers.Dense(
            neurons, input_dim=input_dim, activation=activation))
        nn.add(tf.keras.layers.Dense(neurons, activation=activation))
        nn.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy',
                   optimizer=opt, metrics=['accuracy'])
        return nn

    es = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy', mode='max', verbose=0, patience=20)
    nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size,
                         verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(nn, X_train_s, train["dir"], scoring=score_acc,
                            cv=kfold, fit_params={'callbacks': [es]}).mean()
    return score


# Run Bayesian Optimization
nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=42)
nn_bo.maximize(init_points=25, n_iter=4)


params_nn_ = nn_bo.max['params']
activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
               'elu', 'exponential', LeakyReLU, 'relu']
params_nn_['activation'] = activationL[round(params_nn_['activation'])]
params_nn_
