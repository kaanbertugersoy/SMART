# Import packages
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import tensorflow as tf

from smart.utils.data_scaler import scale_data
from smart.utils.split_data import split_data
from smart.quarries.singleTickerRequester import get_single_ticker_data

from smart.features._deprecated import createAlphaOneFeatures, lagAlphaOneFeatures

from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from scikeras.wrappers import KerasClassifier

from smart.predict.binaryClassification import binary_classification_prediction

LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

score_acc = make_scorer(accuracy_score)

# OLD CODE - DO NOT USE
# LAGGING FEATURES IS DEPRECATED
# ALPHAONE FEATURES ARE DEPRECATED
# NOT DYNAMIC NEURON STRUCTURE

raw_data = get_single_ticker_data(
    ticker="GOOGL", interval="1d", start_date="2000-01-01")
df, features = createAlphaOneFeatures(raw_data)
df_lagged, cols = lagAlphaOneFeatures(df=df, lag_n=8, features=features)
train, test = split_data(df_lagged, split_ratio=0.75)
train_s, test_s = scale_data(train, test)
X_train_s = train_s[cols]

params_nn2 = {
    'neurons': (10, 200),
    'activation': (0, 9),
    'optimizer': (0, 7),
    'learning_rate': (0.0001, 0.01),
    'batch_size': (200, 1000),
    'epochs': (20, 100),
    'layers1': (1, 3),
    'layers2': (1, 3),
    'normalization': (0, 1),
    'dropout': (0, 1),
    'dropout_rate': (0, 0.3)
}


def nn_cl_bo2(neurons, activation, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, normalization, dropout, dropout_rate):
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta',
                  'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
    optimizerD = {'Adam': tf.keras.optimizers.Adam(learning_rate=learning_rate), 'SGD': tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate), 'Adadelta': tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
                  'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate), 'Adamax': tf.keras.optimizers.Adamax(learning_rate=learning_rate),
                  'Nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate), 'Ftrl': tf.keras.optimizers.Ftrl(learning_rate=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU, 'relu']

    neurons = round(neurons)
    activation = activationL[round(activation)]
    optimizer = optimizerD[optimizerL[round(optimizer)]]
    batch_size = round(batch_size)
    epochs = round(epochs)
    layers1 = round(layers1)
    layers2 = round(layers2)

    def nn_cl_fun():
        nn = tf.keras.models.Sequential()
        nn.add(tf.keras.layers.Input(shape=(X_train_s.shape[1],)))
        if normalization > 0.5:
            nn.add(tf.keras.layers.BatchNormalization())
        for i in range(layers1):
            nn.add(tf.keras.layers.Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(tf.keras.layers.Dropout(dropout_rate, seed=42))
        for i in range(layers2):
            nn.add(tf.keras.layers.Dense(neurons, activation=activation))
        nn.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy',
                   optimizer=optimizer, metrics=['accuracy'])
        return nn

    es = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy', mode='max', verbose=0, patience=20)
    nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs,
                         batch_size=batch_size, verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(nn, X_train_s, train["dir"], scoring=score_acc, cv=kfold, fit_params={
                            'callbacks': [es]}).mean()
    return score


def hp_bo(nn_bo):
    hp = nn_bo.max['params']
    learning_rate = hp['learning_rate']
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU, 'relu']
    hp['activation'] = activationL[round(hp['activation'])]
    hp['batch_size'] = round(hp['batch_size'])
    hp['epochs'] = round(hp['epochs'])
    hp['layers1'] = round(hp['layers1'])
    hp['layers2'] = round(hp['layers2'])
    hp['neurons'] = round(hp['neurons'])
    optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta',
                  'Adagrad', 'Adamax', 'Nadam', 'Ftrl', 'Adam']
    optimizerD = {'Adam': tf.keras.optimizers.Adam(learning_rate=learning_rate), 'SGD': tf.keras.optimizers.SGD(learning_rate=learning_rate), 'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate), 'Adadelta': tf.keras.optimizers.Adadelta(
        learning_rate=learning_rate), 'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate), 'Adamax': tf.keras.optimizers.Adamax(learning_rate=learning_rate), 'Nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate), 'Ftrl': tf.keras.optimizers.Ftrl(learning_rate=learning_rate)}
    hp[
        'optimizer'] = optimizerD[optimizerL[round(hp['optimizer'])]]
    return hp


def nn_cl_fun(hp):
    nn = tf.keras.models.Sequential()
    nn.add(tf.keras.layers.Input(shape=(X_train_s.shape[1],)))
    if hp['normalization'] > 0.5:
        nn.add(tf.keras.layers.BatchNormalization())
    for i in range(hp['layers1']):
        nn.add(tf.keras.layers.Dense(
            hp['neurons'], activation=hp['activation']))
    if hp['dropout'] > 0.5:
        nn.add(tf.keras.layers.Dropout(hp['dropout_rate'], seed=42))
    for i in range(hp['layers2']):
        nn.add(tf.keras.layers.Dense(
            hp['neurons'], activation=hp['activation']))
    nn.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer=hp[
        'optimizer'], metrics=['accuracy'])
    return nn


# Runnable code


nn_bo = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=42)
nn_bo.maximize(init_points=25, n_iter=4)  # execute maximization process

hps = hp_bo(nn_bo)
nn = nn_cl_fun(hps)

es = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy', mode='max', verbose=0, patience=20)
nn.fit(X_train_s, train["dir"], epochs=hps['epochs'],
       batch_size=hps['batch_size'], verbose=1, callbacks=[es])

binary_classification_prediction(nn=nn, train_s=train_s, train=train, cols=cols, features=features, test_s=test_s,
                                 test=test, use_train_data=False, short_boundary=0.47, long_boundary=0.53)
