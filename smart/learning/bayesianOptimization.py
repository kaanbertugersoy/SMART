# Import packages
import tensorflow as tf

from bayes_opt import BayesianOptimization
from scikeras.wrappers import KerasClassifier

import smart.learning.models.hpt_si_ann_v1 as Model

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
score_acc = make_scorer(accuracy_score)

# Bayesian Optimization is a probabilistic model-based optimization approach
# which is used to find the maximum value of a function in a number of iterations.
# The function to be optimized is typically the accuracy of a machine learning model.
# The hyperparameters of the model are the input variables of the function. The output
# of the function is the accuracy of the model. The goal is to find the hyperparameters
# that maximize the accuracy of the model.


pbounds = {
    'optimizer': (0, 7),
    'learning_rate': (0.0001, 0.01),
    'batch_size': (0, 6),
    'layers': (1, 4),
    'nrn1': (10, 200),
    'act1': (0, 8),
    'bn1': (0, 1),
    'drp1': (0, 1),
    'nrn2': (10, 200),
    'act2': (0, 8),
    'bn2': (0, 1),
    'drp2': (0, 1),
    'nrn3': (10, 200),
    'act3': (0, 8),
    'bn3': (0, 1),
    'drp3': (0, 1),
    'bn4': (0, 1),
    'drp4': (0, 1),
    'dropout_rate': (0, 0.3)
}

optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta',
              'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
activationL = ['relu', 'sigmoid', 'softplus', 'softsign',
               'tanh', 'selu', 'elu', 'exponential', 'relu']
# batch sizes are restricted to these values for performance reasons, powers of 2, which can be advantageous for computational efficiency on many hardware architectures.

batchL = [16, 32, 64, 128, 256, 512, 1024]


def optimizerF(learning_rate):
    return {'Adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
            'SGD': tf.keras.optimizers.SGD(learning_rate=learning_rate),
            'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            'Adadelta': tf.keras.optimizers.Adadelta(learning_rate=learning_rate),
            'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
            'Adamax': tf.keras.optimizers.Adamax(learning_rate=learning_rate),
            'Nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate),
            'Ftrl': tf.keras.optimizers.Ftrl(learning_rate=learning_rate)}


def maximization_fn(optimizer, learning_rate, batch_size, layers, nrn1, act1, bn1, drp1,
                    nrn2, act2, bn2, drp2, nrn3, act3, bn3, drp3, bn4, drp4, dropout_rate):

    optimizerD = optimizerF(learning_rate)

    optimizer = optimizerD[optimizerL[round(optimizer)]]
    batch_size = batchL[round(batch_size)]
    epochs = 100  # epochs are fixed to 100 because of the early stopping callback
    layers = round(layers)

    neurons = [round(nrn1), round(nrn2), round(nrn3)]
    activations = [activationL[round(act1)], activationL[round(
        act2)], activationL[round(act3)]]
    batch_norms = [bn1, bn2, bn3, bn4]
    dropouts = [drp1, drp2, drp3, drp4]

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='auto', patience=15, restore_best_weights=True)
    lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', mode='auto', factor=0.3, patience=10, min_lr=1e-6)

    nn = KerasClassifier(build_fn=Model.model_build_fn(layers,
                                                       neurons,
                                                       activations,
                                                       batch_norms,
                                                       dropouts,
                                                       dropout_rate,
                                                       optimizer,
                                                       shape=(X.shape[1],)),
                         epochs=epochs,
                         batch_size=batch_size,
                         verbose=0,
                         validation_split=0.2,
                         shuffle=True,
                         class_weight=CW,
                         callbacks=[es, lr])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(nn, X, Y, scoring='accuracy', cv=kfold).mean()
    return score


def hyperparams(nn_bo):
    hp = nn_bo.max['params']
    learning_rate = hp['learning_rate']

    optimizerD = optimizerF(learning_rate)

    layers = round(hp['layers'])
    neurons = [round(hp['nrn1']), round(hp['nrn2']), round(hp['nrn3'])]
    activations = [
        activationL[round(hp['act1'])], activationL[round(hp['act2'])], activationL[round(hp['act3'])]]
    batch_norms = [hp["bn1"], hp["bn2"], hp["bn3"], hp["bn4"]]
    dropouts = [hp["drp1"], hp["drp2"], hp["drp3"], hp["drp4"]]
    dropout_rate = hp['dropout_rate']
    optimizer = optimizerD[optimizerL[round(hp['optimizer'])]]
    batch_size = batchL[round(hp['batch_size'])]
    epochs = 100

    return layers, neurons, activations, batch_norms, dropouts, dropout_rate, optimizer, batch_size, epochs


def maximize_bayesian_optimization(x, y, cw, iterations):

    global X, Y, CW
    X = x
    Y = y
    CW = cw

    # Initialize Bayesian Optimization
    bayesOpt = BayesianOptimization(maximization_fn, pbounds, random_state=42)

    init_points = len(pbounds.keys()) * 2

    # execute maximization process
    bayesOpt.maximize(init_points=init_points,
                      n_iter=max(iterations - init_points, 1))

    return hyperparams(bayesOpt)
