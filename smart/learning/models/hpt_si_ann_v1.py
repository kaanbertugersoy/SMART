import tensorflow as tf

# HyperNet => Hyperparameter Tuning Neural Network
# Currently most dynamic neural network model build function,
# all parameters used in the structure of the model are passed as arguments and tunable

# Created for the purpose of hyperparameter tuning but now extended to be used in other parts of the project


def model_build_fn(layers, neurons, activations, batch_norms, dropouts, dropout_rate, optimizer, shape):

    nn = tf.keras.models.Sequential()

    for i in range(layers):
        if i == 0:
            nn.add(tf.keras.layers.Input(shape=shape))
        else:
            nn.add(tf.keras.layers.Dense(
                round(neurons[i-1]), activation=activations[i-1]))
        # Batch Normalization BEFORE dropout is more common
        if batch_norms[i] > 0.5:
            nn.add(tf.keras.layers.BatchNormalization())
        if dropouts[i] > 0.5:
            nn.add(tf.keras.layers.Dropout(dropout_rate))

    nn.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy',
               optimizer=optimizer, metrics=['accuracy'])
    return nn
