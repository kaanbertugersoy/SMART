import tensorflow as tf

# Default parameters for build_model function
LSTM_LAYERS = 2
LSTM_UNITS = 100
DROPOUT = False
DROPOUT_RATE = 0.3
REGULARIZE = False
REG = tf.keras.regularizers.l2(0.001)
OPTIMIZER_CLASS = tf.keras.optimizers.Adam
LEARNING_RATE = 0.001
SHAPE = 0


# Default parameters for train_model function
VALIDATION_SPLIT = 0.2
SHUFFLE = True
CW = None
EPOCHS = 100
VERBOSE = 1

# Default parameters for LSTM model
TIME_STEPS = 60
NUM_FEATURES = 0


def lstm_model(layers=LSTM_LAYERS,
               units=LSTM_UNITS,
               dropout=DROPOUT,
               rate=DROPOUT_RATE,
               regularize=REGULARIZE,
               reg=REG,
               optimizer_class=OPTIMIZER_CLASS,
               learning_rate=LEARNING_RATE,
               shape=SHAPE,
               time_steps=TIME_STEPS,
               num_features=NUM_FEATURES):

    print(f"time_steps: {time_steps}, num_features: {num_features}")
    if not regularize:
        reg = None

    optimizer = optimizer_class(learning_rate=learning_rate)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(time_steps, num_features)))

    # model.add(tf.keras.layers.LSTM(units, activation='tanh',
    #           return_sequences=True, activity_regularizer=reg))

    # if dropout:
    #     model.add(tf.keras.layers.Dropout(rate, seed=42))

    for _ in range(1, layers):
        model.add(tf.keras.layers.LSTM(units, activation='tanh',
                  return_sequences=True, activity_regularizer=reg))
        if dropout:
            model.add(tf.keras.layers.Dropout(rate, seed=42))

    model.add(tf.keras.layers.LSTM(
        units, activation='tanh', activity_regularizer=reg))
    # if dropout:
    #     model.add(tf.keras.layers.Dropout(rate, seed=42))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
