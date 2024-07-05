import tensorflow as tf

# Default parameters for build_model function
HIDDEN_LAYERS = 2
HIDDEN_UNITS = 100
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


def model(hl=HIDDEN_LAYERS,
          hu=HIDDEN_UNITS,
          dropout=DROPOUT,
          rate=DROPOUT_RATE,
          regularize=REGULARIZE,
          reg=REG,
          optimizer_class=OPTIMIZER_CLASS,
          learning_rate=LEARNING_RATE,
          shape=SHAPE):
    if not regularize:
        reg = None

    optimizer = optimizer_class(learning_rate=learning_rate)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=shape))

    model.add(tf.keras.layers.Dense(
        hu, activation="softsign", activity_regularizer=reg))
    # model.add(tf.keras.layers.BatchNormalization())
    # BatchNormalization after the first hidden layer adds huge performance

    if dropout:
        model.add(tf.keras.layers.Dropout(rate, seed=42))

    for _ in range(hl - 1):
        model.add(tf.keras.layers.Dense(
            hu, activation="softsign", activity_regularizer=reg))
        # model.add(tf.keras.layers.BatchNormalization())
        # BacthNormalization after each hidden layer lowers the performance
        # if dropout:
        #     model.add(tf.keras.layers.Dropout(rate, seed=42))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])

    return model


# Deprecated function instead use train_nn_model from training.py
def train_model(nn,
                train_s,
                cols,
                train,
                epochs=EPOCHS,
                verbose=VERBOSE,
                validation_split=VALIDATION_SPLIT,
                shuffle=SHUFFLE,
                cw=CW):

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    rl = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     'NNModelAlphaOne.keras', save_best_only=True, monitor='val_loss', mode='min')

    nn.fit(x=train_s[cols], y=train["dir"], epochs=epochs, verbose=verbose, validation_split=validation_split,
           shuffle=shuffle, class_weight=cw, callbacks=[es, rl])
