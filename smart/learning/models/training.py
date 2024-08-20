import tensorflow as tf

# Training nearly same for all models so,
# train function is defined globally for all models

# Default params =>
VALIDATION_SPLIT = 0.2
SHUFFLE = True
CW = None
EPOCHS = 100
VERBOSE = 1
SAVE_CHECKPOINT = False
CHECKPOINT_NAME = 'null'
MIN_LR = 1e-6
MONITOR = 'val_loss'
BATCH_SIZE = 32


def train_nn_model(nn,
                   x,
                   y,
                   epochs=EPOCHS,
                   verbose=VERBOSE,
                   validation_split=VALIDATION_SPLIT,
                   shuffle=SHUFFLE,
                   cw=CW,
                   save_checkpoint=SAVE_CHECKPOINT,
                   checkpoint_name=CHECKPOINT_NAME,
                   min_lr=MIN_LR,
                   monitor=MONITOR,
                   batch_size=BATCH_SIZE,
                   callbacks=None):

    es = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, mode='auto', patience=15, restore_best_weights=True)
    rl = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, mode='auto', factor=0.3, patience=10, min_lr=min_lr)
    if save_checkpoint:
        cp = tf.keras.callbacks.ModelCheckpoint(
            f"{checkpoint_name}.keras", save_best_only=True, monitor=monitor, mode='min')

    callbacks = [es, rl, cp] if save_checkpoint else [es, rl]
    if callbacks is not None:
        callbacks += callbacks

    nn.fit(x=x, y=y, epochs=epochs, verbose=verbose, validation_split=validation_split,
           shuffle=shuffle, class_weight=cw, callbacks=callbacks, batch_size=batch_size)
