
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from smart.transformation.reseq import reseq


# one-step ahead forecast consist of these steps:
# 1. reshape the data
# 2. train-test split
# 3. fit the model
# 4. predict
# 5. return the predictions and the model


def one_step_ahead_forecast_pipeline(series, T, n_test, model, is_lstm=False):

    # Shape the data
    X, Y = reseq(series, T=T, K=1, multi_output=False, lstm_reshaping=is_lstm)

    # Train-test split
    X_train, Y_train = X[:-n_test], Y[:-n_test]
    X_test, Y_test = X[-n_test:], Y[-n_test:]

    es = EarlyStopping(monitor='val_loss', mode='auto',
                       patience=15, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', mode='auto',
                           factor=0.3, patience=10, min_lr=1e-6)

    r = model.fit(
        X_train,
        Y_train,
        epochs=50,
        validation_data=(X_test, Y_test),
        callbacks=[es, rl]
    )

    train_pred = model.predict(X_train).flatten()
    test_pred = model.predict(X_test).flatten()

    return train_pred, test_pred, r, model, X_test, Y_test
