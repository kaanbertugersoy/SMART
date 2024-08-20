
from smart.transformation.reseq import reseq
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore

# multi-output forecast consist of these steps:
# 1. reshape the data
# 2. train-test split
# 3. fit the model
# 4. predict
# 5. return the predictions and the model


def multioutput_forecast_pipeline(T, n_test, series, model, is_lstm=False):

    Tx = T
    Ky = n_test

    X, Y = reseq(series, Tx, Ky, multi_output=True, lstm_reshaping=is_lstm)

    X_train, Y_train = X[:-1], Y[:-1]
    X_test, Y_test = X[-1:], Y[-1:]

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

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_pred = train_pred[:, 0]
    test_pred = test_pred[0]

    return test_pred, r
