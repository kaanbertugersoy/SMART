import numpy as np

# multi-step forecast consist of these steps:
# 1. predict
# 2. update the predictions list
# 3. make the new input until the number of predictions is equal to n_test
# 4. return the predictions


def multistep_forecast_pipeline(n_test, pre_trained_model, X_test):
    # multi-step forecast
    multistep_preds = []

    # first test input
    last_x = X_test[0]

    while len(multistep_preds) < n_test:
       # Reshape last_x to match the expected input shape of the model
        # Add leading 1 for batch dimension
        last_x_reshaped = last_x.reshape(1, *last_x.shape)

        p = pre_trained_model.predict(last_x_reshaped)[0]

        # update the predictions list
        multistep_preds.append(p)

        # make the new input
        last_x = np.roll(last_x, -1)
        last_x[-1] = p

    multistep_preds = np.array(multistep_preds)

    return multistep_preds
