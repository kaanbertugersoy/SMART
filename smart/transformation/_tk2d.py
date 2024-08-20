import numpy as np

# !!! Out of use !!!

# First developed sequencing functions for "specifically" 2D time series data.

# But now we developed a more general version of the function that can handle any-dimensional data.
# For that, go to the file: smart/transformation/reseq.py


def t_past_step_k_future_output_resequence_2d(series, T, K, multi_output=False, lstm_reshaping=False):

    X, y = [], []

    for t in range(len(series) - T - K + 1):
        # Select past steps
        past_steps = series[t: t + T, :]
        X.append(past_steps)

        # Select future steps
        target_series = series[:, -1]
        if multi_output:
            future_steps = target_series[t + T:t + T + K, :]
        else:
            future_steps = target_series[t + T + K - 1]
        y.append(future_steps)

    # Reshape
    if lstm_reshaping:
        X = np.array(X).reshape(-1, T, series.shape[1])
    else:
        X = np.array(X).reshape(-1, T * series.shape[1])

    if multi_output:
        y = np.array(y).reshape(-1, K)
    else:
        y = np.array(y).reshape(-1)

    print("X.shape", X.shape, "y.shape", y.shape)

    return X, y
