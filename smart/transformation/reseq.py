
import numpy as np

# Handle missing values before resequencing !
# Data normalization is preferred before resequencing !


def reseq(series, T, K, multi_output=False, lstm_reshaping=False):
    """
    RESEQUENCE a time series for T past observations to predict the next K observations.

    !!! CRITICAL: The first dimension of the input series is assumed to be the timeseries that T and K will be used to resequence the data.
    """

    # Assumes the first dimension is the time dimension
    num_samples = series.shape[0] - T - K + 1

    # Determines output shapes based on input shape
    X_shape = (num_samples, T) + series.shape[1:]
    if multi_output:
        y_shape = (num_samples, K) + series.shape[1:]
    else:
        y_shape = (num_samples,) + series.shape[1:]

    # Preallocation of X and y
    X = np.zeros(X_shape)
    y = np.zeros(y_shape)

    for t in range(num_samples):
        X[t] = series[t:t+T]
        if multi_output:
            y[t] = series[t+T:t+T+K]
        else:
            y[t] = series[t+T+K-1]

    if lstm_reshaping:
        X = X.reshape(-1, T, *series.shape[1:])
    else:
        X = X.reshape(-1, T * np.prod(series.shape[1:]))

    return X, y
