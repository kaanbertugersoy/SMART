import numpy as np

# !!! Out of use !!!

# First developed sequencing function for "specifically" 1D time series data.

# But now we developed a more general version of the function that can handle any-dimensional data.
# For that, go to the file: smart/transformation/reseq.py


def t_past_step_k_future_output_resequence_1d(series, T, K, multi_output=False, lstm_reshaping=False):
    """
    Resequence a 1-D time series for supervised learning.

    Parameters:
    - series: 1D time series data.
    - T: Number of past time steps to include in the input.
    - K: Number of future time steps to predict.
    - multi_output: If True, predicts multiple future time steps; otherwise, predicts a single future time step.
    - lstm_reshaping: If True, reshapes for LSTM input (3D); otherwise, reshapes for general ML input (2D).

    Example:

    series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    T = 3, K = 2 

    X, y = t_past_step_k_future_output_reshape_1d(series, T, K, True, True)

    X shape: (8, 3, 1)
    >>> [ [[1][2][3]] [[2][3][4]] [[3][4][5]] [[4][5][6]] [[5][6][7]] [[6][7][8]] [[7][8][9]] [[8][9][10]] ]

    y shape: (8, 2)
    >>> [ [4 5] [5 6] [6 7] [7 8] [8 9] [9 10] [10 11] [11 12] ]
    """

    X, y = [], []

    for t in range(len(series) - T - K + 1):
        # select past steps
        past_steps = series[t: t + T]
        X.append(past_steps)

        # select future steps
        if multi_output:
            future_steps = series[t + T:t + T + K]
        else:
            future_steps = series[t + T + K - 1]
        y.append(future_steps)

    # reshape
    if lstm_reshaping:
        X = np.array(X).reshape(-1, T, 1)
    else:
        X = np.array(X).reshape(-1, T)
    if multi_output:
        y = np.array(y).reshape(-1, K)
    else:
        y = np.array(y)

    print("X.shape", X.shape, "y.shape", y.shape)

    return X, y
