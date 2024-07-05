import numpy as np


def cw(df):  # to remove class imbalance, upward and downward movements can harm our model if we not assign weights
    # specify the column name of the target variable
    c0, c1 = np.bincount(df["dir"])
    w0 = (1/c0) * (len(df)) / 2.0
    w1 = (1/c1) * (len(df)) / 2.0
    return {0: w0, 1: w1}
