import numpy as np


def interquartile_range_threshold(values):
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    return threshold
