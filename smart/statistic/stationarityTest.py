
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller test for stationarity

# Stationarity: A time series is stationary if the joint distribution of the data
# does not change when shifted in time. This implies that the mean, variance, and
# covariance structure of the data is constant over time.


def adf_test(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("Stationary")
    else:
        print("Non-Stationary")
