import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from smart.statistic.arima.helpers import plot_result, plot_test
from smart.utils.split_data import split_data

import pmdarima as pm


def arima_forecast(timeseries, order, forcast_horizon=10):
    """
    Fits an ARIMA model to the timeseries data and plots the forecast.

    Parameters:
    timeseries: The time series data.
    order: The (p, d, q) order of the ARIMA model.
    forcast_horizon: Number of steps to forecast into the future.

    Returns:
    None
    """
    if not isinstance(timeseries, pd.Series):
        raise ValueError("The timeseries data must be a pandas Series.")

    model = ARIMA(timeseries, order=order)
    fitted_model = model.fit()

    forecast = fitted_model.get_forecast(steps=forcast_horizon)
    forecast_index = pd.date_range(
        start=timeseries.index[-1], periods=forcast_horizon + 1, freq='MS')[1:]
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)

    conf_int = forecast.conf_int(alpha=0.05)
    lower_series = pd.Series(conf_int.iloc[:, 0], index=forecast_index)
    upper_series = pd.Series(conf_int.iloc[:, 1], index=forecast_index)

    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, label='Original')
    plt.plot(forecast_series, label='Forecast', color='red')
    plt.fill_between(forecast_series.index,
                     lower_series,
                     upper_series, color='pink', alpha=0.3)
    plt.legend()
    plt.title(f'ARIMA{order} Forecast')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(timeseries, ax=ax[0])
    plot_pacf(timeseries, ax=ax[1])
    plt.show()


def auto_arima_forecast(timeseries, forecast_horizon):

    if not isinstance(timeseries, pd.Series):
        raise ValueError("The timeseries data must be a pandas Series.")

    train = timeseries.iloc[:-forecast_horizon]
    test = timeseries.iloc[-forecast_horizon:]

    model = pm.auto_arima(train, error_action='ignore', stepwise=False,
                          trace=True, suppress_warnings=True, seasonal=False)
    model.summary()
    model.get_params()

    plot_result(model, timeseries, train, test)
    plot_test(model, test)
