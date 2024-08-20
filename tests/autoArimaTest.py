from smart.statistic.arima.model import auto_arima_forecast
from smart.services.archer.singleTickerRequester import get_single_ticker_data, compute_and_add_log_returns


def autoArimaTest():
    ticker = "NVDA"
    forcest_horizon = 20

    data = get_single_ticker_data(
        ticker=ticker, interval="1d", start_date="2020-01-01")
    data = compute_and_add_log_returns(data)

    auto_arima_forecast(data['CloseLogReturn'], forcest_horizon)
