
import numpy as np

# Random walk steps totally random on a time series,
# the purpose is to create a baseline to compare the model's performance
# the model should at least beat the random walk to be considered as usable


def random_walk_forecast_from_series(time_series):

    # Calculate the differences between consecutive points
    diffs = time_series.diff().dropna()

    # Calculate mean and standard deviation of the differences
    mean = diffs.mean()
    std_dev = diffs.std()

    # Calculate random steps using the mean and standard deviation
    random_steps = np.random.normal(
        loc=mean, scale=std_dev, size=len(time_series))

    # Create the forecast by adding random steps to the shifted time series
    forecast = time_series.shift(1) + random_steps

    # Handle the first NaN value by filling it with the original value
    forecast.iloc[0] = time_series.iloc[0]

    return forecast
