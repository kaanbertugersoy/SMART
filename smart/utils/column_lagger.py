import pandas as pd


def lag_columns(df, lag_count, cols):
    lagged_cols = []

    for col in cols:
        lagged_cols = [df[col].shift(
            lag) for lag in range(1, lag_count + 1)]
        lagged_df = pd.concat(lagged_cols, axis=1)
        lagged_df.cols = [
            f"{col}_lag_{lag}" for lag in range(1, lag_count + 1)]
        lagged_cols.extend(lagged_df.cols)
        df = pd.concat([df, lagged_df], axis=1)

    df.dropna(inplace=True)
    return df, lagged_cols
