import pandas as pd


def lag_columns(df, lag_count, cols):
    lagged_cols = []

    for col in cols:
        lagged_col_list = [df[col].shift(
            lag) for lag in range(1, lag_count + 1)]
        lagged_df = pd.concat(lagged_col_list, axis=1)
        lagged_df.columns = [
            f"{col}_lag_{lag}" for lag in range(1, lag_count + 1)]
        lagged_cols.extend(lagged_df.columns)
        df = pd.concat([df, lagged_df], axis=1)

    df.dropna(inplace=True)
    return df, lagged_cols
