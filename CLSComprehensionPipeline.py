import pandas as pd
import pickle
from smart.importance.comprehension import comprehend
from smart.quarries.singleTickerRequester import get_single_ticker_data
from smart.features.ta import buildFeatures
from smart.utils.split_data import split_data
from smart.utils.data_scaler import scale_data
from smart.utils.interquartile_range import interquartile_range_threshold

import torch
import numpy as np
from smart.importance.comprehension import Dataset

if __name__ == "__main__":
    # Randomly select 100 stocks from the dataset and calculate
    # the importance of each feature for more reliable result
    ticker_list_path = "market_list.csv"
    sample_size = 100

    df = pd.read_csv(ticker_list_path)

    filtered_df = df[(df["assetType"] == "Stock") & (
        df["status"] == "Active") & (df["name"]) & (df["ipoDate"] < "2014-01-01")]
    sample_df = filtered_df.sample(n=sample_size)

    ovrl = {}

    for ticker in sample_df["symbol"]:
        data = get_single_ticker_data(
            ticker=ticker, interval="1d", start_date="1990-01-01")
        if data is None:
            continue  # Skip to the next iteration in case a ticker problem
        df, cols = buildFeatures(df=data)

        train, test = split_data(df, split_ratio=0.75)
        train_s, test_s = scale_data(train, test)

        x = train_s[cols]
        y = train["dir"]

        train_loader = torch.utils.data.DataLoader(
            Dataset(x.values, y.values), shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            Dataset(x.values, y.values), shuffle=False)

        x = test_s[cols]
        importance_values, feat_imp, model = comprehend(x, cols)

        values = np.array([val[0] for val in importance_values])
        threshold = interquartile_range_threshold(values)
        top_features = [(val, ind)
                        for val, ind in importance_values if val > threshold]
        print(f"Top features: {top_features}")

        for value, key in top_features:
            if key in ovrl:
                ovrl[key] += value
            else:
                ovrl[key] = value

    ovrl_list = [(a, b) for a, b in sorted(ovrl.items(), key=lambda x: x[1])]

    with open("feat_imp_scores1.pkl", "wb") as file:
        pickle.dump(ovrl_list, file)
