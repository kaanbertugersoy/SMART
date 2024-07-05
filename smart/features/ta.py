
import pandas as pd
import numpy as np
import pickle

from ta import add_all_ta_features


def buildFeatures(df):
    original_columns = set(df.columns)

    df = add_all_ta_features(df, open="open", high="high",
                             low="low", close="close", volume="volume")
    df.drop(columns=["trend_psar_up", "trend_psar_down"],
            inplace=True)  # they put nan values in the columns

    df.dropna(inplace=True)

    new_columns = set(df.columns) - original_columns
    all_features = list(new_columns)
    return df, all_features


def buildImportantFeatures(df):
    with open("feat_imp_scores1.pkl", "rb") as f:
        feat_imp_scores = pickle.load(f)

    df, all_features = buildFeatures(df)

    sorted_feat_imp_scores = sorted(
        feat_imp_scores, key=lambda item: item[1], reverse=True)

    features = [item[0] for item in sorted_feat_imp_scores]

    return df, features
