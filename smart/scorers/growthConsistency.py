import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def exponential_model(x, a, b):
    return a * np.exp(b * x)


def score_growth_consistency(df):
    total_score = 0
    feature_count = 0
    column_scores = {}

    for column in df.columns:
        y = df[column].dropna().values
        x = np.arange(len(y))

        # Fit the exponential model
        try:
            popt, _ = curve_fit(exponential_model, x, y, maxfev=10000)
            y_pred = exponential_model(x, *popt)
            r2 = r2_score(y, y_pred)
            r2 = max(0, r2)  # Avoid negative R²
        except (RuntimeError, TypeError):
            r2 = 0

        # Scale R² to 0-100
        scaled_score = r2 * 100
        column_scores[column] = scaled_score
        total_score += scaled_score
        feature_count += 1

    if feature_count == 0:
        return 0, column_scores  # Avoid division by zero if no features

    average_score = total_score / feature_count

    return average_score, column_scores
