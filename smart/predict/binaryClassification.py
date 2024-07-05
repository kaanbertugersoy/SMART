import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

LONG_BOUNDARY = 0.53
SHORT_BOUNDARY = 0.47
USE_DYNAMIC_BOUNDARIES = True
PLOT_PERFORMANCE_GRAPH = True


def binary_classification_prediction(nn,
                                     x,
                                     y,
                                     data,
                                     long_boundary=LONG_BOUNDARY,
                                     short_boundary=SHORT_BOUNDARY,
                                     use_dynamic_boundaries=USE_DYNAMIC_BOUNDARIES,
                                     plot_performance_graph=PLOT_PERFORMANCE_GRAPH):

    nn.evaluate(x=x, y=y)  # Returns accuracy and loss.
    pred = nn.predict(x)

    data["proba"] = pred

    if use_dynamic_boundaries:
        short_boundary, long_boundary = dynamic_boundary_selection(
            y_true=y.values, y_pred=data["proba"])

    df = calculate_outcome(
        data, short_boundary, long_boundary)

    # print(f"Last position: {df.position.iloc[-1]} on {df.index[-1]}")
    print(df.tail(10)[["proba", "position",
          "returns", "strategy", "strategy_net"]])

    # Plot the performance of the strategy
    if plot_performance_graph:
        plot_performance(df.creturns, df.cstrategy, df.cstrategy_net)


def plot_performance(creturns, cstrategy, cstrategy_net):
    title = f"Buy and Hold: {round(creturns.iloc[-1], 6)} | GOAT's Return: {round(cstrategy_net.iloc[-1], 6)} | Outperf: {round(cstrategy_net.iloc[-1] - creturns.iloc[-1], 6)}"
    creturns.plot(figsize=(12, 8), title=title)
    cstrategy.plot(figsize=(12, 8), title=title)
    cstrategy_net.plot(figsize=(12, 8), title=title)
    plt.show()


def dynamic_boundary_selection(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    scores = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]

    # most critic part is the margin calculation (NOT 100% OPTIMAL YET)
    confidence = (tpr[optimal_idx] + (1 - fpr[optimal_idx])) / 2
    margin = (1 - confidence) / 2

    lower_boundary = max(0.0, optimal_threshold - margin)
    upper_boundary = min(1.0, optimal_threshold + margin)

    print(f"Optimal Threshold: {optimal_threshold} | Confidence: {confidence}")
    print(
        f"Selected Lower Bound: {lower_boundary} | Selected Upper Bound: {upper_boundary}")
    return lower_boundary, upper_boundary


def calculate_outcome(df, short_boundary, long_boundary):
    df["position"] = np.where(df.proba <= short_boundary, -1, np.nan)
    df["position"] = np.where(df.proba >= long_boundary, 1, df.position)
    df["position"] = np.where((df.proba > short_boundary) & (
        df.proba < long_boundary), 0, df.position)
    df["position"] = df.position.ffill()

    # IMPORTANT: shift by 1 for the next day's return
    df["strategy"] = df.position.shift(1) * df.returns
    # df["strategy"] = df.position * df.returns
    df = df.drop(df.index[0])

    df["creturns"] = df["returns"].cumsum().apply(np.exp)
    df["cstrategy"] = df["strategy"].cumsum().apply(np.exp)

    ptc = 0.000059  # transaction cost
    df["trades"] = df.position.diff().abs()
    df["strategy_net"] = df.strategy - df.trades * ptc

    df["cstrategy_net"] = df["strategy_net"].cumsum().apply(np.exp)

    return df
