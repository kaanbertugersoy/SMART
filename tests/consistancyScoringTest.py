import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from smart.services.archer.financialDataProvider import quarterly_financials
from smart.scorers.growthConsistency import score_growth_consistency

TICKER = "NVDA"


def bar(keys, values, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.bar(keys, values, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def consistancyScoringTest():

    df, cols = quarterly_financials(ticker=TICKER)

    # Draw line plot of df's columns
    df.plot(kind='line', figsize=(10, 6))
    plt.xlabel('Quarter')
    plt.ylabel('Value')
    plt.title('Quarterly Financials')
    plt.legend(cols)
    plt.show()

    avg_score, col_scores = score_growth_consistency(df)

    print(f"Column scores: {col_scores}")
    print(f"Growth consistency score: {avg_score}")

    bar(keys=col_scores.keys(), values=col_scores.values(),
        xlabel='Growth Features', ylabel='Score', title='Consistency Score for Each Growth Feature')
