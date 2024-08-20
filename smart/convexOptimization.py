import cvxpy as cp
import pandas as pd

# Optimal portfolio allocation using convex optimization which maximizes the Sharpe ratio


def optimal_portfolio(X,
                      tickers,
                      risk_aversion=0.01,
                      max_weight=0.5,
                      max_assets=None,
                      max_sector_exposure=0.2,
                      sector_data=None
                      ):

    X = pd.DataFrame(X, columns=tickers)

    mu = X.mean(axis=0).values
    Sigma = X.cov().values  # Covariance matrix

    n = len(tickers)
    weights = cp.Variable(n)

    # Risk and return
    risk = cp.quad_form(weights, Sigma)
    ret = mu.T @ weights

    # Define Sharpe ratio objective
    # Sortino ratio (learn more: https://www.investopedia.com/terms/s/sortinoratio.asp)
    objective = cp.Maximize((ret - risk * risk_aversion))

    # Define constraints
    constraints = [cp.sum(weights) == 1, weights >= 0, weights <= max_weight]

    # Cardinality constraint
    if max_assets is not None:
        k = cp.Variable(boolean=True, shape=n)
        constraints.append(cp.sum(k) <= max_assets)
        constraints.append(weights <= k * max_weight)

    # Sector constraints
    if sector_data is not None:
        for sector in sector_data.columns:
            sector_exposure = sector_data[sector].values @ weights
            constraints.append(sector_exposure <= max_sector_exposure)

    # Define the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    optimal_weights = weights.value
    return optimal_weights
