import cvxpy as cp
import pandas as pd


def optimal_portfolio(X,
                      tickers,
                      risk_free_rate=0.02,
                      transaction_costs=0.001,
                      sector_data=None,
                      max_sector_exposure=0.2,
                      max_weight=0.1,
                      max_assets=None):
    # X: DataFrame of asset returns
    # tickers: List of asset tickers
    # sector_data: DataFrame with sector exposure (optional)
    # risk_free_rate: Annual risk-free rate (default is 2%)
    # transaction_costs: Transaction costs per trade (default is 0.1%)
    # max_sector_exposure: Maximum allowed exposure to a single sector (default is 20%)
    # max_weight: Maximum weight of any single asset (default is 10%)
    # max_assets: Maximum number of assets in the portfolio (optional)

    X = pd.DataFrame(X, columns=tickers)

    mu = X.mean(axis=0).values
    Sigma = X.cov().values  # Covariance matrix

    n = len(tickers)
    weights = cp.Variable(n)

    # Risk and return
    risk = cp.quad_form(weights, Sigma)
    ret = mu @ weights

    # Define Sharpe ratio objective
    objective = cp.Maximize((ret - risk_free_rate) -
                            transaction_costs * cp.sum(weights))

    # Define constraints
    constraints = [cp.sum(weights) == 1, weights >= 0]

    # Maximum weight constraint
    constraints.append(weights <= max_weight)

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
