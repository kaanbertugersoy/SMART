import os
import json
import pandas as pd
import numpy as np
from alpha_vantage.fundamentaldata import FundamentalData
import matplotlib.pyplot as plt

api_key = 'asd'

# For trading algorithms, financial data is a drowback. But for market analysis for long term investing it is inevitable.


def extract_financial_data(api_key, ticker):
    file_path = "smart\store\\financials"
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Correctly form the full file paths
    financial_file = os.path.join(file_path, f"{ticker}.csv")

    # Check if the data is already stored in files
    if not os.path.exists(financial_file):
        df = fetch_alpha_vantage_data(api_key, ticker)
        df.to_csv(financial_file)

    # Read the csv file again, even if it was just downloaded
    df = pd.read_csv(financial_file, index_col='Date', parse_dates=True)

    return df


def fetch_alpha_vantage_data(api_key, ticker):
    fd = FundamentalData(key=api_key, output_format='pandas')

    # Fetch the balance sheet
    balance_sheet, _ = fd.get_balance_sheet_quarterly(symbol=ticker)

    # Fetch the income statement
    income_statement, _ = fd.get_income_statement_quarterly(symbol=ticker)

    # Fetch the cash flow statement
    cash_flow, _ = fd.get_cash_flow_quarterly(symbol=ticker)

    # Clean the data
    bs = clean_financial_data(balance_sheet)
    ist = clean_financial_data(income_statement)
    cf = clean_financial_data(cash_flow)

    # Combine the cleaned DataFrames on their Date index
    df = pd.concat([bs, ist, cf], axis=1, join='inner')

    return df


def clean_financial_data(df):
    df.columns = df.columns.str.strip()  # Strip any whitespace from column names
    df.rename(columns={"fiscalDateEnding": "Date"}, inplace=True)
    df.index = pd.to_datetime(df.Date)
    df.drop(columns=["Date"], inplace=True)
    df_cleaned = df.drop(columns=df.columns[df.isin(['None']).any()])
    df_cleaned.sort_index(inplace=True)
    df_cleaned.dropna(axis=1, inplace=True)
    df_cleaned.drop(columns=["reportedCurrency"], inplace=True)
    return df_cleaned


def build_nn_params(financial_data, scale=False):
    selected_growth_params = [
        'totalRevenue',
        'grossProfit',
        'operatingIncome',
        'netIncome',
        'totalAssets',
        'totalLiabilities',
        'totalShareholderEquity',
        'operatingCashflow',
        'capitalExpenditures',
    ]

    selected_profitability_params = [
        'ebitda',
    ]

    selected_efficiency_params = [
        'operatingExpenses',
        'costofGoodsAndServicesSold'
        'changeInInventory',
        'changeInReceivables',
        'changeInOperatingAssets',
    ]

    selected_liquidity_params = [
        'cashflowFromInvestment',
        'cashflowFromFinancing',
    ]

    for param in selected_growth_params:
        if param in financial_data.columns:
            if financial_data[param].min() < 0:
                financial_data[param] = financial_data[param].apply(
                    lambda x: x + abs(financial_data[param].min()) + 1)
            financial_data[param] = np.log(
                financial_data[param] / financial_data[param].shift(1))
            financial_data[param] = financial_data[param].cumsum().apply(
                np.exp)

    filtered_data = financial_data[[
        param for param in selected_growth_params if param in financial_data.columns]].copy()

    if scale:
        filtered_data = scale_columns(
            filtered_data, selected_growth_params, method="standardization")
    filtered_data.dropna(inplace=True)

    return filtered_data


def scale_columns(df, cols, method="normalization"):
    for col in cols:
        if method == "standardization":
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif method == "normalization":
            df[col] = (df[col] - df[col].min()) / \
                (df[col].max() - df[col].min())
        else:
            raise ValueError(
                "Method must be either 'standardization' or 'normalization'")
    return df


def add_financial_params(dataframe, cols, ticker):
    financial_data = extract_financial_data(api_key, ticker)
    nn_params = build_nn_params(financial_data)
    nn_params.index = nn_params.index.tz_localize(None)

    resampled_data = nn_params.resample('D').ffill()
    merged_data = pd.concat([dataframe, resampled_data], axis=1, join='inner')
    merged_data.ffill(inplace=True)
    merged_data.dropna(inplace=True)
    updated_columns = cols + nn_params.columns.tolist()
    return merged_data, updated_columns


def quarterly_financials(ticker):
    financial_data = extract_financial_data(api_key, ticker)

    nn_params = build_nn_params(financial_data)
    nn_params.index = nn_params.index.tz_localize(None)

    return nn_params, nn_params.columns.tolist()
