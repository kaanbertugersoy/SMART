import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime, timedelta

from smart.convex.optimizer import optimal_portfolio
from smart.quarries.singleTickerRequester import get_single_ticker_data

# Define tickers and date range
tickers = ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL',
           'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL',
           'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG',
           'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'AOS', 'APA',
           'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP',
           'AZO', 'AVB', 'AVY', 'BKR', 'BAC', 'BK', 'BAX', 'BDX', 'BBY', 'BIO', 'BIIB',
           'BLK', 'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'CHRW', 'CDNS',
           'CZR', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE',
           'CDW', 'CE', 'CNC', 'CNP', 'CF', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI',
           'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL',
           'CMCSA', 'CMA', 'CAG', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST',
           'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY',
           'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE',
           'DUK', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ENPH',
           'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES', 'EXC', 'EXPE',
           'EXPD', 'EXR', 'XOM', 'FFIV', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FISV', 'FMC',
           'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN', 'IT', 'GNRC', 'GD',
           'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS',
           'HCA', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM',
           'HPQ', 'HUM', 'HBAN', 'HII', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE',
           'IBM', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J',
           'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI',
           'KLAC', 'KHC', 'KR', 'LB', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS', 'LEN',
           'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO',
           'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK',
           'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MPWR',
           'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM',
           'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NOV', 'NRG',
           'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OTIS',
           'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PENN', 'PNR', 'PEP', 'PRGO', 'PFE',
           'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU',
           'PTC', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX',
           'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL',
           'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SNA',
           'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW',
           'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX',
           'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA',
           'UA', 'UNP', 'UAL', 'UNH', 'UPS', 'URI', 'UHS', 'UNM', 'VLO', 'VTR', 'VRSN', 'VRSK',
           'VZ', 'VRTX', 'VFC', 'VTRS', 'V', 'VNT', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA',
           'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU', 'WRK', 'WY', 'WHR',
           'WMB', 'WYNN', 'XEL', 'XRX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS'
           ]


def plot_portfolio_weights(weights_series):
    plt.figure(figsize=(10, 6))
    weights_series.plot(kind='bar', color='skyblue')
    plt.title('Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.show()


if __name__ == "__main__":
    successful_tickers = []
    returns_dict = {}
    cutoff_date = datetime.now().date() - timedelta(days=365)
    for ticker in tqdm(tickers, desc='Processing tickers', unit='ticker'):
        try:
            data = get_single_ticker_data(
                ticker=ticker, interval="1d", start_date="1990-01-01")

            last_data_date = data["returns"].dropna().index[-1].date()
            if last_data_date < cutoff_date:
                print(
                    f"{ticker} data is not up to date (last available date: {last_data_date}) and will be skipped.")
                continue

            first_data_date = data["returns"].dropna().index[0].date()
            if cutoff_date - first_data_date < timedelta(days=730):
                print(
                    f"{ticker} data length is not sufficient for this operation and will be skipped.")
                continue

            returns_dict[ticker] = data["returns"]
            successful_tickers.append(ticker)
        except:
            print(f"{ticker} is skipped.")
            continue

    tickers = successful_tickers

    # Combine the returns data into DataFrame
    returns_df = pd.concat(returns_dict.values(), axis=1,
                           keys=returns_dict.keys())
    returns_df.dropna(inplace=True)
    returns_list = returns_df.values.tolist()
    weights = optimal_portfolio(returns_list, tickers)
    weights_series = pd.Series(weights, index=tickers)

    print(returns_df.shape)
    print(weights_series.nlargest(10))
    plot_portfolio_weights(weights_series)
