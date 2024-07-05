from ib_insync import *
import pandas as pd
import numpy as np
import datetime as dt
# from IPython.display import display, clear_output
# util.startLoop()
import os

from smart.actioners.reporter import report
from smart.actioners.executer import execute_market_order

# Configuration parameters
sma_s = 2
sma_l = 5
freq = "1 min"
units = 1000
end_time = dt.time(17, 20, 0)  # stop trading at this time
trade_start_time = dt.time(17, 15, 0)  # start trading in the last 5 minutes

# Initialize IB connection
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define contracts
contract = Forex("EURUSD")  # for data streaming
ib.qualifyContracts(contract)

cfd = CFD("EUR", currency="USD")  # for trading
ib.qualifyContracts(cfd)
conID = cfd.conId


def onBarUpdate(bars, hasNewBar):
    global df
    if hasNewBar:
        # Data processing
        df = pd.DataFrame(bars)[["date", "open", "high", "low", "close"]]
        df.set_index("date", inplace=True)

        # Trading Strategy
        df["sma_s"] = df["close"].rolling(sma_s).mean()
        df["sma_l"] = df["close"].rolling(sma_l).mean()
        df.dropna(inplace=True)
        df["position"] = np.where(df["sma_s"] > df["sma_l"], 1, -1)

        # Trading (only in the last 5 minutes)
        current_time = dt.datetime.utcnow().time()
        if trade_start_time <= current_time < end_time:
            target = df["position"].iloc[-1] * units
            execute_market_order(ib, target=target, cfd=cfd, contractId=conID)

        # Display
        os.system("cls")
        print(df)
    else:
        try:
            report(ib, df, session_start)
        except Exception as e:
            print(f"Error reporting: {e}")


if __name__ == "__main__":
    try:
        session_start = pd.to_datetime(dt.datetime.utcnow()).tz_localize("utc")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1 D",  # must be sufficiently long
            barSizeSetting=freq,
            whatToShow="MIDPOINT",
            useRTH=True,
            formatDate=2,
            keepUpToDate=True)
        last_bar = bars[-1].date
        bars.updateEvent += onBarUpdate

        # Stop trading session
        while True:
            ib.sleep(5)  # check every 5 seconds
            if dt.datetime.utcnow().time() >= end_time:
                ib.cancelHistoricalData(bars)
                ib.sleep(10)
                try:
                    report(ib, df, session_start)
                except Exception as e:
                    print(f"Error reporting: {e}")
                print("Session Stopped.")
                ib.disconnect()
                break
            else:
                pass
    except Exception as e:
        print(f"Error: {e}")
        ib.disconnect()
