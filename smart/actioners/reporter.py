from ib_insync import *
import pandas as pd
import os


def report(ib, df, session_start):
    fill_df = pd.DataFrame([{'execId': fs.execution.execId, 'time': fs.execution.time, 'side': fs.execution.side,
                           'cumQty': fs.execution.cumQty, 'avgPrice': fs.execution.avgPrice} for fs in ib.fills()])
    profit_df = pd.DataFrame([{'execId': fs.commissionReport.execId,
                             'realizedPNL': fs.commissionReport.realizedPNL} for fs in ib.fills()])

    report = pd.merge(fill_df, profit_df, on='execId')
    report = report.set_index('time').loc[session_start:]
    report = report.groupby('time').agg(
        {'side': 'first', 'cumQty': 'max', 'avgPrice': 'mean', 'realizedPNL': 'sum'})
    report['cumPNL'] = report.realizedPNL.cumsum()

    os.system('cls')
    print(df, report)
