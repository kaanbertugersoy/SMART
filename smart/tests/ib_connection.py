from ib_insync import *
import pandas as pd

ib = IB()

ib.connect(host='127.0.0.1', port=7496, clientId=1)

ib.isConnected()

contract = Stock("AAPL", "SMART", "USD")

contract

ib.qualifyContracts(contract)

data = ib.reqMktData(contract)

data

data_1 = ib.reqContractDetails(contract)

data_1

df = pd.DataFrame(data_1)

df

ib.disconnect()
