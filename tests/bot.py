from ib_insync import *
util.startLoop()

ib = IB()

ib.connect(host='127.0.0.1', port=7497, clientId=1)

ib.positions()

contract = Forex('EURUSD')
ib.reqMktData(contract)
ticker = ib.ticker(contract)

ticker


def onPendingTickers(tickers):
    print("time: {} | Bid: {} | Ask: {}".format(
        ticker.time, ticker.bid, ticker.ask))


ib.pendingTickersEvent += onPendingTickers

ib.pendingTickersEvent -= onPendingTickers

ib.cancelMktData(contract)

ib.disconnect()
