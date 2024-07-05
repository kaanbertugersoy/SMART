from ib_insync import *


def execute_market_order(ib, target, cfd, contractId):
    # Get current position
    current_pos = 0
    for pos in ib.positions():
        if pos.contract.conId == contractId:
            current_pos = pos.position
            break

    # Identify required trades
    trades = target - current_pos

    # Trade execution
    if trades > 0:
        side = "BUY"
    elif trades < 0:
        side = "SELL"
    else:
        return

    order = MarketOrder(side, abs(trades))
    trade = ib.placeOrder(cfd, order)
