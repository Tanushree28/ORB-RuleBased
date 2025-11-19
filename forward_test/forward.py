"""
forward_orb_ib.py
-----------------
15-minute Opening Range Breakout using 5-minute bars (Interactive Brokers paper account).
"""

from ib_insync import *
import pandas as pd
from datetime import datetime, time, timedelta
import pytz, sys, time as pytime, os

# === Configuration ===
HOST, PORT, CLIENT_ID = "127.0.0.1", 7497, 1
SYMBOL, EXCHANGE, CURRENCY, EXPIRY = "MNQ", "CME", "USD", "202512"

OPEN_START, OPEN_END = time(9, 30), time(9, 45)
BAR_SIZE = "5 mins"
RISK_PER_TRADE = 0.01
TP_MULTIPLIER = 2.0


def create_contract():
    return Future(
        symbol=SYMBOL,
        lastTradeDateOrContractMonth=EXPIRY,
        exchange=EXCHANGE,
        currency=CURRENCY,
    )


def place_bracket_order(ib, contract, qty, entry_price, or_range, is_long):
    """Submit a bracket order (2:1 RR) and log latency."""
    action = "BUY" if is_long else "SELL"
    sl_price = entry_price - or_range if is_long else entry_price + or_range
    tp_price = (
        entry_price + TP_MULTIPLIER * or_range
        if is_long
        else entry_price - TP_MULTIPLIER * or_range
    )
    parent = LimitOrder(action, qty, entry_price, transmit=False)
    stop = StopOrder(
        "SELL" if is_long else "BUY",
        qty,
        sl_price,
        parentId=parent.orderId,
        transmit=False,
    )
    takep = LimitOrder(
        "SELL" if is_long else "BUY",
        qty,
        tp_price,
        parentId=parent.orderId,
        transmit=True,
    )

    print(
        f"Submitting {action} at {entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}"
    )

    # Measure API latency
    start = pytime.time()
    ib.placeOrder(contract, parent)
    ib.placeOrder(contract, stop)
    ib.placeOrder(contract, takep)
    ib.sleep(0.5)
    latency = (pytime.time() - start) * 1000
    print(f"Bracket order sent (paper). Latency: {latency:.2f} ms")

    with open("trade_latency_log.csv", "a") as f:
        f.write(f"{datetime.now()}, {action}, {entry_price}, {latency:.2f}\n")


def main():
    # mark script start time
    script_start = pytime.time()

    util.startLoop()
    ib = IB()
    ib.connect(HOST, PORT, CLIENT_ID)
    if not ib.isConnected():
        print("Not connected. Enable API in TWS/Gateway.")
        return
    print("Connected to IB Paper Account.")

    # --- account info
    net_liq = float(
        next(x.value for x in ib.accountSummary() if x.tag == "NetLiquidation")
    )
    print(f"Net Liquidation Value: {net_liq:,.2f} USD")

    contract = create_contract()

    # --- subscribe bars
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="2 D",
        barSizeSetting=BAR_SIZE,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
        keepUpToDate=True,
    )
    print("Collecting bars for 9:30–9:45 ET...")

    # wait until 9:45 ET
    while datetime.now(pytz.timezone("US/Eastern")).time() < OPEN_END:
        ib.sleep(1)

    # compute ORB
    df = util.df(bars)
    df["datetime"] = pd.to_datetime(df["date"]).dt.tz_convert("US/Eastern")
    or_bars = df[
        (df["datetime"].dt.time >= OPEN_START) & (df["datetime"].dt.time < OPEN_END)
    ]
    if or_bars.empty:
        print("No opening bars collected.")
        ib.disconnect()
        return

    or_high, or_low = or_bars["high"].max(), or_bars["low"].min()
    or_range = or_high - or_low
    print(f"OR High {or_high}, Low {or_low}, Range {or_range}")

    risk_dollars = net_liq * RISK_PER_TRADE
    qty = max(int(risk_dollars / or_range), 1)
    print(f"Will trade {qty} contracts if breakout occurs.")

    # wait until 9:50 ET
    target = datetime.now(pytz.timezone("US/Eastern")).replace(
        hour=9, minute=50, second=0, microsecond=0
    )
    print("Waiting for 9:45–9:50 bar to close...")
    while datetime.now(pytz.timezone("US/Eastern")) < target:
        ib.sleep(1)

    # check breakout
    df = util.df(bars)
    df["datetime"] = pd.to_datetime(df["date"]).dt.tz_convert("US/Eastern")
    latest = df.iloc[-1]
    price = latest["close"]
    print(f"9:45–9:50 bar closed at {price:.2f}")

    if price > or_high:
        print("Long breakout detected.")
        place_bracket_order(ib, contract, qty, or_high, or_range, True)
    elif price < or_low:
        print("Short breakout detected.")
        place_bracket_order(ib, contract, qty, or_low, or_range, False)
    else:
        print("No breakout — exiting without trade.")

    ib.cancelHistoricalData(bars)
    ib.disconnect()

    # mark end time and show total runtime
    script_end = pytime.time()
    total_sec = script_end - script_start
    print(f"\nTotal runtime: {total_sec:.1f} sec ({total_sec / 60:.2f} min)")
    print("Forward test complete.")


if __name__ == "__main__":
    main()
