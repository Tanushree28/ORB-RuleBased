"""
forward_orb_ib_v2.py
---------------------------------
Opening Range Breakout (ORB) using IBKR data.
Upgraded version:
- Continuous breakout monitoring after ORB window
- Correct MNQ futures contract (CME)
- Delayed-feed safe
- 2:1 Risk/Reward
"""

from ib_insync import *
import pandas as pd
from datetime import datetime, time
import pytz, sys, time as pytime

# === Configuration ===
HOST, PORT, CLIENT_ID = "127.0.0.1", 7497, 1

SYMBOL = "MNQ"
EXCHANGE = "CME"
CURRENCY = "USD"
EXPIRY = "202512"
TRADING_CLASS = "MNQ"

OPEN_START = time(9, 30)
OPEN_END = time(9, 45)

BAR_SIZE = "1 min"
RISK_PER_TRADE = 0.01
TP_MULTIPLIER = 2.0


# =====================================================
# Correct MNQ Contract Definition
# =====================================================
def create_contract():
    return Future(
        symbol=SYMBOL,
        lastTradeDateOrContractMonth=EXPIRY,
        exchange=EXCHANGE,
        currency=CURRENCY,
        tradingClass=TRADING_CLASS,
    )


# =====================================================
# Place Bracket Order
# =====================================================
def place_bracket_order(ib, contract, qty, entry_price, or_range, is_long):
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
        f"\nSubmitting {action} at {entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}"
    )

    t0 = pytime.time()
    ib.placeOrder(contract, parent)
    ib.placeOrder(contract, stop)
    ib.placeOrder(contract, takep)
    ib.sleep(0.3)

    latency = (pytime.time() - t0) * 1000
    print(f"Order latency: {latency:.2f} ms")


# =====================================================
# Main Forward Test
# =====================================================
def main():
    util.startLoop()
    ib = IB()
    ib.connect(HOST, PORT, CLIENT_ID)

    if not ib.isConnected():
        print("❌ IBKR not connected.")
        return

    print("Connected to IBKR.")

    contract = create_contract()

    # Use delayed free data
    ib.reqMarketDataType(3)

    # ---- Account ----
    net_liq = float(
        next(x.value for x in ib.accountSummary() if x.tag == "NetLiquidation")
    )
    print(f"\nNetLiq: {net_liq:,.2f} USD")

    # ---- Request Bars ----
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

    print("\nCollecting ORB bars (9:30–9:45)...")

    # Wait until ORB window ends
    et = pytz.timezone("US/Eastern")
    while datetime.now(et).time() < OPEN_END:
        ib.sleep(1)

    df = util.df(bars)
    df["datetime"] = pd.to_datetime(df["date"]).dt.tz_convert(et)

    orb_df = df[
        (df["datetime"].dt.time >= OPEN_START) & (df["datetime"].dt.time < OPEN_END)
    ]

    if orb_df.empty:
        print("❌ No ORB bars collected.")
        ib.disconnect()
        return

    or_high = orb_df["high"].max()
    or_low = orb_df["low"].min()
    or_range = or_high - or_low

    print(f"\n=== ORB RANGE ===")
    print(f"High:  {or_high:.2f}")
    print(f"Low:   {or_low:.2f}")
    print(f"Range: {or_range:.2f} pts")

    # ---- Position Sizing ----
    risk_dollars = net_liq * RISK_PER_TRADE
    qty = max(int(risk_dollars / or_range), 1)
    print(f"\nPosition Size = {qty} contracts")

    # =====================================================
    # NEW LOGIC: Continuous Breakout Monitoring
    # =====================================================
    print("\nMonitoring for breakout after 9:45...")

    breakout_triggered = False

    while not breakout_triggered:
        df = util.df(bars)
        df["datetime"] = pd.to_datetime(df["date"]).dt.tz_convert(et)
        last = df.iloc[-1]

        bar_high = last["high"]
        bar_low = last["low"]
        price = last["close"]

        print(f"Last Price {price:.2f} | High {bar_high:.2f} | Low {bar_low:.2f}")

        # === Long Breakout ===
        if bar_high > or_high:
            print(f"\n📈 LONG BREAKOUT at {price:.2f}")
            place_bracket_order(ib, contract, qty, or_high, or_range, True)
            breakout_triggered = True

        # === Short Breakout ===
        elif bar_low < or_low:
            print(f"\n📉 SHORT BREAKOUT at {price:.2f}")
            place_bracket_order(ib, contract, qty, or_low, or_range, False)
            breakout_triggered = True

        else:
            ib.sleep(5)  # Check every 5 seconds

    ib.disconnect()
    print("\nForward test completed.")


if __name__ == "__main__":
    main()
