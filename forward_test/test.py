"""
forward_orb_ib_v3.py
---------------------------------
15-Minute Opening Range Breakout using IBKR delayed data.

- Computes ORB (9:30–9:45 ET)
- Polls 1-minute bars every 10 seconds
- Detects breakout after ORB
- Places 2:1 RR bracket order
"""

from ib_insync import *
import pandas as pd
from datetime import datetime, time
import pytz, time as pytime

# === Configuration ===
HOST, PORT, CLIENT_ID = "127.0.0.1", 7497, 2
SYMBOL, EXCHANGE, CURRENCY, EXPIRY, TRADING_CLASS = "MNQ", "CME", "USD", "202512", "MNQ"
TZ_ET = pytz.timezone("US/Eastern")
ORB_START, ORB_END = time(9, 30), time(9, 45)
RISK_PER_TRADE, TP_MULTIPLIER = 0.01, 2.0


def create_contract():
    return Future(
        symbol=SYMBOL,
        exchange=EXCHANGE,
        currency=CURRENCY,
        lastTradeDateOrContractMonth=EXPIRY,
        tradingClass=TRADING_CLASS,
    )


def place_bracket_order(ib, contract, qty, entry, rng, is_long):
    """Place 2:1 RR bracket order and record latency."""
    action = "BUY" if is_long else "SELL"
    sl = entry - rng if is_long else entry + rng
    tp = entry + TP_MULTIPLIER * rng if is_long else entry - TP_MULTIPLIER * rng

    parent = LimitOrder(action, qty, entry, transmit=False)
    stop = StopOrder(
        "SELL" if is_long else "BUY", qty, sl, parentId=parent.orderId, transmit=False
    )
    takep = LimitOrder(
        "SELL" if is_long else "BUY", qty, tp, parentId=parent.orderId, transmit=True
    )

    print(f"\nSubmitting {action} order → Entry={entry:.2f}, SL={sl:.2f}, TP={tp:.2f}")
    t0 = pytime.time()
    ib.placeOrder(contract, parent)
    ib.placeOrder(contract, stop)
    ib.placeOrder(contract, takep)
    ib.sleep(0.3)
    print(f"Latency: {(pytime.time() - t0) * 1000:.2f} ms")


def main():
    util.startLoop()
    ib = IB()

    print("Connecting to IBKR…")
    ib.connect(HOST, PORT, CLIENT_ID)
    if not ib.isConnected():
        print("❌ Connection failed.")
        return
    print("✅ Connected!")

    ib.reqMarketDataType(3)  # 3 = delayed-free
    net_liq = float(
        next(x.value for x in ib.accountSummary() if x.tag == "NetLiquidation")
    )
    print(f"NetLiq: {net_liq:,.2f} USD")

    contract = create_contract()

    # === Step 1 – Fetch ORB bars
    print("\nFetching 1-min bars for ORB (9:30–9:45 ET)…")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="2 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    df = util.df(bars)
    df["datetime"] = pd.to_datetime(df["date"]).dt.tz_convert(TZ_ET)
    today = datetime.now(TZ_ET).date()
    df = df[df["datetime"].dt.date == today]

    orb_df = df[
        (df["datetime"].dt.time >= ORB_START) & (df["datetime"].dt.time < ORB_END)
    ]
    if orb_df.empty:
        print("❌ No ORB data. Exiting.")
        ib.disconnect()
        return

    OR_HIGH, OR_LOW = orb_df["high"].max(), orb_df["low"].min()
    OR_RANGE = OR_HIGH - OR_LOW
    print(f"\nORB High {OR_HIGH:.2f}  Low {OR_LOW:.2f}  Range {OR_RANGE:.2f}")

    qty = max(int((net_liq * RISK_PER_TRADE) / OR_RANGE), 1)
    print(f"Position size: {qty} contracts")

    # === Step 2 – Poll for breakout
    print("\nMonitoring for breakout after 9:45 ET…")
    breakout_found = False

    while not breakout_found:
        latest = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="1200 S",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not latest:
            print("⚠ No bars returned – retrying…")
            pytime.sleep(5)
            continue

        df2 = util.df(latest)
        df2["datetime"] = pd.to_datetime(df2["date"]).dt.tz_convert(TZ_ET)
        df2 = df2[df2["datetime"].dt.date == today]
        last = df2.iloc[-1]
        t, p, h, l = last["datetime"], last["close"], last["high"], last["low"]
        print(f"{t} → Price={p:.2f} High={h:.2f} Low={l:.2f}")

        if h > OR_HIGH:
            breakout_time = t.strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"\n📈 LONG breakout detected @ {breakout_time}  price={p:.2f}")
            place_bracket_order(ib, contract, qty, OR_HIGH, OR_RANGE, True)
            breakout_found = True

        elif l < OR_LOW:
            breakout_time = t.strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"\n📉 SHORT breakout detected @ {breakout_time}  price={p:.2f}")

            place_bracket_order(ib, contract, qty, OR_LOW, OR_RANGE, False)
            breakout_found = True

        else:
            pytime.sleep(10)

    ib.disconnect()
    print("\n✅ Forward test complete.")


if __name__ == "__main__":
    main()
