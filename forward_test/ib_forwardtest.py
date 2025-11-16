"""
Interactive Brokers ORB Forward Test Script
-------------------------------------------
Real-time ORB strategy using IBKR paper account.

- Waits for market open (9:30 AM)
- Collects 5-min bars until 9:45 to form ORB range
- Places Buy/Sell orders on breakout
- Applies SL:TP = 2:1
- Logs latency and trade details

Requirements:
    pip install ib_insync pandas

Make sure:
    - IB Gateway or TWS is running (Paper account, port 7497)
    - You have delayed or live market data access
"""

from ib_insync import *
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# --- CONFIG ---
SYMBOL = "MNQ"
BAR_SIZE = "5 mins"
ORB_START = "09:30"
ORB_END = "09:45"
SL_TP_RATIO = (2, 1)   # SL : TP = 2 : 1 for forward test
DATA_DIR = "forward_data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- CONNECT ---
ib = IB()
ib.connect("127.0.0.1", 7497, clientId=2)
ib.reqMarketDataType(3)  # 3 = delayed data

# --- CREATE CONTRACT ---
def create_future(symbol):
    contract = Future(symbol=symbol, exchange="CME", currency="USD")
    return contract

contract = create_future(SYMBOL)

# --- STREAM DATA ---
print(f"Starting forward test for {SYMBOL}...")
bars = ib.reqRealTimeBars(contract, 5, "TRADES", True)
live_data = []

start_time = datetime.now()
print("Waiting for market to open at 9:30 AM...")

# Wait until 9:30 AM
while True:
    now = datetime.now()
    if now.time() >= datetime.strptime(ORB_START, "%H:%M").time():
        break
    time.sleep(5)

print("Market open detected! Collecting ORB data (9:30–9:45)...")

# Collect bars until 9:45 AM
while True:
    ib.sleep(5)
    if len(bars) > 0:
        last = bars[-1]
        latency = (datetime.now() - last.time.replace(tzinfo=None)).total_seconds()
        live_data.append([last.time, last.open, last.high, last.low, last.close, latency])
        print(f"{last.time}: Close={last.close}, Delay={latency:.2f}s")

    if datetime.now().time() >= datetime.strptime(ORB_END, "%H:%M").time():
        break

# --- CREATE ORB RANGE ---
df = pd.DataFrame(live_data, columns=["time", "open", "high", "low", "close", "latency"])
df.to_csv(f"{DATA_DIR}/{SYMBOL}_orb_data.csv", index=False)
orb_high = df["high"].max()
orb_low = df["low"].min()
avg_delay = df["latency"].mean()
print(f"\nORB High={orb_high:.2f}, ORB Low={orb_low:.2f}, Avg Delay={avg_delay:.2f}s")

# --- PLACE ORDERS ---
print("Monitoring breakout for entries...")

order_id = 1
active_order = None
entry_price = None
direction = None
stop_loss = None
take_profit = None

while True:
    ib.sleep(5)
    latest_bar = bars[-1]
    price = latest_bar.close
    latency = (datetime.now() - latest_bar.time.replace(tzinfo=None)).total_seconds()

    # Entry Conditions
    if active_order is None:
        if price > orb_high:
            direction = "LONG"
            entry_price = price
            stop_loss = entry_price - 2 * (entry_price - orb_low)
            take_profit = entry_price + 1 * (entry_price - orb_low)
        elif price < orb_low:
            direction = "SHORT"
            entry_price = price
            stop_loss = entry_price + 2 * (orb_high - entry_price)
            take_profit = entry_price - 1 * (orb_high - entry_price)

        if direction:
            action = "BUY" if direction == "LONG" else "SELL"
            order = MarketOrder(action, 1)
            trade = ib.placeOrder(contract, order)
            active_order = trade
            print(f"{direction} Entry at {entry_price:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f}")

    # Exit Conditions
    if direction == "LONG":
        if price <= stop_loss:
            print(f"LONG Stop hit at {price:.2f}")
            break
        elif price >= take_profit:
            print(f"LONG Target hit at {price:.2f}")
            break
    elif direction == "SHORT":
        if price >= stop_loss:
            print(f"SHORT Stop hit at {price:.2f}")
            break
        elif price <= take_profit:
            print(f"SHORT Target hit at {price:.2f}")
            break

    # log latency continuously
    print(f"Live={price:.2f} | Delay={latency:.2f}s")
    df = pd.concat([df, pd.DataFrame([[datetime.now(), price, latency]], columns=["time", "price", "latency"])])
    df.to_csv(f"{DATA_DIR}/{SYMBOL}_latency_log.csv", index=False)

print("\nForward test completed. Results saved in /forward_data/")
ib.disconnect()
