from ib_insync import *
import pandas as pd
from datetime import datetime, time
import pytz

# -----------------------------
# Configuration
# -----------------------------
HOST = "127.0.0.1"
PORT = 7497
CLIENT_ID = 2

SYMBOL = "MNQ"
EXCHANGE = "CME"
CURRENCY = "USD"
EXPIRY = "202512"  # adjust if needed

# Time window
START_T = time(9, 0)
END_T = time(10, 0)

TZ_ET = pytz.timezone("US/Eastern")


# -----------------------------
# Create Contract
# -----------------------------
def create_contract():
    return Future(
        symbol=SYMBOL,
        lastTradeDateOrContractMonth=EXPIRY,
        exchange=EXCHANGE,
        currency=CURRENCY,
    )


# -----------------------------
# Main
# -----------------------------
def main():
    util.startLoop()
    ib = IB()

    print("Connecting to IBKR...")
    ib.connect(HOST, PORT, CLIENT_ID)

    if not ib.isConnected():
        print("❌ Could not connect. Enable API in TWS.")
        return

    print("Connected!")

    contract = create_contract()

    print("Requesting 1-minute MNQ data...")
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

    if df.empty:
        print("❌ No data returned from IBKR.")
        ib.disconnect()
        return

    # Convert to ET timezone
    df["datetime"] = pd.to_datetime(df["date"]).dt.tz_convert(TZ_ET)

    # Filter for today's date
    today = datetime.now(TZ_ET).date()
    df = df[df["datetime"].dt.date == today]

    # Filter 9:00–10:00 AM ET
    mask = (df["datetime"].dt.time >= START_T) & (df["datetime"].dt.time < END_T)
    window_df = df.loc[mask, ["datetime", "open", "high", "low", "close", "volume"]]

    print("\n=== MNQ 1-Minute Data (9:00–10:00 AM ET) ===")
    print(window_df)

    # Save timezone-naive version for Excel
    out = window_df.copy()
    out["datetime"] = out["datetime"].dt.tz_localize(None)

    # Save to files
    out.to_csv("mnq_1m_9_to_10.csv", index=False)
    out.to_excel("mnq_1m_9_to_10.xlsx", index=False)

    print("\n📁 Saved:")
    print("  mnq_1m_9_to_10.csv")
    print("  mnq_1m_9_to_10.xlsx")

    ib.disconnect()


if __name__ == "__main__":
    main()
