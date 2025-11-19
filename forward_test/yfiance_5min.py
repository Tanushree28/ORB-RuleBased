import yfinance as yf
import pandas as pd
from datetime import datetime, time
import pytz

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
TICKER = "MNQ=F"
INTERVAL = "1m"  # change to "5m" if you want 5-minute candles
EXPORT = True  # ALWAYS export CSV + Excel

TZ_ET = pytz.timezone("US/Eastern")

# ORB window (9:30–9:45 AM ET)
OPEN_START = time(9, 30)
OPEN_END = time(9, 45)

# Extra window: 9:00–10:00 AM
WIN_START = time(9, 0)
WIN_END = time(10, 0)


# ---------------------------------------------------------
# DOWNLOAD DATA
# ---------------------------------------------------------
def download_data():
    df = yf.download(
        TICKER,
        period="1d",
        interval=INTERVAL,
        prepost=False,
        progress=False,
    )

    if df.empty:
        print("❌ No data returned.")
        return None

    # Flatten multi-index
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Ensure timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(TZ_ET)
    else:
        df.index = df.index.tz_convert(TZ_ET)

    # Only today
    today = datetime.now(TZ_ET).date()
    df = df[df.index.date == today]

    if df.empty:
        print("❌ No today's data found.")
        return None

    return df


# ---------------------------------------------------------
# TIME WINDOW FILTER
# ---------------------------------------------------------
def get_window(df, start_t, end_t):
    mask = (df.index.time >= start_t) & (df.index.time < end_t)
    return df.loc[mask]


# ---------------------------------------------------------
# CALCULATE STATS
# ---------------------------------------------------------
def compute_stats(df):
    orb_df = get_window(df, OPEN_START, OPEN_END)
    if orb_df.empty:
        print("❌ No ORB window found.")
        return None

    return {
        "OR_High": float(orb_df["High"].max()),
        "OR_Low": float(orb_df["Low"].min()),
        "OR_Range": float(orb_df["High"].max() - orb_df["Low"].min()),
        "Day_High": float(df["High"].max()),
        "Day_Low": float(df["Low"].min()),
        "Day_Range": float(df["High"].max() - df["Low"].min()),
    }


# ---------------------------------------------------------
# SAVE CSV + EXCEL
# ---------------------------------------------------------
def save_files(df, stats, df_9_10):
    today = datetime.now().strftime("%Y-%m-%d")

    # ---- FIX: remove timezone for Excel ----
    df = df.copy()
    df_9_10 = df_9_10.copy()

    df.index = df.index.tz_localize(None)
    df_9_10.index = df_9_10.index.tz_localize(None)

    # CSV paths
    df.to_csv(f"candles_full_{INTERVAL}_{today}.csv")
    pd.DataFrame([stats]).to_csv(f"orb_stats_{INTERVAL}_{today}.csv", index=False)
    df_9_10.to_csv(f"window_9_10_{INTERVAL}_{today}.csv")

    # # Excel paths
    # df.to_excel(f"candles_full_{INTERVAL}_{today}.xlsx")
    # pd.DataFrame([stats]).to_excel(f"orb_stats_{INTERVAL}_{today}.xlsx", index=False)
    # df_9_10.to_excel(f"window_9_10_{INTERVAL}_{today}.xlsx")

    print("\nExported CSV + Excel successfully.")


# ---------------------------------------------------------
# MAIN (auto-run)
# ---------------------------------------------------------
def main():
    print(f"\n=== Running {INTERVAL} data for {TICKER} ===")

    df = download_data()
    if df is None:
        return

    stats = compute_stats(df)
    if stats is None:
        return

    df_9_10 = get_window(df, WIN_START, WIN_END)

    today = datetime.now(TZ_ET).date()
    print(f"\nTicker: {TICKER} | Date: {today}")

    print("\n--- ORB (9:30–9:45) ---")
    print(stats)

    print("\n--- 9:00–10:00 AM Minute-by-Minute ---")
    print(df_9_10)

    if EXPORT:
        save_files(df, stats, df_9_10)


# Auto-run
if __name__ == "__main__":
    main()
