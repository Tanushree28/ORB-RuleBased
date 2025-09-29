# today_performance.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import time
from zoneinfo import ZoneInfo

SYMBOLS = [
    "AAPL",
    "AMZN",
    "EURUSD=X",
    "GC=F",
    "GLD",
    "GOOGL",
    "MNQ=F",
    "MSFT",
    "NQ=F",
    "QQQ",
    "SPY",
    "TSLA",
]
INTERVAL = "5m"
CAPITAL = 10000

OR_START = time(9, 30)
OR_END = time(9, 45)
TRADE_START = time(9, 45)
SESSION_END = time(16, 0)


def fetch_intraday(symbol):
    df = yf.Ticker(symbol).history(
        period="2d", interval=INTERVAL, auto_adjust=True, prepost=False
    )
    if df.empty:
        return pd.DataFrame()
    # Convert index to US/Eastern if UTC
    if df.index.tz is None:
        df = df.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df = df.tz_convert("US/Eastern")
    df = df.reset_index()
    return df


def filter_rth(df):
    df["t"] = df["Datetime"].dt.time
    return df[(df["t"] >= OR_START) & (df["t"] <= SESSION_END)].copy()


def compute_or(df_rth):
    mask = (df_rth["t"] >= OR_START) & (df_rth["t"] < OR_END)
    or_df = df_rth.loc[mask]
    if or_df.empty:
        raise ValueError("No OR bars")
    return or_df["High"].max(), or_df["Low"].min()


def trading_slice(df_rth):
    return df_rth[df_rth["t"] >= TRADE_START].copy()


def evaluate(trading_bars, direction, orb_high, orb_low, tp, risk_pct):
    risk_amt = CAPITAL * risk_pct
    entry = orb_high if direction == "long" else orb_low
    stop = orb_low if direction == "long" else orb_high
    rng = orb_high - orb_low
    target = (orb_high + tp * rng) if direction == "long" else (orb_low - tp * rng)

    triggered = False
    outcome = "No Trade"
    pnl = 0.0

    for _, row in trading_bars.iterrows():
        high = row["High"]
        low = row["Low"]
        if not triggered:
            if direction == "long" and high >= entry:
                triggered = True
            elif direction == "short" and low <= entry:
                triggered = True
        if not triggered:
            continue
        if direction == "long":
            if high >= target:
                outcome = "TP"
                pnl = risk_amt * tp
                break
            if low <= stop:
                outcome = "SL"
                pnl = -risk_amt
                break
        else:
            if low <= target:
                outcome = "TP"
                pnl = risk_amt * tp
                break
            if high >= stop:
                outcome = "SL"
                pnl = -risk_amt
                break
    else:
        if triggered:
            outcome = "No Exit"

    return triggered, outcome, pnl


def run_symbol(symbol):
    df = fetch_intraday(symbol)
    if df.empty:
        print(f"No data for {symbol}")
        return None
    df_rth = filter_rth(df)
    try:
        orb_high, orb_low = compute_or(df_rth)
    except ValueError as e:
        print(f"{symbol}: {e}")
        return None
    tb = trading_slice(df_rth)
    results = []
    for tp in [0.5, 1.0, 2.0]:
        for rp in [0.01, 0.02]:
            for direction in ("long", "short"):
                triggered, outcome, pnl = evaluate(
                    tb, direction, orb_high, orb_low, tp, rp
                )
                results.append(
                    {
                        "symbol": symbol,
                        "tp": tp,
                        "risk_pct": rp,
                        "direction": direction,
                        "triggered": triggered,
                        "outcome": outcome,
                        "pnl": pnl,
                    }
                )
    return pd.DataFrame(results)


def main():
    frames = []
    for sym in SYMBOLS:
        print("Processing", sym)
        df = run_symbol(sym)
        if df is not None:
            frames.append(df)
    if not frames:
        print("No results")
        return
    allres = pd.concat(frames, ignore_index=True)
    # simple summary: net PnL & trades triggered per symbol
    summary = (
        allres[allres["triggered"]]
        .groupby("symbol", as_index=False)
        .agg(net_pnl=("pnl", "sum"), trades=("outcome", "count"))
    )
    print("\n=== Today’s ORB Performance ===")
    print(summary.to_string(index=False, formatters={"net_pnl": lambda x: f"${x:.2f}"}))
    # Save to CSV
    summary.to_csv("today_orb_summary.csv", index=False)
    allres.to_csv("today_orb_full.csv", index=False)
    print("Saved summary to today_orb_summary.csv")


if __name__ == "__main__":
    main()
