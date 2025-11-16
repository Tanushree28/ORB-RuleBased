# """
# Massive / Polygon Futures ORB Backtest
# ======================================

# This script fetches historical intraday futures data (e.g., MNQZ4, NQZ4) from Polygon / Massive (formerly Polygon.io) and runs an Opening Range Breakout (ORB)
# backtest with a 2:1 TP/SL ratio.

# Example:
#     CME:MNQZ4  → Micro E-mini Nasdaq December 2024
#     CME:NQZ4   → E-mini Nasdaq December 2024

# Usage:
#     1. Install dependencies:
#        pip install pandas requests openpyxl
#     2. Replace API_KEY with your Massive/Polygon API key.
#     3. Run:
#        python massive_orb_backtest.py
# """

# import os
# import datetime as dt
# from typing import List, Dict, Any
# import pandas as pd
# import requests


# # ==============================================================
# # 1. Fetch historical bars from Polygon/Massive API
# # ==============================================================
# def fetch_massive_futures_bars(api_key: str,
#                                contract: str,
#                                start_date: dt.date,
#                                end_date: dt.date,
#                                multiplier: int = 5,
#                                timespan: str = "minute") -> pd.DataFrame:
#     """
#     Fetch aggregated OHLCV bars for a CME futures contract from Polygon/Massive.
#     This uses the standard v2 aggregate endpoint (works for all CME futures).

#     Example endpoint:
#       https://api.polygon.io/v2/aggs/ticker/CME:MNQZ4/range/5/minute/2024-01-01/2024-01-31?apiKey=...

#     Parameters
#     ----------
#     api_key : str
#         Your Massive (Polygon) API key.
#     contract : str
#         Futures contract symbol (e.g. "MNQZ4" for Dec 2024 Micro E-mini Nasdaq).
#     start_date : datetime.date
#         Start of range.
#     end_date : datetime.date
#         End of range.
#     multiplier : int
#         Bar size in minutes (5 = 5-minute bars).
#     timespan : str
#         "minute", "hour", or "day"

#     Returns
#     -------
#     pandas.DataFrame
#         Columns: [timestamp, open, high, low, close, volume]
#     """
#     base_url = "https://api.polygon.io"
#     endpoint = f"/v2/aggs/ticker/CME:{contract}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

#     params = {"apiKey": api_key}
#     response = requests.get(base_url + endpoint, params=params, timeout=30)

#     if response.status_code != 200:
#         raise RuntimeError(f"Polygon/Massive API returned {response.status_code}: {response.text}")

#     data = response.json()
#     bars = data.get("results", [])
#     if not bars:
#         return pd.DataFrame()

#     df = pd.DataFrame(bars)
#     df.rename(columns={
#         "t": "timestamp",
#         "o": "open",
#         "h": "high",
#         "l": "low",
#         "c": "close",
#         "v": "volume"
#     }, inplace=True)
#     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
#     df.sort_values("timestamp", inplace=True)
#     df.reset_index(drop=True, inplace=True)
#     return df


# # ==============================================================
# # 2. ORB backtest logic
# # ==============================================================
# def orb_backtest(df: pd.DataFrame,
#                  opening_start: dt.time,
#                  opening_end: dt.time,
#                  tp_multiplier: float = 2.0) -> pd.DataFrame:
#     """
#     Run an Opening Range Break-Out backtest on a DataFrame of bar data.
#     """
#     results: List[Dict[str, Any]] = []
#     if df.empty:
#         return pd.DataFrame()

#     for date, daily in df.groupby(df["timestamp"].dt.date):
#         opening = daily[(daily["timestamp"].dt.time >= opening_start) &
#                         (daily["timestamp"].dt.time < opening_end)]
#         trading = daily[daily["timestamp"].dt.time >= opening_end]
#         if opening.empty or trading.empty:
#             continue

#         or_high = opening["high"].max()
#         or_low = opening["low"].min()
#         or_range = or_high - or_low
#         trade = {"date": date, "range": or_range, "outcome": "NoTrade", "pnl": 0.0}
#         triggered = False

#         for _, row in trading.iterrows():
#             price = row["close"]
#             # Long breakout
#             if not triggered and price > or_high:
#                 triggered = True
#                 entry = price
#                 sl, tp = or_low, or_high + tp_multiplier * or_range
#                 for _, tr in trading.iterrows():
#                     if tr["low"] <= sl:
#                         trade.update({"outcome": "SL", "pnl": -1.0})
#                         break
#                     if tr["high"] >= tp:
#                         trade.update({"outcome": "TP", "pnl": tp_multiplier})
#                         break
#                 else:
#                     trade.update({"outcome": "EOD",
#                                   "pnl": (trading.iloc[-1]['close'] - entry) / or_range})
#                 break

#             # Short breakout
#             elif not triggered and price < or_low:
#                 triggered = True
#                 entry = price
#                 sl, tp = or_high, or_low - tp_multiplier * or_range
#                 for _, tr in trading.iterrows():
#                     if tr["high"] >= sl:
#                         trade.update({"outcome": "SL", "pnl": -1.0})
#                         break
#                     if tr["low"] <= tp:
#                         trade.update({"outcome": "TP", "pnl": tp_multiplier})
#                         break
#                 else:
#                     trade.update({"outcome": "EOD",
#                                   "pnl": (entry - trading.iloc[-1]['close']) / or_range})
#                 break
#         results.append(trade)
#     return pd.DataFrame(results)


# # ==============================================================
# # 3. Main function
# # ==============================================================
# def main():
#     API_KEY = "wP1mSgJ_h3rKAjDpc54vxTkAS_TL9a9v"  # 🔑 replace with your Massive/Polygon API key
#     CONTRACT = "MNQZ4"              # December 2024 Micro E-mini Nasdaq
#     LOOKBACK_DAYS = 730             # 2 years
#     MULTIPLIER = 5                  # 5-minute bars
#     OPEN_START_CT = dt.time(8, 30)  # 8:30–8:45 CT = 9:30–9:45 ET
#     OPEN_END_CT = dt.time(8, 45)
#     TP_MULTIPLIER = 2.0
#     OUTPUT_DIR = "massive_data"
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     end_date = dt.date.today()
#     start_date = end_date - dt.timedelta(days=LOOKBACK_DAYS)
#     print(f"Fetching {MULTIPLIER}-minute bars for {CONTRACT} from {start_date} to {end_date}...")

#     try:
#         df = fetch_massive_futures_bars(API_KEY, CONTRACT, start_date, end_date,
#                                         multiplier=MULTIPLIER, timespan="minute")
#     except RuntimeError as e:
#         print(e)
#         return

#     if df.empty:
#         print("No data returned. Check your contract symbol or date range.")
#         return

#     # Save raw data
#     raw_path = os.path.join(OUTPUT_DIR, f"{CONTRACT}_raw_bars.xlsx")
#     df.to_excel(raw_path, index=False)
#     print(f"✅ Saved {len(df)} bars to {raw_path}")

#     # Run ORB backtest
#     trades = orb_backtest(df, OPEN_START_CT, OPEN_END_CT, TP_MULTIPLIER)
#     if trades.empty:
#         print("No trades generated.")
#         return

#     total = len(trades)
#     wins = (trades["pnl"] > 0).sum()
#     losses = (trades["pnl"] < 0).sum()
#     no_trades = (trades["pnl"] == 0).sum()
#     win_rate = wins / total * 100
#     avg_pnl = trades["pnl"].mean()
#     cum_pnl = trades["pnl"].sum()

#     print("\n====== ORB Backtest Summary ======")
#     print(f"Contract: {CONTRACT}")
#     print(f"Period: {start_date} → {end_date}")
#     print(f"Total Trades: {total}")
#     print(f"Win Rate: {win_rate:.2f}%")
#     print(f"Average PnL (R): {avg_pnl:.2f}")
#     print(f"Cumulative PnL (R): {cum_pnl:.2f}")
#     print(f"No-Trade Days: {no_trades}")
#     print("==================================")

#     # Save trade-by-trade log
#     report_path = os.path.join(OUTPUT_DIR, f"{CONTRACT}_orb_report.xlsx")
#     trades.to_excel(report_path, index=False)
#     print(f"💾 Saved trade log to {report_path}")


# if __name__ == "__main__":
#     main()


# import requests
# API_KEY = "wP1mSgJ_h3rKAjDpc54vxTkAS_TL9a9v"

# url = f"https://api.polygon.io/v3/reference/tickers?market=futures&limit=50&apiKey={API_KEY}"
# r = requests.get(url)
# data = r.json()
# for res in data.get("results", []):
#     print(res["ticker"], res["name"])

import requests
API_KEY = "wP1mSgJ_h3rKAjDpc54vxTkAS_TL9a9v"
url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&limit=5&apiKey={API_KEY}"
print(requests.get(url).json())
