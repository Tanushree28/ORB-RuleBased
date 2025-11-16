# ===============================================================
#  alpaca_orb_multi.py
#  Fetch 2 years of 5-min bars for multiple symbols -> ORB backtest
#  Save per-symbol Excel + combined summary.txt
# ===============================================================

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from pytz import timezone
import pandas as pd
import datetime as dt
import os

API_KEY  = 'PKNAYVMKOPTITGODCCF2BFIF7L'
API_SECRET = '2QHmMRVqELazbMVoMC9NqX88H8sHBa2NkmmwpXXuVhjx'

TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)
LOOKBACK_DAYS = 730
TP_MULTIPLIER = 2.0

SYMBOLS = ["AAPL", "TSLA", "AMZN", "GOOGL", "MSFT", "META", "NQ=F", "MNQ=F", "EURUSD", "GC=F"]

os.makedirs("alpaca_data", exist_ok=True)
summary_path = os.path.join("alpaca_data", "ORB_Summary.txt")
summary_lines = []

client = StockHistoricalDataClient(API_KEY, API_SECRET)
start_date = dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)
end_date   = dt.date.today()

for symbol in SYMBOLS:
    print(f"\nFetching {symbol} ...")
    try:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TIMEFRAME,
            start=start_date,
            end=end_date
        )
        bars = client.get_stock_bars(request)
        df = bars.df.reset_index()
        if df.empty:
            print(f"No data for {symbol}")
            continue

        df['timestamp'] = df['timestamp'].dt.tz_convert(timezone('US/Eastern')).dt.tz_localize(None)
        raw_path = os.path.join("alpaca_data", f"{symbol}_raw.xlsx")
        df.to_excel(raw_path, index=False)

        # ----- ORB Backtest -----
        results = []
        for date, daily in df.groupby(df['timestamp'].dt.date):
            open_start, open_end = dt.time(9,30), dt.time(9,45)
            opening = daily[(daily['timestamp'].dt.time >= open_start) & (daily['timestamp'].dt.time < open_end)]
            trading = daily[daily['timestamp'].dt.time >= open_end]
            if opening.empty or trading.empty:
                continue
            or_high, or_low = opening['high'].max(), opening['low'].min()
            or_range = or_high - or_low
            trade = {'date': date, 'range': or_range, 'outcome': 'NoTrade', 'pnl': 0.0}
            triggered = False
            for _, row in trading.iterrows():
                price = row['close']
                # Long
                if not triggered and price > or_high:
                    triggered = True
                    entry = price
                    sl, tp = or_low, or_high + TP_MULTIPLIER * or_range
                    for _, tr in trading.iterrows():
                        if tr['low'] <= sl:
                            trade.update({'outcome': 'SL', 'pnl': -1})
                            break
                        if tr['high'] >= tp:
                            trade.update({'outcome': 'TP', 'pnl': +2})
                            break
                    else:
                        trade.update({'outcome': 'EOD',
                                      'pnl': (trading.iloc[-1]['close'] - entry)/or_range})
                    break
                # Short
                elif not triggered and price < or_low:
                    triggered = True
                    entry = price
                    sl, tp = or_high, or_low - TP_MULTIPLIER * or_range
                    for _, tr in trading.iterrows():
                        if tr['high'] >= sl:
                            trade.update({'outcome': 'SL', 'pnl': -1})
                            break
                        if tr['low'] <= tp:
                            trade.update({'outcome': 'TP', 'pnl': +2})
                            break
                    else:
                        trade.update({'outcome': 'EOD',
                                      'pnl': (entry - trading.iloc[-1]['close'])/or_range})
                    break
            results.append(trade)

        trades = pd.DataFrame(results)
        if trades.empty:
            summary_lines.append(f"\n{symbol}\nNo trades generated.\n")
            continue

        total = len(trades)
        wins = (trades['pnl'] > 0).sum()
        losses = (trades['pnl'] < 0).sum()
        no_trades = (trades['pnl'] == 0).sum()
        win_rate = 100 * wins / total
        avg_pnl = trades['pnl'].mean()
        cum_pnl = trades['pnl'].sum()

        # save detailed report
        report_path = os.path.join("alpaca_data", f"{symbol}_report.xlsx")
        trades.to_excel(report_path, index=False)

        # save summary lines
        summary = (
            f"\n====== {symbol} ORB Summary ======\n"
            f"Period: {start_date} → {end_date}\n"
            f"Total Trades: {total}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Average PnL (R): {avg_pnl:.2f}\n"
            f"Cumulative PnL (R): {cum_pnl:.2f}\n"
            f"No-Trade Days: {no_trades}\n"
            f"==================================\n"
        )
        print(summary)
        summary_lines.append(summary)

    except Exception as e:
        summary_lines.append(f"\n{symbol}\nError: {e}\n")
        print(f"{symbol} failed: {e}")

# write all summaries to one text file
with open(summary_path, "w", encoding="utf-8") as f:
    f.writelines(summary_lines)
print(f"\nCombined summary saved to {summary_path}")
