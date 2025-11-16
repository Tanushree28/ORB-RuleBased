import os
import datetime as dt
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

TP_MULTIPLIER = 2.0
os.makedirs("ib_data", exist_ok=True)
summary_path = os.path.join("ib_data", "IB_Daily_ORB_Summary.txt")
summary_lines = []

class App(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.contracts = {}

    def historicalData(self, reqId, bar):
        if reqId not in self.data:
            self.data[reqId] = []
        self.data[reqId].append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close
        })

    def historicalDataEnd(self, reqId, start, end):
        symbol = self.contracts[reqId]
        print(f"\nData complete for {symbol}. Running Daily ORB backtest...")
        self.run_daily_orb(symbol, reqId)

    def run_daily_orb(self, symbol, reqId):
        df = pd.DataFrame(self.data[reqId])
        if df.empty:
            print(f"No data for {symbol}")
            return

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df.to_excel(os.path.join("ib_data", f"{symbol}_raw.xlsx"), index=False)

        results = []
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            or_high, or_low = prev['high'], prev['low']
            or_range = or_high - or_low
            if or_range <= 0:
                continue

            trade = {'date': curr['date'], 'range': or_range, 'outcome': 'NoTrade', 'pnl': 0.0}

            # Long breakout
            if curr['high'] > or_high:
                entry = or_high
                sl = or_low
                tp = or_high + TP_MULTIPLIER * or_range
                if curr['low'] <= sl:
                    trade.update({'outcome': 'SL', 'pnl': -1})
                elif curr['high'] >= tp:
                    trade.update({'outcome': 'TP', 'pnl': +2})
                else:
                    # End of day close
                    trade.update({'outcome': 'EOD', 'pnl': (curr['close'] - entry) / or_range})

            # Short breakout
            elif curr['low'] < or_low:
                entry = or_low
                sl = or_high
                tp = or_low - TP_MULTIPLIER * or_range
                if curr['high'] >= sl:
                    trade.update({'outcome': 'SL', 'pnl': -1})
                elif curr['low'] <= tp:
                    trade.update({'outcome': 'TP', 'pnl': +2})
                else:
                    trade.update({'outcome': 'EOD', 'pnl': (entry - curr['close']) / or_range})

            results.append(trade)

        trades = pd.DataFrame(results)
        if trades.empty:
            summary_lines.append(f"\n{symbol}\n No trades generated.\n")
            return

        total = len(trades)
        wins = (trades['pnl'] > 0).sum()
        losses = (trades['pnl'] < 0).sum()
        no_trades = (trades['pnl'] == 0).sum()
        win_rate = 100 * wins / total
        avg_pnl = trades['pnl'].mean()
        cum_pnl = trades['pnl'].sum()

        trades.to_excel(os.path.join("ib_data", f"{symbol}_daily_report.xlsx"), index=False)

        summary = (
            f"\n====== {symbol} DAILY ORB Summary ======\n"
            f"Period: {df['date'].min().date()} → {df['date'].max().date()}\n"
            f"Total Trades: {total}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Average PnL (R): {avg_pnl:.2f}\n"
            f"Cumulative PnL (R): {cum_pnl:.2f}\n"
            f"No-Trade Days: {no_trades}\n"
            f"========================================\n"
        )
        print(summary)
        summary_lines.append(summary)

        with open("ib_data/IB_Daily_ORB_Summary.txt", "a", encoding="utf-8") as f:
            f.write(summary)


def create_future_contract(symbol, expiry):
    c = Contract()
    c.symbol = symbol
    c.secType = "FUT"
    c.exchange = "CME"
    c.currency = "USD"
    c.lastTradeDateOrContractMonth = expiry
    return c

if __name__ == "__main__":
    app = App()
    app.connect("127.0.0.1", 7497, clientId=1)

    mnq = create_future_contract("MNQ", "202512")
    nq = create_future_contract("NQ", "202512")

    app.contracts = {1: "MNQ", 2: "NQ"}
    app.reqHistoricalData(1, mnq, '', '2 Y', '1 day', 'TRADES', 1, 1, False, [])
    app.reqHistoricalData(2, nq, '', '2 Y', '1 day', 'TRADES', 1, 1, False, [])
    app.run()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print(f"\nCombined summary saved to {summary_path}")
