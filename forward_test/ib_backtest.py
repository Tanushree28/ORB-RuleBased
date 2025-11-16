from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import datetime as dt
import time
import os


class HistApp(EWrapper, EClient):
    """Subclass EWrapper and EClient to handle historical data callbacks."""
    def __init__(self):
        EClient.__init__(self, self)
        # store the bars for each request
        self.df = pd.DataFrame()
        # internal flag used by fetch_symbol to know when a response has
        # completed
        self.end = False

    def historicalData(self, reqId, bar):
        """Callback for each historical bar received.

        Bars are appended to ``self.df`` with standard OHLCV fields.
        """
        self.df = self.df.append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
        }, ignore_index=True)

    def historicalDataEnd(self, reqId, start, end):
        """Callback triggered when a historical data request completes."""
        self.end = True


def fetch_symbol(app: HistApp, contract: Contract, months_back: int = 3,
                 bar_size: str = "5 mins") -> pd.DataFrame:
    """Fetch historical bars for a contract by stepping back month by month.

    Parameters
    ----------
    app : HistApp
        Connected IB API application instance.
    contract : Contract
        IB contract definition (stock, future, etc.).
    months_back : int, optional
        Number of months of history to retrieve (default 24 = 2 years).
    bar_size : str, optional
        Bar size string accepted by IB API (e.g. ``"5 mins"``).

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame containing all requested bars.

    Notes
    -----
    This function respects IB's pacing limits by waiting 11 seconds
    between requests and listens to the ``historicalDataEnd`` callback to
    detect when the server has finished sending bars.
    """
    all_data = []
    # start from the current time and step back month by month
    end_time = dt.datetime.now()
    for _ in range(months_back):
        end_str = end_time.strftime("%Y%m%d %H:%M:%S")
        app.end = False
        app.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=end_str,
            durationStr="1 M",  # one month of data per request
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=1,  # regular trading hours only
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )
        # wait until the callback sets the ``end`` flag
        while not app.end:
            time.sleep(1)
        all_data.append(app.df)
        # reset the DataFrame for the next iteration
        app.df = pd.DataFrame()
        # step back one month
        end_time -= dt.timedelta(days=30)
        # respect IB pacing: max 60 requests in 10 minutes
        time.sleep(11)
    return pd.concat(all_data).drop_duplicates().reset_index(drop=True)


def create_future(symbol: str, expiry: str = "202503") -> Contract:
    """Create a futures contract for CME Globex."""
    c = Contract()
    c.symbol = symbol
    c.secType = "FUT"
    c.exchange = "GLOBEX"
    c.currency = "USD"
    c.lastTradeDateOrContractMonth = expiry
    return c


def main() -> None:
    """Connect to IB, fetch historical data and write CSV files."""
    # ensure output directory exists
    output_dir = "b_data"
    os.makedirs(output_dir, exist_ok=True)

    # create application and connect
    app = HistApp()
    print("Connecting to IB...")
    app.connect("127.0.0.1", 7497, clientId=123)
    # small delay to allow connection handshake
    time.sleep(2)
    if app.isConnected():
        print("Connection established successfully.")
    else:
        print("Failed to establish connection.")

    # request delayed data if not subscribed to real-time
    app.reqMarketDataType(3)
    # define contracts to fetch (extend list as needed)
    mnq = create_future("MNQ", "202503")
    # add more contracts (e.g. stocks or other futures) here
    symbols = [("MNQ", mnq)]

    for symbol, contract in symbols:
        print(f"Fetching data for {symbol}...")
        data = fetch_symbol(app, contract, months_back=3, bar_size="5 mins")
        # file name includes symbol and bar size
        filename = f"{symbol}_5min_24m.csv"
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Saved {len(data)} rows to {filepath}")

    # disconnect
    app.disconnect()
    print("All done.")


if __name__ == "__main__":
    main()