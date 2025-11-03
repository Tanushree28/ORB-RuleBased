"""
Example Python script to connect to Interactive Brokers (IBKR) Trader Workstation (TWS)
and place a simple order using the ib_insync library. This script assumes you have
created an Interactive Brokers account, enabled the paper trading account, and
subscribed to the necessary market data. To use delayed data only (no market data
subscription), call reqMarketDataType(3) after connecting.

Key steps before running this script:
 1. Open and fund an IBKR account and request a paper trading account.
 2. Install IB Trader Workstation (TWS) or the IB Gateway and log in using
    your paper-trading credentials.
 3. In TWS, go to 'Configure' -> 'API' -> 'Settings' and enable "Socket
    clients". Keep the default paper-trading port (7497) or change it as
    needed. You may also want to disable read-only restrictions for testing.
 4. Install the ib_insync library into your Python environment via
    ``pip install ib_insync``. ib_insync wraps the official ibapi and
    simplifies event handling.
 5. Adjust the contract details (symbol, expiry) to the futures contract you
    want to trade.

Note: Access to real-time futures data via IBKR requires a market-data
subscription. Without a subscription, IBKR will provide delayed streaming data
(approximately 10–15 minutes behind). You can request delayed data by calling
ib.reqMarketDataType(3) after connecting.
"""
from ib_insync import IB, util, Future, MarketOrder

# Replace with your TWS/Gateway host and port. Paper trading uses port 7497 by default.
HOST = '127.0.0.1'
PORT = 7497  # Paper trading port. Use 7496 for live trading (if enabled).
CLIENT_ID = 1  # Unique ID for this API client session.

# Create the IB connection instance
ib = IB()

try:
    # Connect to TWS or IB Gateway
    print(f"Connecting to TWS on {HOST}:{PORT} ...")
    ib.connect(HOST, PORT, clientId=CLIENT_ID)

    # If you do not have a real-time market data subscription, request delayed data
    # Uncomment the next line to enable delayed streaming quotes
    # ib.reqMarketDataType(3)

    # Define a futures contract. Adjust symbol and expiration as needed.
    # Example: Micro E-mini Nasdaq-100 (MNQ) December 2024 contract (MNQZ4).
    contract = Future(
        symbol='MNQ',  # root symbol for Micro E-mini Nasdaq-100
        lastTradeDateOrContractMonth='202412',  # YYYYMM for the contract expiry
        exchange='CME',
        currency='USD'
    )

    # Resolve the contract details from IB (fetches the contract identifier)
    ib.qualifyContracts(contract)
    print(f"Qualified contract: {contract}")

    # Request market data (tick-by-tick price updates). Data will stream until canceled.
    ticker = ib.reqMktData(contract, '', False, False)
    # Allow a short time for the first price to arrive
    ib.sleep(2)
    print(f"Last price for {contract.symbol}: {ticker.last}")

    # Prepare a market order to buy 1 contract
    order = MarketOrder('BUY', 1)

    # Place the order and get a Trade object back
    trade = ib.placeOrder(contract, order)
    print(f"Placed order: {trade}")

    # Wait a few seconds for fill or status update
    ib.sleep(5)
    print(f"Order status: {trade.orderStatus.status}, Filled: {trade.orderStatus.filled}")

finally:
    # Always disconnect when done to free resources
    ib.disconnect()
    print("Disconnected from TWS")
