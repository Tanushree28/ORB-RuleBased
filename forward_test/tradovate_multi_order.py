"""

This script demonstrates how to connect to the Tradovate REST API, authenticate, look up futures contracts and submit bracket (entry,
take‑profit and stop‑loss) orders for multiple instruments.  It is tailored for the demo environment, but you can modify the ``BASE_URL``
to point at the live endpoints once you have a funded account.

**Important caveats**

* The Tradovate API uses OAuth‑like credentials.  You must create anAPI key and secret in your account’s API Access settings, then
  assign the key to your username.  See the Autoview guide for
  step‑by‑step instructions on generating API keys and copying thesecret.

* This script does **not** include your actual credentials.  Replace the ``USERNAME``, ``PASSWORD``, ``CLIENT_ID`` and ``CLIENT_SECRET``
  variables with your Tradovate login and API information.  Never commit your secrets to source control.
* Tradovate only offers futures and options.  It does **not** trade
  equities like Netflix or Amazon.  For illustration, we include
  symbols such as ``MNQ`` (Micro E‑mini NASDAQ), ``NQ`` (E‑mini NASDAQ), ``MES`` (Micro E‑mini S&P 500), ``MGC`` (Micro Gold), ``M6E`` (Euro FX
  micro), ``MCL`` (Micro crude oil) and others.  If you need to trade stocks, consider a broker like Thinkorswim or Interactive Brokers.
* Always test in the demo environment before live trading.  Marketsare volatile and trading carries risk.

The overall flow is:

1. Request an access token from the Tradovate auth endpoint.
2. Query the contract IDs for each symbol in your watch list.
3. Submit a market order with attached profit target and stop loss.

You can adapt this example to work with your existing ORB strategysignals by calling ``submit_order()`` whenever a breakout trigger
occurs.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Replace these constants with your account credentials.  The CLIENT_ID and
# CLIENT_SECRET come from your API key in Tradovate’s "API Access" page.
# USERNAME and PASSWORD are your Tradovate login credentials.  APP_ID and
# APP_VERSION are arbitrary identifiers required by the API; Tradovate
# suggests using your company or app name and version.

USERNAME = "Scorpius_Coder"
PASSWORD = "YOUR_TRADOVATE_PASSWORD"
CLIENT_ID = "YOUR_API_CLIENT_ID"
CLIENT_SECRET = "YOUR_API_CLIENT_SECRET"
APP_ID = "my-orb-app"
APP_VERSION = "1.0.0"

# Base URL for the demo environment.  For live trading, change
# "demo.tradovateapi.com" to "live.tradovateapi.com".
BASE_URL = "https://demo.tradovateapi.com/v1"

# List of instruments to trade.  Each entry includes:
#   symbol: the root symbol used by Tradovate (e.g., MNQ for the Micro
#           NASDAQ).  The script automatically fetches the front contract.
#   quantity: how many contracts to trade per order.
#   tp_ticks: take‑profit distance in ticks (one tick = one price
#             increment).  You can adjust this to match your ORB logic.
#   sl_ticks: stop‑loss distance in ticks.
INSTRUMENTS = [
    {"symbol": "MNQ", "quantity": 1, "tp_ticks": 100, "sl_ticks": 50},
    {"symbol": "NQ", "quantity": 1, "tp_ticks": 40, "sl_ticks": 20},
    {"symbol": "MES", "quantity": 1, "tp_ticks": 80, "sl_ticks": 40},
    {"symbol": "MGC", "quantity": 1, "tp_ticks": 50, "sl_ticks": 25},
    {"symbol": "M6E", "quantity": 1, "tp_ticks": 20, "sl_ticks": 10},
    {"symbol": "MCL", "quantity": 1, "tp_ticks": 30, "sl_ticks": 15},
    {"symbol": "E6", "quantity": 1, "tp_ticks": 20, "sl_ticks": 10},
    {"symbol": "GC", "quantity": 1, "tp_ticks": 10, "sl_ticks": 5},
    {"symbol": "ES", "quantity": 1, "tp_ticks": 20, "sl_ticks": 10},
    {"symbol": "YM", "quantity": 1, "tp_ticks": 20, "sl_ticks": 10},
]


# -----------------------------------------------------------------------------
# Helper data classes
# -----------------------------------------------------------------------------


@dataclass
class AuthTokens:
    access_token: str
    expiration_time: float  # Unix timestamp when the token expires
    refresh_token: Optional[str]


@dataclass
class Contract:
    id: int
    symbol: str
    full_code: str
    tick_size: float


# -----------------------------------------------------------------------------
# Tradovate API functions
# -----------------------------------------------------------------------------


def request_access_token() -> AuthTokens:
    """Authenticate with Tradovate and obtain an access token."""
    url = f"{BASE_URL}/auth/accesstokenrequest"
    payload = {
        "name": USERNAME,
        "password": PASSWORD,
        "appId": APP_ID,
        "appVersion": APP_VERSION,
        "cid": CLIENT_ID,
        "sec": CLIENT_SECRET,
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    expiration_time = data.get("expirationTime", 0) / 1000.0
    return AuthTokens(
        access_token=data["accessToken"],
        expiration_time=expiration_time,
        refresh_token=data.get("refreshToken"),
    )


def lookup_contract(symbol: str, tokens: AuthTokens) -> Contract:
    """Look up the latest front month contract for a given root symbol."""
    headers = {"Authorization": f"Bearer {tokens.access_token}"}
    product_url = f"{BASE_URL}/marketdata/contract/list"
    params = {"symbol": symbol, "hasQuotes": "true"}
    resp = requests.get(product_url, headers=headers, params=params)
    resp.raise_for_status()
    contracts = resp.json()
    if not contracts:
        raise ValueError(f"No contracts found for symbol {symbol}")
    c = contracts[0]
    return Contract(
        id=c["id"],
        symbol=symbol,
        full_code=c["code"],
        tick_size=c["tickSize"],
    )


def submit_order(
    contract: Contract,
    quantity: int,
    tp_ticks: int,
    sl_ticks: int,
    tokens: AuthTokens,
) -> Dict:
    """
    Submit a bracket order (entry + TP + SL) for the specified contract.

    This uses Tradovate’s ``orderStrategy/startOrderStrategy`` endpoint to
    create an ``EntryLimit`` order with attached ``TakeProfit`` and ``Stop``
    orders.  For market entries, set ``orderType" to "Market" and omit
    ``price".  The TP and SL offsets are defined in ticks.

    Returns the JSON response from the API.
    """
    headers = {"Authorization": f"Bearer {tokens.access_token}"}
    url = f"{BASE_URL}/orderStrategy/startOrderStrategy"
    payload = {
        "accountSpec": "MyAccount",  # replace with your account spec (e.g., "TRADOVATEDEMO")
        "symbol": contract.full_code,
        "orderQty": quantity,
        "orderType": "Market",
        "action": "Buy",  # use "Sell" for short entries
        "isBracket": True,
        "takeProfitTicks": tp_ticks,
        "stopLossTicks": sl_ticks,
    }
    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


def place_orders_for_all(instruments: List[Dict[str, any]]) -> None:
    """Authenticate once and place a bracket order for each instrument."""
    tokens = request_access_token()
    print("Authenticated successfully.  Access token acquired.")
    for instr in instruments:
        symbol = instr["symbol"]
        quantity = instr["quantity"]
        tp_ticks = instr["tp_ticks"]
        sl_ticks = instr["sl_ticks"]
        try:
            contract = lookup_contract(symbol, tokens)
            response = submit_order(contract, quantity, tp_ticks, sl_ticks, tokens)
            print(
                f"Submitted order for {symbol} (contract {contract.full_code}): {response}"
            )
        except Exception as e:
            print(f"Error submitting order for {symbol}: {e}")
        time.sleep(1.0)  # avoid API rate limits


if __name__ == "__main__":
    place_orders_for_all(INSTRUMENTS)
