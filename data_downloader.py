#!/usr/bin/env python3
"""
Data Downloader for ORB Strategy
Downloads intraday price data (today-only) for various instruments
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import yaml
import os
from tqdm import tqdm
import time


class DataDownloader:
    def __init__(self, config_path="configs/config.yaml"):
        """Initialize data downloader with configuration"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

        # Extract all symbols from config
        self.symbols = self._extract_symbols()

    def _extract_symbols(self):
        """Extract all trading symbols from config"""
        symbols = []
        for category in ["futures", "forex", "commodities", "stocks"]:
            for item in self.config.get("symbols", {}).get(category, []):
                # Some entries (e.g., stocks) may only have symbol & name
                symbols.append(
                    {
                        "symbol": item["symbol"],
                        "name": item.get("name", item["symbol"]),
                        "category": category,
                    }
                )
        return symbols

    def _today_et_range(self):
        """Return (start_date_str, end_date_str) for 'today in ET'"""
        now_et = datetime.now(ZoneInfo("America/New_York"))
        today_et = now_et.date()
        start = today_et.strftime("%Y-%m-%d")
        # Yahoo treats end as exclusive; +1 day to include today's bars
        end = (today_et + timedelta(days=1)).strftime("%Y-%m-%d")
        return start, end

    def download_data(self, symbol, interval, start_date, end_date):
        """Download historical data for a single symbol"""
        try:
            data = yf.Ticker(symbol).history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True,  # ok; analyzer filters regular session anyway
            )

            if data.empty:
                print(f"Warning: No data available for {symbol}")
                return None

            data = data.reset_index()

            # Keep required columns; yfinance uses "Datetime" for intraday
            if "Datetime" not in data.columns:
                # Fall back to "Date" (rare for intraday, but safe)
                data.rename(columns={"Date": "Datetime"}, inplace=True)

            expected_cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
            for col in expected_cols:
                if col not in data.columns:
                    data[col] = np.nan
            data = data[expected_cols]
            data["Symbol"] = symbol
            return data

        except Exception as e:
            print(f"Error downloading {symbol}: {str(e)}")
            return None

    def save_data(self, data, symbol, interval):
        """Save downloaded data to CSV file"""
        if data is None or data.empty:
            return False

        safe_symbol = symbol.replace("=", "_").replace("^", "_")
        filename = f"{self.data_dir}/{safe_symbol}_{interval}.csv"
        data.to_csv(filename, index=False)
        print(f"Saved {len(data)} records to {filename}")
        return True

    def download_all(self):
        """Download data for all configured symbols and intervals (today-only)"""
        data_cfg = self.config.get("data", {})
        intervals = data_cfg.get("intervals", ["5m"])
        # Force 5m for analyzer; but respect config if you want others too
        if "5m" not in intervals:
            intervals = ["5m"] + list(intervals)

        # Today-only in ET:
        start_date, end_date = self._today_et_range()

        print(f"Downloading data for TODAY (ET): {start_date} to {end_date}")
        print(f"Intervals: {intervals}")
        print(f"Total symbols: {len(self.symbols)}")
        print("-" * 50)

        successful = 0
        failed = []
        total_downloads = len(self.symbols) * len(intervals)
        pbar = tqdm(total=total_downloads, desc="Downloading")

        for symbol_info in self.symbols:
            symbol = symbol_info["symbol"]
            name = symbol_info["name"]

            for interval in intervals:
                pbar.set_description(f"Downloading {name} ({symbol}) - {interval}")

                data = self.download_data(symbol, interval, start_date, end_date)

                if self.save_data(data, symbol, interval):
                    successful += 1
                else:
                    failed.append(f"{symbol}_{interval}")

                time.sleep(0.4)  # gentle rate limiting
                pbar.update(1)

        pbar.close()

        print("\n" + "=" * 50)
        print("Download Summary:")
        print(f"Successful: {successful}/{total_downloads}")
        if failed:
            print(f"Failed downloads: {', '.join(failed)}")

        return successful, failed

    def verify_data(self):
        """Verify downloaded data files (today-only check)"""
        print("\nVerifying downloaded data...")
        for symbol_info in self.symbols:
            symbol = symbol_info["symbol"]
            safe_symbol = symbol.replace("=", "_").replace("^", "_")
            for interval in self.config.get("data", {}).get("intervals", ["5m"]):
                filename = f"{self.data_dir}/{safe_symbol}_{interval}.csv"
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    print(f"✓ {symbol} ({interval}): {len(df)} records")
                else:
                    print(f"✗ {symbol} ({interval}): File not found")

    def get_data(self, symbol, interval):
        """Load data for a specific symbol and interval"""
        safe_symbol = symbol.replace("=", "_").replace("^", "_")
        filename = f"{self.data_dir}/{safe_symbol}_{interval}.csv"
        if not os.path.exists(filename):
            print(f"Data file not found: {filename}")
            return None
        df = pd.read_csv(filename)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
        return df


def main():
    print("=" * 50)
    print("ORB Strategy Data Downloader (TODAY-ONLY)")
    print("=" * 50)

    downloader = DataDownloader()
    successful, failed = downloader.download_all()
    downloader.verify_data()

    print("\nData download complete!")
    print("\nTesting data loading...")
    test_data = downloader.get_data("AAPL", "5m")
    if test_data is not None and len(test_data) > 0:
        print(f"Successfully loaded AAPL 5m data: {len(test_data)} records")
        print(f"Date range: {test_data.index[0]} to {test_data.index[-1]}")


if __name__ == "__main__":
    main()
