"""Generate summary tables for the ORB parameter plan.

The script focuses on the combinations requested in the current plan:
- Opening range durations of 5 minutes and 15 minutes
- Take-profit multipliers of 0.5x, 1.0x, and 2.0x
- Risk per trade levels of 1% and 2%

The raw data already contains the backtest sweep for these parameters.
We aggregate the results to identify which configurations are broadly
robust and which symbols respond best to the plan.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_PATH = Path("reports/systematic_backtest_results.csv")
OUTPUT_DIR = Path("reports")
COMBO_SUMMARY_PATH = OUTPUT_DIR / "plan_parameter_combo_summary.csv"
SYMBOL_BEST_PATH = OUTPUT_DIR / "plan_parameter_symbol_best.csv"
DURATION_COMPARISON_PATH = OUTPUT_DIR / "plan_parameter_duration_comparison.csv"


def load_results(path: Path) -> pd.DataFrame:
    """Load the backtest results and enforce consistent dtypes."""

    df = pd.read_csv(path)
    df["orb_duration"] = df["orb_duration"].astype(int)
    df["tp_multiplier"] = df["tp_multiplier"].astype(float)
    df["risk_per_trade"] = df["risk_per_trade"].astype(float)
    return df


def build_combo_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics for each parameter combination."""

    combo_fields = ["interval", "orb_duration", "tp_multiplier", "risk_per_trade"]
    summary = (
        df.groupby(combo_fields)
        .agg(
            symbols_tested=("symbol", "nunique"),
            pct_positive=("return_pct", lambda x: float((x > 0).mean())),
            avg_return=("return_pct", "mean"),
            median_return=("return_pct", "median"),
            avg_win_rate=("win_rate", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
        )
        .reset_index()
        .sort_values("avg_return", ascending=False)
    )
    return summary


def build_symbol_best(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the best-performing combination for each symbol."""

    order = [
        "symbol",
        "category",
        "interval",
        "orb_duration",
        "tp_multiplier",
        "risk_per_trade",
        "return_pct",
        "win_rate",
        "profit_factor",
        "total_trades",
        "max_drawdown",
    ]

    idx = df.groupby("symbol")["return_pct"].idxmax()
    best = (
        df.loc[idx, order]
        .sort_values("return_pct", ascending=False)
        .reset_index(drop=True)
    )
    return best


def build_duration_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how each symbol performs across ORB durations."""

    # Best setup per symbol per ORB duration regardless of interval.
    idx = df.groupby(["symbol", "orb_duration"])["return_pct"].idxmax()
    best_per_duration = (
        df.loc[
            idx,
            [
                "symbol",
                "orb_duration",
                "interval",
                "tp_multiplier",
                "risk_per_trade",
                "return_pct",
                "win_rate",
                "profit_factor",
            ],
        ]
        .sort_values(["return_pct"], ascending=False)
        .reset_index(drop=True)
    )

    # Pivot to show the spread between 5-minute and 15-minute ranges.
    pivot = (
        best_per_duration.pivot(
            index="symbol",
            columns="orb_duration",
            values="return_pct",
        )
        .rename(columns={5: "return_pct_orb5", 15: "return_pct_orb15"})
        .reset_index()
    )
    pivot["return_pct_diff_5_minus_15"] = pivot.get("return_pct_orb5", 0) - pivot.get(
        "return_pct_orb15", 0
    )

    merged = best_per_duration.merge(pivot, on="symbol", how="left")
    return merged


def main() -> None:
    df = load_results(RESULTS_PATH)

    combo_summary = build_combo_summary(df)
    symbol_best = build_symbol_best(df)
    duration_comparison = build_duration_comparison(df)

    combo_summary.to_csv(COMBO_SUMMARY_PATH, index=False)
    symbol_best.to_csv(SYMBOL_BEST_PATH, index=False)
    duration_comparison.to_csv(DURATION_COMPARISON_PATH, index=False)


if __name__ == "__main__":
    main()
