"""Run the Polygon systematic sweep and build a comprehensive visual report."""

from __future__ import annotations

from polygon_systematic_backtest import (
    load_polygon_symbols,
    run_backtests,
    summarise_results,
)
from visualize_polygon_systematic import (
    REPORTS_DIR,
    build_combo_summary,
    build_symbol_summary,
)

from pathlib import Path
from typing import Dict, Tuple
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR if (BASE_DIR / "configs").exists() else BASE_DIR.parent

for path in (BASE_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _ensure_reports_dir() -> Path:
    """Ensure the Polygon reports directory exists and return it."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _combo_identifier(row: pd.Series) -> str:
    """Create a readable label for a parameter combination."""

    return (
        f"{row.interval} | {int(row.orb_duration)}m | "
        f"TP {row.tp_multiplier:.1f}x | Risk {row.risk_per_trade * 100:.0f}%"
    )


def _summarise_categories(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Polygon returns and win rates by symbol category."""

    if results_df.empty:
        return pd.DataFrame()

    summary = (
        results_df.groupby("category")
        .agg(
            avg_return=("return_pct", "mean"),
            avg_win_rate=("win_rate", "mean"),
            total_symbols=("symbol", "nunique"),
        )
        .reset_index()
    )

    summary = summary.sort_values("avg_return", ascending=False)
    return summary


def _create_overview_figure(
    results_df: pd.DataFrame,
    combo_summary: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    save_path: Path,
) -> Path:
    """Build the comprehensive multi-panel Polygon figure and save it."""

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(24, 16))

    # 1. Top parameter sets by average return
    ax1 = plt.subplot(3, 4, 1)
    top_combos = combo_summary.sort_values("avg_return", ascending=False).head(10)
    if top_combos.empty:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "No combo data", ha="center", va="center")
    else:
        labels = [_combo_identifier(row) for row in top_combos.itertuples()]
        returns = top_combos["avg_return"].tolist()
        colors = [
            "darkgreen" if val > 20 else "green" if val > 0 else "red"
            for val in returns
        ]
        bars = ax1.barh(range(len(top_combos)), returns, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(top_combos)))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("Avg Return (%)")
        ax1.set_title("Top 10 Parameter Sets", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        for idx, (bar, val) in enumerate(zip(bars, returns)):
            ax1.text(val, idx, f"{val:.1f}%", va="center", fontsize=8)

    # 2. Category performance
    ax2 = plt.subplot(3, 4, 2)
    if category_summary.empty:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No category data", ha="center", va="center")
    else:
        bars = ax2.bar(
            category_summary["category"],
            category_summary["avg_return"],
            color=["green" if v > 0 else "red" for v in category_summary["avg_return"]],
            alpha=0.7,
        )
        ax2.set_title("Average Return by Category", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Avg Return (%)")
        ax2.set_xticklabels(category_summary["category"], rotation=45, ha="right")
        ax2.axhline(0, color="black", linewidth=1)
        for bar, (_, row) in zip(bars, category_summary.iterrows()):
            height = bar.get_height()
            label = f"n={int(row.total_symbols)}"
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                label,
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )

    # 3. Win rate distribution
    ax3 = plt.subplot(3, 4, 3)
    win_rates = results_df["win_rate"].dropna() * 100
    if win_rates.empty:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "No win rate data", ha="center", va="center")
    else:
        ax3.hist(win_rates, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
        ax3.axvline(
            33.33, color="red", linestyle="--", linewidth=2, label="Breakeven for 2x TP"
        )
        ax3.axvline(
            win_rates.mean(),
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {win_rates.mean():.1f}%",
        )
        ax3.set_title("Win Rate Distribution", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Win Rate (%)")
        ax3.set_ylabel("Frequency")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    # 4. Profit factor vs returns
    ax4 = plt.subplot(3, 4, 4)
    pf = results_df["profit_factor"].astype(float)
    returns = results_df["return_pct"].astype(float)
    mask = pf.notna() & returns.notna()
    if mask.sum() == 0:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "No profit factor data", ha="center", va="center")
    else:
        scatter = ax4.scatter(
            pf[mask],
            returns[mask],
            c=returns[mask],
            cmap="RdYlGn",
            alpha=0.6,
            s=40,
        )
        ax4.axvline(1, color="black", linewidth=1)
        ax4.axhline(0, color="black", linewidth=1)
        ax4.set_title("Profit Factor vs Return", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Profit Factor")
        ax4.set_ylabel("Return (%)")
        fig.colorbar(scatter, ax=ax4, fraction=0.046, pad=0.04, label="Return (%)")
        ax4.grid(True, alpha=0.3)

    # 5. Average return by symbol
    ax5 = plt.subplot(3, 4, (5, 8))
    if symbol_summary.empty:
        ax5.axis("off")
        ax5.text(0.5, 0.5, "No symbol summary", ha="center", va="center")
    else:
        sorted_symbols = symbol_summary.sort_values("avg_return", ascending=False)
        colors = [
            "darkgreen" if v > 20 else "green" if v > 0 else "red"
            for v in sorted_symbols["avg_return"]
        ]
        ax5.bar(
            range(len(sorted_symbols)),
            sorted_symbols["avg_return"],
            color=colors,
            alpha=0.7,
        )
        ax5.set_xticks(range(len(sorted_symbols)))
        ax5.set_xticklabels(sorted_symbols["symbol"], rotation=90, fontsize=8)
        ax5.axhline(0, color="black", linewidth=1)
        ax5.set_title("Average Return by Symbol", fontsize=12, fontweight="bold")
        ax5.set_ylabel("Avg Return (%)")
        ax5.grid(True, alpha=0.3)

    # 6. Trade count distribution
    ax6 = plt.subplot(3, 4, 9)
    trades = results_df["total_trades"].dropna()
    if trades.empty:
        ax6.axis("off")
        ax6.text(0.5, 0.5, "No trade data", ha="center", va="center")
    else:
        ax6.hist(trades, bins=15, edgecolor="black", alpha=0.7, color="orange")
        ax6.set_title("Trade Count Distribution", fontsize=12, fontweight="bold")
        ax6.set_xlabel("Trades per Scenario")
        ax6.set_ylabel("Frequency")
        ax6.grid(True, alpha=0.3)

    # 7. Positive ratio vs avg return by interval/risk buckets
    ax7 = plt.subplot(3, 4, 10)
    if combo_summary.empty:
        ax7.axis("off")
        ax7.text(0.5, 0.5, "No combo data", ha="center", va="center")
    else:
        grouped: Dict[Tuple[str, float], pd.DataFrame] = {}
        for (interval, risk), sub_df in combo_summary.groupby(
            ["interval", "risk_per_trade"]
        ):
            grouped[(interval, risk)] = sub_df

        for (interval, risk), sub_df in grouped.items():
            ax7.scatter(
                sub_df["avg_return"],
                sub_df["positive_ratio"] * 100,
                label=f"{interval} | {int(risk * 100)}%",
                alpha=0.6,
                s=50,
            )

        ax7.axvline(0, color="black", linewidth=1, alpha=0.5)
        ax7.axhline(60, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax7.set_title("Combination Quality", fontsize=12, fontweight="bold")
        ax7.set_xlabel("Avg Return (%)")
        ax7.set_ylabel("Positive Symbols (%)")
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)

    # 8. Summary text panel
    ax8 = plt.subplot(3, 4, (11, 12))
    ax8.axis("off")

    total_runs = len(results_df)
    positive_runs = int((results_df["return_pct"] > 0).sum())
    avg_return_all = results_df["return_pct"].mean()
    avg_win_rate_all = results_df["win_rate"].mean() * 100
    total_trades_all = int(results_df["total_trades"].fillna(0).sum())

    best_combo_text = "N/A"
    if not top_combos.empty:
        best = top_combos.iloc[0]
        best_combo_text = _combo_identifier(best)

    best_symbol_text = "N/A"
    if not symbol_summary.empty:
        best_symbol = symbol_summary.sort_values("avg_return", ascending=False).iloc[0]
        best_symbol_text = f"{best_symbol['symbol']} ({best_symbol['avg_return']:.1f}%)"

    best_category_text = "N/A"
    if not category_summary.empty:
        best_category = category_summary.iloc[0]
        best_category_text = (
            f"{best_category['category']} ({best_category['avg_return']:.1f}% | "
            f"n={int(best_category['total_symbols'])})"
        )

    summary = f"""
    POLYGON SYSTEMATIC ORB SUMMARY
    {"=" * 45}

    Total Runs: {total_runs}
    Positive Runs: {positive_runs} ({
        (positive_runs / total_runs * 100) if total_runs else 0:.1f}%)

    Average Return (all runs): {avg_return_all:.2f}%
    Average Win Rate: {avg_win_rate_all:.1f}%
    Total Trades Simulated: {total_trades_all}

    Top Parameter Set: {best_combo_text}
    Top Symbol (avg return): {best_symbol_text}
    Leading Category: {best_category_text}

    Robust Combo Count (avg_return>0 & positive_ratio>=0.6): {
        int(
            (
                (combo_summary["avg_return"] > 0)
                & (combo_summary["positive_ratio"] >= 0.6)
            ).sum()
        )
    }
    """

    ax8.text(
        0.05,
        0.95,
        summary,
        transform=ax8.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(
        "Polygon Systematic ORB Strategy – Comprehensive Overview",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def main() -> None:
    """Execute the Polygon systematic sweep and render a comprehensive dashboard."""

    print("=" * 94)
    print("POLYGON SYSTEMATIC ORB BACKTEST – COMPREHENSIVE REPORT")
    print("=" * 94)

    symbols = load_polygon_symbols()
    if not symbols:
        print("No Polygon symbols configured – exiting.")
        return

    results_df = run_backtests(symbols)
    if results_df.empty:
        print("No results generated – exiting without report.")
        return

    summarise_results(results_df)

    combo_summary = build_combo_summary(results_df)
    symbol_summary = build_symbol_summary(results_df)
    category_summary = _summarise_categories(results_df)

    reports_dir = _ensure_reports_dir()
    chart_path = reports_dir / "polygon_systematic_comprehensive_overview.png"
    _create_overview_figure(
        results_df,
        combo_summary,
        symbol_summary,
        category_summary,
        chart_path,
    )

    print(f"✓ Polygon comprehensive systematic report saved to {chart_path}")


if __name__ == "__main__":
    main()
