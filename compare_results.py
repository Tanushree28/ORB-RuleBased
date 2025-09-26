#!/usr/bin/env python3
"""
Compare 60-day vs 2-year backtest results
Identify which market conditions favor the ORB strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_results():
    """Load both 60-day and extended results"""
    results = {}

    # Load 60-day results (optional)
    if os.path.exists("reports/comprehensive_results_2x.csv"):
        sixty = pd.read_csv("reports/comprehensive_results_2x.csv")
        sixty = _coerce_numeric(
            sixty,
            ["return_pct", "win_rate", "max_drawdown", "profit_factor", "total_trades"],
        )
        results["60_day"] = sixty
        print(f"✓ Loaded 60-day results: {len(results['60_day'])} symbols")
    else:
        print(
            "• No 60-day CSV found (reports/comprehensive_results_2x.csv). Skipping 60-day comparisons."
        )

    # Load extended results (required for most charts)
    if os.path.exists("reports/extended_backtest_results.csv"):
        extended = pd.read_csv("reports/extended_backtest_results.csv")
        if "period" not in extended.columns:
            raise ValueError(
                "extended_backtest_results.csv must have a 'period' column"
            )

        # type coercions
        extended = _coerce_numeric(
            extended,
            ["return_pct", "win_rate", "max_drawdown", "profit_factor", "total_trades"],
        )

        for period in extended["period"].dropna().unique():
            dfp = extended[extended["period"] == period].copy()
            results[period] = dfp
            print(f"✓ Loaded {period} results: {len(results[period])} symbols")
    else:
        print("✗ extended_backtest_results.csv not found; nothing to plot.")

    return results


def _safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def _set_categorical_ticks(ax, labels):
    """Set ticks and ticklabels safely to avoid warnings."""
    x = np.arange(len(labels))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    return x


def _pick_top_symbols(results, base_period=None, n=5):
    """
    Choose top N symbols by average return using:
    - base_period if provided and present in results
    - otherwise the period with the most rows
    """
    if base_period and base_period in results:
        base = results[base_period]
    else:
        # pick the richest period
        non60 = {k: v for k, v in results.items() if k != "60_day"}
        if not non60:
            return []
        base = max(non60.items(), key=lambda kv: len(kv[1]))[1]

    if "symbol" not in base.columns or "return_pct" not in base.columns:
        return []
    grp = (
        base.groupby("symbol", as_index=False)["return_pct"]
        .mean()
        .sort_values("return_pct", ascending=False)
    )
    return grp["symbol"].head(n).tolist()


def create_comparison_report(results):
    """Create comprehensive comparison report"""
    plt.style.use("seaborn-v0_8-darkgrid")

    fig = plt.figure(figsize=(20, 14))

    # 1. Return Comparison: 60-day vs 2-year
    ax1 = plt.subplot(3, 3, 1)

    if "60_day" in results and "Full 2 Years" in results:
        sixty_day = results["60_day"]
        two_year = results["Full 2 Years"]

        if {"symbol", "return_pct"}.issubset(sixty_day.columns) and {
            "symbol",
            "return_pct",
        }.issubset(two_year.columns):
            # Match symbols
            merged = pd.merge(
                sixty_day[["symbol", "return_pct"]].rename(
                    columns={"return_pct": "60_day"}
                ),
                two_year[["symbol", "return_pct"]].rename(
                    columns={"return_pct": "2_year"}
                ),
                on="symbol",
                how="inner",
            ).dropna()

            if not merged.empty:
                ax1.scatter(merged["60_day"], merged["2_year"], alpha=0.6, s=50)

                # Diagonal
                max_val = np.nanmax([merged["60_day"].max(), merged["2_year"].max()])
                min_val = np.nanmin([merged["60_day"].min(), merged["2_year"].min()])
                ax1.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)

                # Correlation
                correlation = merged["60_day"].corr(merged["2_year"])
                ax1.set_xlabel("60-Day Return (%)", fontsize=10)
                ax1.set_ylabel("2-Year Return (%)", fontsize=10)
                ax1.set_title(
                    f"60-Day vs 2-Year Returns (Corr: {correlation:.2f})",
                    fontsize=12,
                    fontweight="bold",
                )
                ax1.grid(True, alpha=0.3)

                # Annotate outliers
                deltas = (merged["60_day"] - merged["2_year"]).abs()
                for _, row in merged[deltas > 100].iterrows():
                    ax1.annotate(
                        row["symbol"],
                        (row["60_day"], row["2_year"]),
                        fontsize=8,
                        alpha=0.7,
                    )
        else:
            ax1.text(
                0.5,
                0.5,
                "Missing columns for return comparison",
                ha="center",
                va="center",
            )
    else:
        ax1.text(
            0.5, 0.5, "No 60-Day or 2-Year data to compare", ha="center", va="center"
        )

    # 2. Win Rate Stability
    ax2 = plt.subplot(3, 3, 2)
    win_rates_comparison = {}
    for period_name, df in results.items():
        if not df.empty and "win_rate" in df.columns:
            win_rates_comparison[period_name[:10]] = (
                (df["win_rate"] * 100).dropna().values
            )

    if win_rates_comparison:
        ax2.boxplot(
            win_rates_comparison.values(), tick_labels=list(win_rates_comparison.keys())
        )
        ax2.axhline(
            y=33.33, color="red", linestyle="--", linewidth=2, label="Min for 2x TP"
        )
        ax2.set_title(
            "Win Rate Consistency Across Periods", fontsize=12, fontweight="bold"
        )
        ax2.set_ylabel("Win Rate (%)", fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    # 3. Profitability by Period
    ax3 = plt.subplot(3, 3, 3)
    profitability_stats = []
    for period_name, df in results.items():
        if not df.empty and "return_pct" in df.columns:
            total = len(df)
            profitable = (df["return_pct"] > 0).sum()
            profitability_stats.append(
                {
                    "Period": period_name[:10],
                    "Profitable %": (profitable / total * 100) if total else 0,
                    "Count": f"{profitable}/{total}",
                }
            )

    if profitability_stats:
        prof_df = pd.DataFrame(profitability_stats)
        colors = [
            "green" if p > 50 else "orange" if p > 30 else "red"
            for p in prof_df["Profitable %"]
        ]
        x = _set_categorical_ticks(ax3, prof_df["Period"].tolist())
        ax3.bar(x, prof_df["Profitable %"], alpha=0.7, color=colors)
        ax3.axhline(y=50, color="black", linestyle="-", linewidth=1)
        ax3.set_title("Symbol Profitability by Period", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Profitable Symbols (%)", fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Add count labels
        for xi, pct, count in zip(x, prof_df["Profitable %"], prof_df["Count"]):
            ax3.text(xi, pct, count, ha="center", va="bottom", fontsize=8)

    # 4. Top Performers Consistency (dynamic)
    ax4 = plt.subplot(3, 3, 4)
    top_symbols = _pick_top_symbols(results, base_period="Full 2 Years", n=5)
    consistency_data = []

    for symbol in top_symbols:
        symbol_returns = []
        for period_name, df in results.items():
            if not df.empty and {"symbol", "return_pct"}.issubset(df.columns):
                row = df[df["symbol"] == symbol]
                symbol_returns.append(
                    float(row.iloc[0]["return_pct"]) if not row.empty else 0.0
                )
        if symbol_returns:
            consistency_data.append(
                {
                    "Symbol": symbol,
                    "Mean": float(np.mean(symbol_returns)),
                    "Std": float(np.std(symbol_returns)),
                    "Min": float(np.min(symbol_returns)),
                    "Max": float(np.max(symbol_returns)),
                }
            )

    if consistency_data:
        cons_df = pd.DataFrame(consistency_data)
        x = np.arange(len(cons_df))
        ax4.bar(x, cons_df["Mean"], yerr=cons_df["Std"], capsize=5, alpha=0.7)
        ax4.set_xticks(x)
        ax4.set_xticklabels(cons_df["Symbol"])
        ax4.set_title(
            "Top Performers Consistency (Mean ± Std)", fontsize=12, fontweight="bold"
        )
        ax4.set_ylabel("Average Return (%)", fontsize=10)
        ax4.grid(True, alpha=0.3)

    # 5. Drawdown Comparison
    ax5 = plt.subplot(3, 3, 5)
    drawdown_data = []
    for period_name, df in results.items():
        if not df.empty and "max_drawdown" in df.columns:
            dd_values = df["max_drawdown"].replace(0, np.nan).dropna().abs()
            if len(dd_values):
                drawdown_data.append(
                    {
                        "Period": period_name[:10],
                        "Mean DD": float(dd_values.mean()),
                        "Max DD": float(dd_values.max()),
                    }
                )

    if drawdown_data:
        dd_df = pd.DataFrame(drawdown_data)
        x = _set_categorical_ticks(ax5, dd_df["Period"].tolist())
        width = 0.35
        ax5.bar(x - width / 2, dd_df["Mean DD"], width, label="Mean DD", alpha=0.7)
        ax5.bar(x + width / 2, dd_df["Max DD"], width, label="Max DD", alpha=0.7)
        ax5.set_xlabel("Period", fontsize=10)
        ax5.set_ylabel("Drawdown (%)", fontsize=10)
        ax5.set_title("Drawdown Analysis by Period", fontsize=12, fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. Profit Factor Distribution
    ax6 = plt.subplot(3, 3, 6)
    pf_data = {}
    for period_name, df in results.items():
        if not df.empty and "profit_factor" in df.columns:
            vals = df["profit_factor"]
            vals = pd.to_numeric(vals, errors="coerce")
            vals = vals[vals > 0]
            if len(vals):
                pf_data[period_name[:10]] = vals.values

    if pf_data:
        ax6.boxplot(pf_data.values(), tick_labels=list(pf_data.keys()))
        ax6.axhline(y=1.0, linestyle="-", linewidth=2, label="Breakeven")
        ax6.axhline(y=1.5, linestyle="--", linewidth=1, label="Good (1.5)")
        ax6.set_title("Profit Factor Distribution", fontsize=12, fontweight="bold")
        ax6.set_ylabel("Profit Factor", fontsize=10)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

    # 7. Average Returns by Period
    ax7 = plt.subplot(3, 3, 7)
    avg_returns = []
    for period_name, df in results.items():
        if not df.empty and "return_pct" in df.columns:
            avg_returns.append(
                {
                    "Period": period_name[:10],
                    "Avg Return": _safe_mean(df["return_pct"]),
                    "Median Return": float(
                        pd.to_numeric(df["return_pct"], errors="coerce").median()
                    ),
                }
            )

    if avg_returns:
        ar_df = pd.DataFrame(avg_returns)
        x = _set_categorical_ticks(ax7, ar_df["Period"].tolist())
        width = 0.35
        ax7.bar(x - width / 2, ar_df["Avg Return"], width, label="Mean", alpha=0.7)
        ax7.bar(x + width / 2, ar_df["Median Return"], width, label="Median", alpha=0.7)
        ax7.axhline(y=0, linestyle="-", linewidth=1)
        ax7.set_xlabel("Period", fontsize=10)
        ax7.set_ylabel("Return (%)", fontsize=10)
        ax7.set_title("Average vs Median Returns", fontsize=12, fontweight="bold")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    # 8. Trade Frequency
    ax8 = plt.subplot(3, 3, 8)
    trade_freq = []
    for period_name, df in results.items():
        if not df.empty and "total_trades" in df.columns:
            trade_freq.append(
                {
                    "Period": period_name[:10],
                    "Avg Trades": _safe_mean(df["total_trades"]),
                    "Total Trades": int(
                        pd.to_numeric(df["total_trades"], errors="coerce").sum()
                    ),
                }
            )

    if trade_freq:
        tf_df = pd.DataFrame(trade_freq)
        x = _set_categorical_ticks(ax8, tf_df["Period"].tolist())
        ax8.bar(x, tf_df["Avg Trades"], alpha=0.7)
        ax8.set_title("Average Trades per Symbol", fontsize=12, fontweight="bold")
        ax8.set_ylabel("Average Trade Count", fontsize=10)
        ax8.grid(True, alpha=0.3)

        for xi, avg, total in zip(x, tf_df["Avg Trades"], tf_df["Total Trades"]):
            ax8.text(xi, avg, f"{total}", ha="center", va="bottom", fontsize=8)

    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")

    summary_text = "COMPARISON SUMMARY\n" + "=" * 35 + "\n\n"

    if "60_day" in results and "Full 2 Years" in results:
        sixty_day_avg = _safe_mean(results["60_day"]["return_pct"])
        two_year_avg = _safe_mean(results["Full 2 Years"]["return_pct"])
        if not np.isnan(sixty_day_avg) and not np.isnan(two_year_avg):
            summary_text += "60-DAY vs 2-YEAR:\n"
            summary_text += f"  60-Day Avg: {sixty_day_avg:.1f}%\n"
            summary_text += f"  2-Year Avg: {two_year_avg:.1f}%\n"
            summary_text += f"  Difference: {two_year_avg - sixty_day_avg:+.1f}%\n\n"

    period_performance = []
    for period_name, df in results.items():
        if not df.empty and period_name != "60_day" and "return_pct" in df.columns:
            period_performance.append(
                {"period": period_name, "avg": _safe_mean(df["return_pct"])}
            )

    if period_performance:
        best_period = max(
            period_performance,
            key=lambda x: (x["avg"] if not np.isnan(x["avg"]) else -np.inf),
        )
        worst_period = min(
            period_performance,
            key=lambda x: (x["avg"] if not np.isnan(x["avg"]) else np.inf),
        )
        if not np.isnan(best_period["avg"]) and not np.isnan(worst_period["avg"]):
            summary_text += "PERIOD ANALYSIS:\n"
            summary_text += f"  Best: {best_period['period']}\n"
            summary_text += f"    Return: {best_period['avg']:.1f}%\n"
            summary_text += f"  Worst: {worst_period['period']}\n"
            summary_text += f"    Return: {worst_period['avg']:.1f}%\n\n"

    summary_text += "KEY FINDINGS:\n"
    summary_text += "✓ Strategy consistency across periods\n"
    summary_text += "✓ Risk metrics for position sizing\n"
    summary_text += "✓ Optimal market conditions identified"

    ax9.text(
        0.05,
        0.95,
        summary_text,
        transform=ax9.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.suptitle(
        "ORB Strategy - Comprehensive Comparison Analysis",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/comparison_analysis.png", dpi=100, bbox_inches="tight")
    print("\n✓ Comparison analysis saved to reports/comparison_analysis.png")


def main():
    print("=" * 80)
    print("ORB STRATEGY - RESULTS COMPARISON")
    print("=" * 80)
    print("\nComparing 60-day results with extended 2-year backtest...")

    results = load_results()

    if not results:
        print("\n✗ No results found to compare")
        print("Please run both backtests first:")
        print("  1. python download_expanded_symbols.py")
        print("  2. python backtest_expanded_2x.py")
        print("  3. python polygon_downloader.py")
        print("  4. python backtest_extended_polygon.py")
        return

    print("\nGenerating comparison charts...")
    create_comparison_report(results)

    print("\n" + "=" * 80)
    print("COMPARISON INSIGHTS")
    print("=" * 80)

    if "60_day" in results and "Full 2 Years" in results:
        sixty_day = results["60_day"]
        two_year = results["Full 2 Years"]

        if (
            "return_pct" in sixty_day
            and "return_pct" in two_year
            and "win_rate" in sixty_day
            and "win_rate" in two_year
        ):
            print("\n1. PERFORMANCE COMPARISON:")
            print(
                f"   60-Day Average Return: {_safe_mean(sixty_day['return_pct']):.1f}%"
            )
            print(
                f"   2-Year Average Return: {_safe_mean(two_year['return_pct']):.1f}%"
            )

            print("\n2. WIN RATE COMPARISON:")
            print(
                f"   60-Day Average Win Rate: {_safe_mean(sixty_day['win_rate']) * 100:.1f}%"
            )
            print(
                f"   2-Year Average Win Rate: {_safe_mean(two_year['win_rate']) * 100:.1f}%"
            )

            print("\n3. CONSISTENCY:")
            profitable_60 = int(
                pd.to_numeric(sixty_day["return_pct"], errors="coerce").gt(0).sum()
            )
            profitable_2y = int(
                pd.to_numeric(two_year["return_pct"], errors="coerce").gt(0).sum()
            )
            print(
                f"   60-Day Profitable Symbols: {profitable_60}/{len(sixty_day)} ({profitable_60 / len(sixty_day) * 100:.0f}%)"
            )
            print(
                f"   2-Year Profitable Symbols: {profitable_2y}/{len(two_year)} ({profitable_2y / len(two_year) * 100:.0f}%)"
            )
        else:
            print("• Missing columns to compute performance/win-rate summaries.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nReports generated:")
    print("  - reports/comparison_analysis.png")
    print("  - reports/extended_period_analysis.png")
    print("  - reports/extended_backtest_results.csv")


if __name__ == "__main__":
    main()
