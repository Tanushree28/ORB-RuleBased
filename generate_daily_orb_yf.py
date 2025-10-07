"""Daily Opening Range Breakout analyzer using Yahoo Finance data (TODAY-ONLY)."""

from __future__ import annotations

import os
import time
import itertools
import math
from dataclasses import dataclass, asdict
from datetime import datetime, time
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path("configs/config.yaml")
DATA_DIR = Path("data")
REPORT_ROOT = Path("reports/daily_orb_yf")

# Opening range window (US/Eastern)
DEFAULT_OPENING_WINDOWS = {
    "15m": (time(9, 30), time(9, 45)),
}

# Start trading on/after the bar that opens at 9:45 ET
DEFAULT_TRADING_WINDOWS = {
    "15m": time(9, 45),
}

SESSION_END = time(16, 0)


# -------------------------- Safe file writes (Windows) --------------------------


def safe_write_csv(
    df: pd.DataFrame, path: Path, attempts: int = 5, wait_s: float = 0.5
):
    """Write CSV with retry and atomic replace to avoid Windows file locks."""
    path = Path(path)
    for i in range(attempts):
        # keep the real suffix LAST (e.g., foo.tmp.123.0.csv)
        tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{i}{path.suffix}")
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)  # atomic move
            return path
        except PermissionError:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(wait_s)
    # fallback: unique filename
    alt = path.with_name(
        f"{path.stem}_{datetime.now().strftime('%H%M%S')}{path.suffix}"
    )
    df.to_csv(alt, index=False)
    print(f"[WARN] Could not overwrite {path}. Wrote {alt} instead.")
    return alt


def safe_write_text(path: Path, text: str, attempts: int = 5, wait_s: float = 0.5):
    """Write text with retry + atomic replace (Windows-friendly)."""
    path = Path(path)
    for i in range(attempts):
        tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{i}{path.suffix}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
            return path
        except PermissionError:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(wait_s)
    alt = path.with_name(
        f"{path.stem}_{datetime.now().strftime('%H%M%S')}{path.suffix}"
    )
    with open(alt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[WARN] Could not overwrite {path}. Wrote {alt} instead.")
    return alt


def safe_savefig(fig, path: Path, attempts: int = 5, wait_s: float = 0.5, **kwargs):
    """Save figure with retry and atomic replace; keep correct extension last."""
    path = Path(path)
    for i in range(attempts):
        # keep the real suffix LAST (e.g., chart.tmp.123.0.png)
        tmp = path.with_name(f"{path.stem}.tmp.{os.getpid()}.{i}{path.suffix}")
        try:
            fig.savefig(str(tmp), **kwargs)  # pass as str for safety
            os.replace(tmp, path)
            return path
        except PermissionError:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(wait_s)
    alt = path.with_name(
        f"{path.stem}_{datetime.now().strftime('%H%M%S')}{path.suffix}"
    )
    fig.savefig(str(alt), **kwargs)
    print(f"[WARN] Could not overwrite {path}. Wrote {alt} instead.")
    return alt


# --------------------------------- Data model ----------------------------------


@dataclass
class TradeResult:
    date: pd.Timestamp
    symbol: str
    opening_range: str
    risk_pct: float
    tp_multiplier: float
    direction: str
    triggered: bool
    outcome: str
    pnl: float
    return_pct: float
    orb_high: float
    orb_low: float
    orb_range: float

    @property
    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------- Utilities -----------------------------------


def load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def safe_symbol(symbol: str) -> str:
    return symbol.replace("=", "_").replace("^", "_")


def _parse_to_et_naive(series: pd.Series) -> pd.Series:
    """
    Robustly parse timestamps from CSV into ET (naive) for time-of-day slicing.
    - If strings are tz-aware (UTC or otherwise), convert to US/Eastern then drop tz.
    - If strings are naive, assume they are US/Eastern and keep as-is.
    """
    dt_utc = pd.to_datetime(series, errors="coerce", utc=True)
    if dt_utc.notna().sum() >= int(0.8 * len(series)):  # mostly tz-aware
        return dt_utc.dt.tz_convert("US/Eastern").dt.tz_localize(None)

    dt_naive = pd.to_datetime(series, errors="coerce")
    return dt_naive


def load_bars(symbol: str, interval: str = "5m") -> pd.DataFrame:
    csv_path = DATA_DIR / f"{safe_symbol(symbol)}_{interval}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing cached data for {symbol} ({interval}) at {csv_path}"
        )

    frame = pd.read_csv(csv_path)
    if "Datetime" not in frame.columns:
        raise ValueError(f"Expected 'Datetime' column in {csv_path}")

    frame["Datetime"] = _parse_to_et_naive(frame["Datetime"])
    frame = frame.dropna(subset=["Datetime"]).copy()
    frame = frame.set_index("Datetime").sort_index()

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(frame.columns)
    for col in missing:
        frame[col] = np.nan  # tolerate missing cols
    return frame[list(required)]


def slice_session(frame: pd.DataFrame, session_date: pd.Timestamp) -> pd.DataFrame:
    """Extract a single regular-hours session in ET (9:30–16:00)."""
    mask = frame.index.normalize() == session_date.normalize()
    session = frame.loc[mask].copy()
    if session.empty:
        return session
    open_start = DEFAULT_OPENING_WINDOWS["15m"][0]
    mask_time = (session.index.time >= open_start) & (session.index.time <= SESSION_END)
    return session.loc[mask_time]


def compute_opening_range(
    session: pd.DataFrame,
    window_key: str,
) -> Tuple[float, float, float, pd.DataFrame]:
    start_time, end_time = DEFAULT_OPENING_WINDOWS[window_key]
    window_mask = (session.index.time >= start_time) & (session.index.time < end_time)
    or_slice = session.loc[window_mask]

    if or_slice.empty:
        raise ValueError("Opening range slice is empty")

    orb_high = float(or_slice["High"].max())
    orb_low = float(or_slice["Low"].min())
    orb_range = orb_high - orb_low
    if not math.isfinite(orb_range) or orb_range <= 0:
        raise ValueError("Opening range must be positive")

    trade_start = DEFAULT_TRADING_WINDOWS[window_key]
    trading_slice = session.loc[session.index.time >= trade_start]
    if trading_slice.empty:
        raise ValueError("No trading bars following the opening range")

    return orb_high, orb_low, orb_range, trading_slice


def evaluate_direction(
    trading_bars: Iterable[pd.Series],
    capital: float,
    risk_pct: float,
    orb_range: float,
    orb_high: float,
    orb_low: float,
    tp_multiplier: float,
    direction: str,
) -> Tuple[bool, str, float]:
    risk_amount = capital * risk_pct
    entry = orb_high if direction == "long" else orb_low
    stop = orb_low if direction == "long" else orb_high
    target = (
        orb_high + tp_multiplier * orb_range
        if direction == "long"
        else orb_low - tp_multiplier * orb_range
    )

    triggered = False
    outcome = "No Trade"
    pnl = 0.0

    for _, bar in trading_bars.iterrows():
        high = float(bar["High"])
        low = float(bar["Low"])

        if not triggered:
            if direction == "long" and high >= entry:
                triggered = True
            elif direction == "short" and low <= entry:
                triggered = True

        if not triggered:
            continue

        if direction == "long":
            if high >= target:
                outcome = "TP"
                pnl = risk_amount * tp_multiplier
                break
            if low <= stop:
                outcome = "SL"
                pnl = -risk_amount
                break
        else:
            if low <= target:
                outcome = "TP"
                pnl = risk_amount * tp_multiplier
                break
            if high >= stop:
                outcome = "SL"
                pnl = -risk_amount
                break
    else:
        if triggered:
            outcome = "No Exit"

    return triggered, outcome, pnl


def analyze_session(
    symbol: str,
    session_date: pd.Timestamp,
    session: pd.DataFrame,
    capital: float,
    window_key: str,
    risk_pct: float,
    tp_multiplier: float,
) -> List[TradeResult]:
    orb_high, orb_low, orb_range, trading_bars = compute_opening_range(
        session, window_key
    )
    results: List[TradeResult] = []

    for direction in ("long", "short"):
        triggered, outcome, pnl = evaluate_direction(
            trading_bars=trading_bars,
            capital=capital,
            risk_pct=risk_pct,
            orb_range=orb_range,
            orb_high=orb_high,
            orb_low=orb_low,
            tp_multiplier=tp_multiplier,
            direction=direction,
        )
        results.append(
            TradeResult(
                date=session_date,
                symbol=symbol,
                opening_range=window_key,
                risk_pct=risk_pct,
                tp_multiplier=tp_multiplier,
                direction=direction,
                triggered=triggered,
                outcome=outcome,
                pnl=pnl,
                return_pct=(pnl / capital) * 100 if capital else 0.0,
                orb_high=orb_high,
                orb_low=orb_low,
                orb_range=orb_range,
            )
        )
    return results


def analyze_symbol(
    symbol: str,
    bars: pd.DataFrame,
    capital: float,
    opening_windows: List[str],
    risk_levels: List[float],
    tp_levels: List[float],
    today_only: bool = True,
) -> List[TradeResult]:
    """Run parameter sweep for one symbol (today-only if requested)."""
    records: List[TradeResult] = []

    if today_only:
        # Determine 'today' in ET to match session slicing
        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        unique_sessions = [pd.Timestamp(today_et)]
    else:
        unique_sessions = np.unique(bars.index.normalize())

    for session_date in unique_sessions:
        session_ts = pd.Timestamp(session_date)
        session = slice_session(bars, session_ts)
        if session.empty:
            continue

        for window_key, risk_pct, tp_multiplier in itertools.product(
            opening_windows, risk_levels, tp_levels
        ):
            try:
                records.extend(
                    analyze_session(
                        symbol=symbol,
                        session_date=session_ts,
                        session=session,
                        capital=capital,
                        window_key=window_key,
                        risk_pct=risk_pct,
                        tp_multiplier=tp_multiplier,
                    )
                )
            except ValueError:
                continue

    return records


# ------------------------------ Summary & Charts -------------------------------


def summarize_results(records: pd.DataFrame, report_dir: Path) -> None:
    triggered = records[records["triggered"]]
    if triggered.empty:
        print("No triggered trades to summarize.")
        return

    # --- Outcome breakdown ---
    outcome_counts = (
        triggered.groupby(["symbol", "opening_range", "tp_multiplier", "risk_pct"])[
            "outcome"
        ]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    summary = (
        triggered.groupby(
            ["symbol", "opening_range", "tp_multiplier", "risk_pct"], as_index=False
        )
        .agg(
            trades=("outcome", "count"),
            wins=("outcome", lambda s: (s == "TP").sum()),
            net_pnl=("pnl", "sum"),
            avg_return_pct=("return_pct", "mean"),
        )
        .assign(
            win_rate=lambda df: np.where(
                df["trades"] > 0, df["wins"] / df["trades"], 0.0
            )
        )
        .sort_values("net_pnl", ascending=False)
    )

    # merge outcome breakdowns into summary
    summary = pd.merge(
        summary,
        outcome_counts,
        on=["symbol", "opening_range", "tp_multiplier", "risk_pct"],
        how="left",
    )

    summary.to_csv(report_dir / "parameter_summary.csv", index=False)

    # You’ll now see extra columns: TP, SL, No Exit, No Trade
    print("\nSample rows with outcome counts:")
    print(summary.head().to_string(index=False))

    # keep your combo-view plots as before...

    safe_write_csv(summary, report_dir / "parameter_summary.csv")

    combo_view = (
        summary.groupby(["opening_range", "tp_multiplier", "risk_pct"], as_index=False)
        .agg(net_pnl=("net_pnl", "sum"))
        .sort_values("net_pnl", ascending=False)
    )
    combo_view["label"] = combo_view.apply(
        lambda row: f"{row['opening_range']} | TP {row['tp_multiplier']} | Risk {int(row['risk_pct'] * 100)}%",
        axis=1,
    )

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(combo_view["label"], combo_view["net_pnl"])
    ax.set_title("Net PnL by Parameter Combination")
    ax.set_ylabel("Net PnL")
    ax.set_xlabel("Opening Range | TP | Risk")
    ax.tick_params(axis="x", labelrotation=25)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    safe_savefig(fig, report_dir / "parameter_performance.png", dpi=200)
    plt.close(fig)

    leaders = (
        summary.groupby("symbol", as_index=False)["net_pnl"]
        .sum()
        .sort_values("net_pnl", ascending=False)
    )
    print("\nTop performing symbols (net PnL):")
    print(leaders.head(5).to_string(index=False))

    print("\nTop parameter combinations (all symbols):")
    print(
        combo_view.head(5)[
            ["opening_range", "tp_multiplier", "risk_pct", "net_pnl"]
        ].to_string(index=False)
    )


def flatten_symbols(config: Dict[str, object]) -> List[str]:
    symbols: List[str] = []
    for category in ("futures", "forex", "commodities", "stocks"):
        for item in config.get("symbols", {}).get(category, []):
            symbols.append(item["symbol"])
    return symbols


# ===================== COMPREHENSIVE REPORT BUILDERS =====================


def _label_combo(row):
    return f"{row['opening_range']} | TP {row['tp_multiplier']} | Risk {int(row['risk_pct'] * 100)}%"


def _build_combo_summary(triggered_df: pd.DataFrame) -> pd.DataFrame:
    if triggered_df.empty:
        return pd.DataFrame()
    g = triggered_df.groupby(
        ["opening_range", "tp_multiplier", "risk_pct"], as_index=False
    ).agg(
        net_pnl=("pnl", "sum"),
        avg_return=("return_pct", "mean"),
        win_rate=("outcome", lambda s: (s.eq("TP").sum()) / len(s) if len(s) else 0.0),
        total_trades=("outcome", "count"),
    )
    g["label"] = g.apply(_label_combo, axis=1)
    return g.sort_values("net_pnl", ascending=False)


def _build_symbol_summary(triggered_df: pd.DataFrame) -> pd.DataFrame:
    if triggered_df.empty:
        return pd.DataFrame()
    return (
        triggered_df.groupby("symbol", as_index=False)
        .agg(
            net_pnl=("pnl", "sum"),
            avg_return=("return_pct", "mean"),
            win_rate=(
                "outcome",
                lambda s: (s.eq("TP").sum()) / len(s) if len(s) else 0.0,
            ),
            trades=("outcome", "count"),
        )
        .sort_values("net_pnl", ascending=False)
    )


def _plot_top_symbol_detail(all_df: pd.DataFrame, report_dir: Path) -> None:
    triggered = all_df[all_df["triggered"]].copy()
    sym_sum = _build_symbol_summary(triggered)
    if sym_sum.empty:
        return
    top_symbol = sym_sum.iloc[0]["symbol"]

    sym_df = triggered[triggered["symbol"] == top_symbol].copy()
    sym_df = sym_df.sort_values(["date", "direction"]).reset_index(drop=True)
    sym_df["cum_pnl"] = sym_df["pnl"].cumsum()

    outcome_counts = sym_df["outcome"].value_counts()

    or_high = sym_df["orb_high"].iloc[0] if not sym_df.empty else float("nan")
    or_low = sym_df["orb_low"].iloc[0] if not sym_df.empty else float("nan")
    or_range = sym_df["orb_range"].iloc[0] if not sym_df.empty else float("nan")

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(sym_df.index.values, sym_df["cum_pnl"])
    ax1.set_title(f"{top_symbol} – Intraday Equity (today)")
    ax1.set_xlabel("Trade # (triggered)")
    ax1.set_ylabel("Cumulative PnL")
    ax1.axhline(0, linewidth=1, color="black")

    ax2 = fig.add_subplot(gs[0, 1])
    if not outcome_counts.empty:
        ax2.bar(outcome_counts.index.astype(str), outcome_counts.values)
        ax2.set_title("Outcome Breakdown")
        ax2.set_xlabel("Outcome")
        ax2.set_ylabel("Count")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    text = f"OR High:  {or_high:.4f}\nOR Low:   {or_low:.4f}\nOR Range: {or_range:.4f}"
    ax3.text(0.05, 0.9, text, va="top", fontfamily="monospace")

    sym_combo = (
        sym_df.groupby(["opening_range", "tp_multiplier", "risk_pct"], as_index=False)
        .agg(net_pnl=("pnl", "sum"), trades=("outcome", "count"))
        .sort_values("net_pnl", ascending=False)
    )
    sym_combo["label"] = sym_combo.apply(_label_combo, axis=1)

    ax4 = fig.add_subplot(gs[:, 2])
    if not sym_combo.empty:
        ax4.barh(range(len(sym_combo)), sym_combo["net_pnl"])
        ax4.set_yticks(range(len(sym_combo)))
        ax4.set_yticklabels(sym_combo["label"], fontsize=8)
        ax4.invert_yaxis()
        ax4.set_title("Net PnL by Combo (Top Symbol)")
        ax4.set_xlabel("Net PnL")

    fig.tight_layout()
    out = report_dir / f"top_symbol_report_{top_symbol.replace('=', '_')}.png"
    safe_savefig(fig, out, dpi=150, bbox_inches="tight")
    plt.close(fig)


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def build_comprehensive_report(all_df: pd.DataFrame, report_dir: Path) -> None:
    """Create a multi-panel overview (today-only) + a top-symbol detail chart."""
    # Work only with rows that actually triggered trades
    triggered = all_df[all_df["triggered"]].copy()

    # Pre-computed summaries
    combo_summary = _build_combo_summary(triggered)
    symbol_summary = _build_symbol_summary(triggered)

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(3, 4, figure=fig)

    # 1) Top combos by net PnL
    ax1 = fig.add_subplot(gs[0, :2])
    topc = combo_summary.head(8)
    if topc.empty:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "No combo data", ha="center", va="center")
    else:
        # Determine highlighted bar (highest net PnL among the displayed)
        best_combo_row = topc.sort_values("net_pnl", ascending=False).iloc[0]
        best_combo_label = best_combo_row["label"]

        # Colors: highlight the best combo
        combo_colors = [
            "darkorange" if label == best_combo_label else "steelblue"
            for label in topc["label"]
        ]
        bars = ax1.barh(
            range(len(topc)), topc["net_pnl"], color=combo_colors, alpha=0.85
        )
        ax1.set_yticks(range(len(topc)))
        ax1.set_yticklabels(topc["label"], fontsize=9)
        ax1.invert_yaxis()
        ax1.set_title("Top Parameter Combos (Net PnL)")
        ax1.set_xlabel("Net PnL")

        # Value labels
        offset = max(1.0, float(topc["net_pnl"].abs().max())) * 0.02
        for bar, net_pnl in zip(bars, topc["net_pnl"].fillna(0.0).to_numpy()):
            sign = 1 if net_pnl >= 0 else -1
            ax1.text(
                bar.get_width() + sign * offset,
                bar.get_y() + bar.get_height() / 2,
                f"${net_pnl:,.0f}",
                va="center",
                ha="left" if sign > 0 else "right",
            )

        ax1.text(
            0.02,
            0.98,
            "Highlighted = Highest Net PnL",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe6c7", alpha=0.6),
        )

    # 2) Avg return by symbol (with total PnL labels)
    ax2 = fig.add_subplot(gs[0, 2:])
    if symbol_summary.empty:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No symbol data", ha="center", va="center")
    else:
        # Sort by avg return for plotting, but highlight the symbol with highest total PnL
        s = symbol_summary.sort_values("avg_return", ascending=False).reset_index(
            drop=True
        )
        best_total_pnl_row = symbol_summary.sort_values(
            "net_pnl", ascending=False
        ).iloc[0]
        best_total_pnl_symbol = best_total_pnl_row["symbol"]

        colors = [
            "darkorange" if sym == best_total_pnl_symbol else "steelblue"
            for sym in s["symbol"]
        ]
        bars = ax2.bar(range(len(s)), s["avg_return"], color=colors, alpha=0.85)
        ax2.set_xticks(range(len(s)))
        ax2.set_xticklabels(s["symbol"], rotation=45, ha="right")
        ax2.set_title("Average Return by Symbol (today)")
        ax2.set_ylabel("Avg Return (%)")
        ax2.axhline(0, color="black", linewidth=1)

        # Put total PnL labels above/below each bar
        return_offset = max(1.0, float(s["avg_return"].abs().max())) * 0.02
        for bar, net_pnl in zip(bars, s["net_pnl"].fillna(0.0).to_numpy()):
            height = bar.get_height()
            sign = 1 if height >= 0 else -1
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + sign * return_offset,
                f"${net_pnl:,.0f}",
                ha="center",
                va="bottom" if sign > 0 else "top",
                fontsize=9,
            )

        ax2.text(
            0.02,
            0.95,
            "Highlighted = Highest Total PnL",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe6c7", alpha=0.6),
        )

    # 3) Win-rate distribution (by symbol+combo)
    ax3 = fig.add_subplot(gs[1, :2])
    if triggered.empty:
        ax3.axis("off")
        ax3.text(0.5, 0.5, "No triggered trades", ha="center", va="center")
    else:
        wr = (
            triggered.groupby(["symbol", "opening_range", "tp_multiplier", "risk_pct"])[
                "outcome"
            ]
            .apply(lambda s_: (s_ == "TP").sum() / len(s_))
            .reset_index(name="win_rate")
        )
        ax3.hist(wr["win_rate"] * 100, bins=10, edgecolor="black")
        ax3.set_title("Win Rate Distribution (by symbol+combo)")
        ax3.set_xlabel("Win Rate (%)")
        ax3.set_ylabel("Frequency")

    # 4) Trade counts per symbol
    ax4 = fig.add_subplot(gs[1, 2:])
    if symbol_summary.empty:
        ax4.axis("off")
        ax4.text(0.5, 0.5, "No symbol data", ha="center", va="center")
    else:
        s_counts = symbol_summary.sort_values("trades", ascending=False).reset_index(
            drop=True
        )
        ax4.bar(range(len(s_counts)), s_counts["trades"])
        ax4.set_xticks(range(len(s_counts)))
        ax4.set_xticklabels(s_counts["symbol"], rotation=45, ha="right")
        ax4.set_title("Triggered Trade Counts (today)")
        ax4.set_ylabel("Count")

    # 5) Text summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    total_trades = int(triggered.shape[0])
    total_symbols = int(triggered["symbol"].nunique())

    # Best combo (by net PnL)
    if combo_summary.empty:
        best_combo_label = "N/A"
    else:
        best_combo_label = combo_summary.sort_values("net_pnl", ascending=False).iloc[
            0
        ]["label"]

    # Top symbols (by total PnL and by avg return)
    if symbol_summary.empty:
        top_total_pnl_symbol = "N/A"
        top_total_pnl_value = 0.0
        top_avg_return_symbol = "N/A"
        top_avg_return_value = "N/A"
    else:
        top_total_row = symbol_summary.sort_values("net_pnl", ascending=False).iloc[0]
        top_total_pnl_symbol = top_total_row["symbol"]
        top_total_pnl_value = (
            float(top_total_row["net_pnl"])
            if pd.notna(top_total_row["net_pnl"])
            else 0.0
        )

        top_avg_row = symbol_summary.sort_values("avg_return", ascending=False).iloc[0]
        top_avg_return_symbol = top_avg_row["symbol"]
        top_avg_return_value = (
            f"{float(top_avg_row['avg_return']):.2f}%"
            if pd.notna(top_avg_row["avg_return"])
            else "N/A"
        )

    summary_text = (
        "TODAY SUMMARY\n"
        "=============\n"
        f"Symbols traded: {total_symbols}\n"
        f"Triggered trades: {total_trades}\n"
        f"Best combo: {best_combo_label}\n"
        f"Top symbol (total PnL): {top_total_pnl_symbol} (${top_total_pnl_value:,.2f})\n"
        f"Top symbol (avg return): {top_avg_return_symbol} ({top_avg_return_value})\n"
    )
    ax5.text(0.02, 0.95, summary_text, va="top", fontfamily="monospace")

    fig.suptitle("Daily ORB – Comprehensive Overview", fontsize=16, fontweight="bold")
    # Make room for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    safe_savefig(
        fig, report_dir / "comprehensive_overview.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Top-symbol detail
    _plot_top_symbol_detail(all_df, report_dir)


def build_futures_focus_report(
    all_df: pd.DataFrame, report_dir: Path, focus_symbols: List[str] = ("MNQ=F", "NQ=F")
) -> None:
    """
    Create a today-only summary comparing MNQ=F / NQ=F vs others.
    Outputs:
      - focus_symbol_summary.csv (MNQ=F, NQ=F only)
      - others_symbol_summary.csv (all non-focus symbols)
      - focus_vs_others_summary.csv (side-by-side aggregates)
      - futures_focus.png (two-panel chart)
    """
    if all_df.empty:
        print("[FUTURES FOCUS] No records.")
        return

    triggered = all_df[all_df["triggered"]].copy()
    if triggered.empty:
        print("[FUTURES FOCUS] No triggered trades.")
        return

    # Per-symbol rollup (today)
    sym_sum = (
        triggered.groupby("symbol", as_index=False)
        .agg(
            net_pnl=("pnl", "sum"),
            avg_return=("return_pct", "mean"),
            win_rate=(
                "outcome",
                lambda s: (s.eq("TP").sum()) / len(s) if len(s) else 0.0,
            ),
            trades=("outcome", "count"),
        )
        .sort_values("net_pnl", ascending=False)
    )

    focus_df = sym_sum[sym_sum["symbol"].isin(focus_symbols)].copy()
    others_df = sym_sum[~sym_sum["symbol"].isin(focus_symbols)].copy()

    # Aggregate view: focus (sum/sum/mean) vs others (sum/sum/mean)
    def _agg_block(df: pd.DataFrame, label: str) -> Dict[str, float]:
        if df.empty:
            return {
                "group": label,
                "net_pnl": 0.0,
                "avg_return": 0.0,
                "win_rate": 0.0,
                "trades": 0,
            }
        return {
            "group": label,
            "net_pnl": float(df["net_pnl"].sum()),
            "avg_return": float(df["avg_return"].mean()),
            "win_rate": float(df["win_rate"].mean()),
            "trades": int(df["trades"].sum()),
        }

    # Build a combined table: rows for MNQ=F, NQ=F, Others (aggregate)
    rows = []
    for sym in focus_symbols:
        row = sym_sum[sym_sum["symbol"] == sym]
        if row.empty:
            rows.append(
                {
                    "group": sym,
                    "net_pnl": 0.0,
                    "avg_return": 0.0,
                    "win_rate": 0.0,
                    "trades": 0,
                }
            )
        else:
            r = row.iloc[0]
            rows.append(
                {
                    "group": sym,
                    "net_pnl": float(r["net_pnl"]),
                    "avg_return": float(r["avg_return"]),
                    "win_rate": float(r["win_rate"]),
                    "trades": int(r["trades"]),
                }
            )
    rows.append(_agg_block(others_df, "Others (aggregate)"))
    comp_df = pd.DataFrame(rows)

    # Save CSVs (Windows-safe)
    safe_write_csv(focus_df, report_dir / "focus_symbol_summary.csv")
    safe_write_csv(others_df, report_dir / "others_symbol_summary.csv")
    safe_write_csv(comp_df, report_dir / "focus_vs_others_summary.csv")

    # Chart: two panels: Net PnL (left) and Avg Return % (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Left: Net PnL
    axes[0].bar(comp_df["group"], comp_df["net_pnl"])
    axes[0].set_title("Net PnL – MNQ/NQ vs Others (today)")
    axes[0].set_ylabel("Net PnL")
    axes[0].tick_params(axis="x", labelrotation=0)

    # Right: Avg Return % (mean of per-symbol avg_return)
    axes[1].bar(comp_df["group"], comp_df["avg_return"])
    axes[1].set_title("Average Return % – MNQ/NQ vs Others (today)")
    axes[1].set_ylabel("Avg Return (%)")
    axes[1].tick_params(axis="x", labelrotation=0)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=1)

    fig.tight_layout()
    safe_savefig(fig, report_dir / "futures_focus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Console summary for quick glance
    print("\n[FUTURES FOCUS] MNQ/NQ vs Others (today)")
    print(comp_df.to_string(index=False))


# ------------------------------------- Performance Summary Table ------------------------------------
def print_and_save_performance_report(all_df: pd.DataFrame, report_dir: Path) -> None:
    """Build a detailed performance report, print it, and save to performance_report.txt."""
    if all_df.empty:
        msg = "[REPORT] No trades to summarize."
        print(msg)
        safe_write_text(report_dir / "performance_report.txt", msg + "\n")
        return

    triggered = all_df[all_df["triggered"]].copy()
    if triggered.empty:
        msg = "[REPORT] No triggered trades to summarize."
        print(msg)
        safe_write_text(report_dir / "performance_report.txt", msg + "\n")
        return

    # Map categories from config
    config = load_config(CONFIG_PATH)
    symbol_to_cat: Dict[str, str] = {}
    for cat, syms in config.get("symbols", {}).items():
        for s in syms:
            symbol_to_cat[s["symbol"]] = cat

    # --- By symbol ---
    sym_stats = triggered.groupby("symbol", as_index=False).agg(
        total_trades=("outcome", "count"),
        wins=("outcome", lambda s: (s == "TP").sum()),
        losses=("outcome", lambda s: (s == "SL").sum()),
        net_pnl=("pnl", "sum"),
        avg_win=("pnl", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
        avg_loss=("pnl", lambda s: s[s < 0].mean() if (s < 0).any() else 0.0),
        max_dd=("pnl", lambda s: s.min()),  # largest losing trade (quick proxy)
    )
    sym_stats["win_rate"] = sym_stats["wins"] / sym_stats["total_trades"]
    # Profit factor: (avg win * #wins) / (|avg loss| * #losses)
    sym_stats["profit_factor"] = sym_stats.apply(
        lambda r: ((r["avg_win"] * r["wins"]) / abs(r["avg_loss"] * r["losses"]))
        if (r["losses"] > 0 and r["avg_loss"] < 0)
        else np.nan,
        axis=1,
    )
    init_cap = float(config["strategy"]["risk_management"]["initial_capital"])
    sym_stats["return_pct"] = (sym_stats["net_pnl"] / init_cap) * 100.0
    sym_stats["category"] = (
        sym_stats["symbol"].map(symbol_to_cat).fillna("Uncategorised")
    )

    # Best & worst trade across all triggered
    best_idx = triggered["pnl"].idxmax()
    worst_idx = triggered["pnl"].idxmin()
    best_trade = triggered.loc[best_idx] if pd.notna(best_idx) else None
    worst_trade = triggered.loc[worst_idx] if pd.notna(worst_idx) else None

    # --- By category ---
    cat_stats = sym_stats.groupby("category", as_index=False).agg(
        symbols=("symbol", "count"),
        total_trades=("total_trades", "sum"),
        avg_win_rate=("win_rate", "mean"),
        total_pnl=("net_pnl", "sum"),
        avg_return=("return_pct", "mean"),
    )

    # --- Overall ---
    total_trades = int(sym_stats["total_trades"].sum())
    total_wins = int(sym_stats["wins"].sum())
    total_losses = int(sym_stats["losses"].sum())
    total_pnl = float(sym_stats["net_pnl"].sum())
    overall_wr = (total_wins / total_trades) if total_trades else 0.0
    avg_trade_pnl = (total_pnl / total_trades) if total_trades else 0.0

    # Format tables
    sym_table = sym_stats[
        [
            "symbol",
            "category",
            "total_trades",
            "win_rate",
            "net_pnl",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "max_dd",
            "return_pct",
        ]
    ].to_string(
        index=False,
        formatters={
            "win_rate": "{:.1%}".format,
            "net_pnl": "${:,.2f}".format,
            "avg_win": "${:,.2f}".format,
            "avg_loss": "${:,.2f}".format,
            "profit_factor": (lambda x: "—" if pd.isna(x) else f"{x:.2f}"),
            "max_dd": "${:,.2f}".format,
            "return_pct": "{:.1f}%".format,
        },
    )

    cat_table = cat_stats.to_string(
        index=False,
        formatters={
            "avg_win_rate": "{:.1%}".format,
            "total_pnl": "${:,.2f}".format,
            "avg_return": "{:.1f}%".format,
        },
    )

    # Build full text
    lines = []
    lines.append(
        "--------------------------------------------------------------------------------"
    )
    lines.append("Performance by Symbol:")
    lines.append(
        "--------------------------------------------------------------------------------"
    )
    lines.append(sym_table)
    lines.append("")
    lines.append(
        "--------------------------------------------------------------------------------"
    )
    lines.append("Performance by Category:")
    lines.append(
        "--------------------------------------------------------------------------------"
    )
    lines.append(cat_table)
    lines.append("")
    lines.append(
        "--------------------------------------------------------------------------------"
    )
    lines.append("Overall Statistics:")
    lines.append(
        "--------------------------------------------------------------------------------"
    )
    lines.append(f"Total Trades Executed: {total_trades}")
    lines.append(f"Total Winning Trades: {total_wins}")
    lines.append(f"Total Losing Trades: {total_losses}")
    lines.append(f"Overall Win Rate: {overall_wr:.1%}")
    lines.append(f"Total PnL: ${total_pnl:,.2f}")
    lines.append(f"Average Trade PnL: ${avg_trade_pnl:,.2f}")
    if best_trade is not None:
        lines.append("\nBest Trade:")
        lines.append(f"  Symbol: {best_trade['symbol']}")
        lines.append(f"  Type: {best_trade['direction'].upper()}")
        lines.append(f"  PnL: ${best_trade['pnl']:.2f}")
        lines.append(f"  Date: {best_trade['date']}")
    if worst_trade is not None:
        lines.append("\nWorst Trade:")
        lines.append(f"  Symbol: {worst_trade['symbol']}")
        lines.append(f"  Type: {worst_trade['direction'].upper()}")
        lines.append(f"  PnL: ${worst_trade['pnl']:.2f}")
        lines.append(f"  Date: {worst_trade['date']}")

    report_text = "\n".join(lines)

    # Print to console
    print("\n" + report_text + "\n")

    # Save to file
    safe_write_text(report_dir / "performance_report.txt", report_text + "\n")


# ------------------------------------- main ------------------------------------


def main() -> None:
    config = load_config(CONFIG_PATH)

    capital = float(config["strategy"]["risk_management"]["initial_capital"])
    symbol_list = flatten_symbols(config)

    parameter_block = config.get("daily_orb", {}).get("parameter_sweep", {})
    opening_windows = parameter_block.get("opening_windows", ["15m"])
    risk_levels = parameter_block.get("risk_percents", [0.01, 0.02])
    tp_levels = parameter_block.get("tp_multipliers", [2.0, 1.0, 0.5])

    today_only = config.get("data", {}).get("today_only", True)

    missing_windows = set(opening_windows) - set(DEFAULT_OPENING_WINDOWS.keys())
    if missing_windows:
        raise ValueError(f"Unsupported opening window(s): {sorted(missing_windows)}")

    report_dir = REPORT_ROOT / datetime.now(ZoneInfo("America/New_York")).strftime(
        "%m%d%Y"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[TradeResult] = []

    for symbol in symbol_list:
        print(f"Analyzing {symbol}...")
        try:
            bars = load_bars(symbol, interval="5m")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [WARN] skipping {symbol}: {exc}")
            continue

        records = analyze_symbol(
            symbol=symbol,
            bars=bars,
            capital=capital,
            opening_windows=opening_windows,
            risk_levels=risk_levels,
            tp_levels=tp_levels,
            today_only=today_only,
        )

        if not records:
            print("  No valid trades for available sessions.")
            continue

        symbol_df = pd.DataFrame(r.as_dict for r in records)
        safe_write_csv(
            symbol_df, report_dir / f"{safe_symbol(symbol)}_daily_records.csv"
        )
        all_records.extend(records)

    if not all_records:
        print("No trades were generated for any symbol.")
        print(f"Reports directory: {report_dir.resolve()}")
        return

    all_df = pd.DataFrame(r.as_dict for r in all_records)
    safe_write_csv(all_df, report_dir / "combined_daily_records.csv")

    summarize_results(all_df, report_dir)

    build_comprehensive_report(all_df, report_dir)

    build_futures_focus_report(all_df, report_dir, focus_symbols=["MNQ=F", "NQ=F"])

    print_and_save_performance_report(all_df, report_dir)

    print(f"\nReports saved to: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
