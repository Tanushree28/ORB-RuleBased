"""Generate chronological MNQ trade summary and visualization.

This utility aggregates the MNQ daily ORB records that are produced by ``generate_daily_orb_yf.py`` and condenses them into the same style of
chronological log that is maintained manually in the spreadsheet titled"MNQ Multiple Chronological 2-1".  The script focuses exclusively on the
2:1 reward-to-risk configuration (``tp_multiplier == 2``) and produces:

* A day-by-day table showing buy/sell trade results, daily totals and
  running cumulative performance
* Helper columns that flag double/single win/lose sessions together with
  cumulative counts for each pattern
* Weekly and monthly rollups that mirror the high-level totals on the
  spreadsheet
* A markdown summary file with headline metrics and narrative takeaways
* A colour-coded PNG image of the table so it can be dropped into reports

Example usage::

    $ python generate_mnq_chronological_summary.py \
        --records-root reports/daily_orb_yf \
        --output-dir reports/mnq_summary

The command above will scan every dated folder under ``reports/daily_orb_yf``
for the MNQ records, build the chronological view, save a markdown digest
and export ``mnq_chronological_log.png`` inside ``reports/mnq_summary``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


MNQ_FILE_PATTERN = "MNQ_F_daily_records.csv"
PLOT_COLUMNS = [
    "Date",
    "Day",
    "Buy Result",
    "Sell Result",
    "First Trade",
    "First Result",
    "Second Trade",
    "Second Result",
    "Daily Result",
    "Cumulative",
    "Double Win",
    "Double Lose",
    "Single Win",
    "Single Lose",
    "Double Win Cum",
    "Double Lose Cum",
    "Single Win Cum",
    "Single Lose Cum",
]

PATTERN_FLAG_COLUMNS = ["Double Win", "Double Lose", "Single Win", "Single Lose"]
PATTERN_CUM_COLUMNS = [
    "Double Win Cum",
    "Double Lose Cum",
    "Single Win Cum",
    "Single Lose Cum",
]


@dataclass
class DailyTradeOutcome:
    """Container with daily buy/sell results and derived statistics."""

    date: pd.Timestamp
    day_name: str
    buy_triggered: bool
    sell_triggered: bool
    buy_score: Optional[int]
    sell_score: Optional[int]
    first_trade: str
    second_trade: str
    first_score: Optional[int]
    second_score: Optional[int]
    daily_result: int
    cumulative: int
    double_win: int
    double_loss: int
    single_win: int
    single_loss: int
    wins: int
    losses: int
    trades: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "Date": self.date.date(),
            "Day": self.day_name,
            "Buy Result": self.buy_score,
            "Sell Result": self.sell_score,
            "First Trade": self.first_trade,
            "First Result": self.first_score,
            "Second Trade": self.second_trade,
            "Second Result": self.second_score,
            "Daily Result": self.daily_result,
            "Cumulative": self.cumulative,
            "Double Win": self.double_win,
            "Double Lose": self.double_loss,
            "Single Win": self.single_win,
            "Single Lose": self.single_loss,
            "Wins": self.wins,
            "Losses": self.losses,
            "Trades": self.trades,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path("reports/daily_orb_yf"),
        help="Directory that contains dated folders with *_daily_records.csv files",
    )
    parser.add_argument(
        "--symbol",
        default="MNQ=F",
        help="Symbol column to use when reading the daily records",
    )
    parser.add_argument(
        "--tp-multiplier",
        type=float,
        default=2.0,
        help="Take-profit multiplier to filter on (default: 2.0)",
    )
    parser.add_argument(
        "--risk-pct",
        type=float,
        default=0.01,
        help="Risk percentage per trade (default: 1%%)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/mnq_summary"),
        help="Directory where markdown and figure outputs will be stored",
    )
    parser.add_argument(
        "--start-with",
        choices=["buy", "sell"],
        default="buy",
        help=(
            "Starting direction for the chronological rotation."
            " Days alternate BUY/SELL in sequence."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of most recent sessions to include",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Print a console preview of the chronological table, weekly, and "
            "monthly rollups after generation."
        ),
    )
    return parser.parse_args()


def discover_record_files(root: Path) -> List[Path]:
    files = sorted(root.rglob(MNQ_FILE_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"Could not find any '{MNQ_FILE_PATTERN}' files under {root.resolve()}"
        )
    return files


def load_records(files: Iterable[Path], symbol: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["symbol"] == symbol]
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        raise ValueError(
            f"No records for symbol '{symbol}' were found in the discovered files"
        )
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("date", inplace=True)
    return combined


def to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "t", "1", "yes"}
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(value)


def outcome_to_score(outcome: str) -> Optional[int]:
    if not isinstance(outcome, str):
        return None
    outcome = outcome.strip().lower()
    if outcome == "tp":
        return 2
    if outcome == "sl":
        return -1
    if outcome in {"no exit", "no_exit", "flat"}:
        return 0
    return None


def rotate_direction(prev: str) -> str:
    return "sell" if prev == "buy" else "buy"


def build_daily_rows(
    df: pd.DataFrame, start_direction: str, limit: Optional[int] = None
) -> List[DailyTradeOutcome]:
    rows: List[DailyTradeOutcome] = []
    running_total = 0
    active_direction = start_direction

    grouped = df.groupby("date", sort=True)
    if limit is not None:
        grouped = list(grouped)[-limit:]

    for date, day_df in grouped:
        buy_row = day_df[day_df["direction"] == "long"].head(1)
        sell_row = day_df[day_df["direction"] == "short"].head(1)

        buy_triggered = (
            to_bool(buy_row.iloc[0]["triggered"]) if not buy_row.empty else False
        )
        sell_triggered = (
            to_bool(sell_row.iloc[0]["triggered"]) if not sell_row.empty else False
        )

        buy_score = (
            outcome_to_score(buy_row.iloc[0]["outcome"])
            if buy_triggered and not buy_row.empty
            else None
        )
        sell_score = (
            outcome_to_score(sell_row.iloc[0]["outcome"])
            if sell_triggered and not sell_row.empty
            else None
        )

        first_trade = active_direction.capitalize()
        second_trade = rotate_direction(active_direction).capitalize()

        first_score = buy_score if active_direction == "buy" else sell_score
        second_score = sell_score if active_direction == "buy" else buy_score

        # Replace None with 0 for daily tally but keep None for display clarity
        normalized_scores = [s for s in [buy_score, sell_score] if s is not None]
        daily_result = sum(normalized_scores)

        wins = sum(1 for score in normalized_scores if score > 0)
        losses = sum(1 for score in normalized_scores if score < 0)
        trades = len(normalized_scores)

        double_win = int(wins == 2)
        double_loss = int(losses == 2)
        single_win = int(wins == 1 and losses == 0)
        single_loss = int(losses == 1 and wins == 0)

        running_total += daily_result

        rows.append(
            DailyTradeOutcome(
                date=pd.Timestamp(date),
                day_name=pd.Timestamp(date).day_name(),
                buy_triggered=buy_triggered,
                sell_triggered=sell_triggered,
                buy_score=buy_score,
                sell_score=sell_score,
                first_trade=first_trade,
                second_trade=second_trade,
                first_score=first_score,
                second_score=second_score,
                daily_result=daily_result,
                cumulative=running_total,
                double_win=double_win,
                double_loss=double_loss,
                single_win=single_win,
                single_loss=single_loss,
                wins=wins,
                losses=losses,
                trades=trades,
            )
        )

        active_direction = rotate_direction(active_direction)

    return rows


def format_table_values(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    formatted["Date"] = pd.to_datetime(formatted["Date"]).dt.strftime("%Y-%m-%d")

    for column in ["Buy Result", "Sell Result", "First Result", "Second Result"]:
        if column not in formatted.columns:
            continue
        formatted[column] = formatted[column].apply(
            lambda x: "" if pd.isna(x) else f"{int(x):+d}" if x != 0 else "0"
        )

    for column in ["Daily Result", "Cumulative"]:
        formatted[column] = formatted[column].map(lambda x: f"{int(x):+d}")

    for column in PATTERN_FLAG_COLUMNS:
        if column not in formatted.columns:
            continue

        def _fmt_flag(value: float) -> str:
            if pd.isna(value):
                return ""
            as_int = int(value)
            return "" if as_int == 0 else str(as_int)

        formatted[column] = formatted[column].map(_fmt_flag)

    for column in PATTERN_CUM_COLUMNS:
        if column not in formatted.columns:
            continue
        formatted[column] = formatted[column].map(
            lambda x: "" if pd.isna(x) else f"{int(x)}"
        )

    formatted["First Trade"] = formatted["First Trade"].astype(str)
    formatted["Second Trade"] = formatted["Second Trade"].astype(str)

    return formatted


def color_for_value(value: Optional[int]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#f6f6f6"
    if value > 0:
        return "#b7e1cd"  # soft green
    if value < 0:
        return "#f4c7c3"  # soft red
    return "#fce8b2"  # neutral amber for flat / 0


def plot_table(df: pd.DataFrame, output_path: Path) -> None:
    formatted = format_table_values(df[PLOT_COLUMNS])

    fig_height = max(4, 0.4 * len(formatted) + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#d9ead3")
            cell.set_text_props(weight="bold")
            continue

        column_name = formatted.columns[col_idx]
        raw_value = df.iloc[row_idx - 1][column_name]
        if column_name in {
            "First Result",
            "Second Result",
            "Daily Result",
            "Cumulative",
        }:
            cell.set_facecolor(color_for_value(raw_value))
        elif column_name in {"First Trade", "Second Trade"}:
            cell.set_facecolor("#e7e6ff")
        elif column_name in PATTERN_FLAG_COLUMNS:
            cell.set_facecolor("#fde9d9" if raw_value else "#f6f6f6")
        elif column_name in PATTERN_CUM_COLUMNS:
            cell.set_facecolor("#fff2cc")
        else:
            cell.set_facecolor("#f6f6f6")

    plt.title("MNQ Chronological 2:1 ORB Results", fontsize=16, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def compute_weekly_monthly(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tmp = df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])

    weekly = (
        tmp.set_index("Date")
        .resample("W-MON")["Daily Result"]
        .sum()
        .reset_index()
        .rename(columns={"Date": "Week Start", "Daily Result": "Weekly Result"})
    )

    monthly = tmp.set_index("Date").resample("ME")["Daily Result"].sum().reset_index()
    monthly["Month"] = monthly["Date"].dt.strftime("%Y-%m")
    monthly.rename(columns={"Daily Result": "Monthly Result"}, inplace=True)
    monthly = monthly[["Month", "Monthly Result"]]

    return {"weekly": weekly, "monthly": monthly}


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def streak_lengths(series: pd.Series, condition) -> int:
    best = 0
    current = 0
    for value in series:
        if condition(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def write_markdown_summary(
    df: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    output_path: Path,
) -> None:
    total_trades = int(df["Trades"].sum())
    total_wins = int(df["Wins"].sum())
    total_losses = int(df["Losses"].sum())
    win_rate = (total_wins / total_trades * 100) if total_trades else 0
    net_points = int(df["Daily Result"].sum())

    win_streak = streak_lengths(df["Daily Result"], lambda x: x > 0)
    loss_streak = streak_lengths(df["Daily Result"], lambda x: x < 0)

    cumulative = df["Cumulative"]
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_drawdown = int(drawdown.max()) if not drawdown.empty else 0

    double_wins = int(df["Double Win"].sum())
    double_losses = int(df["Double Lose"].sum())

    start_date = df.iloc[0]["Date"]
    end_date = df.iloc[-1]["Date"]

    highlight_month = monthly.loc[monthly["Monthly Result"].idxmax()]
    weakest_month = monthly.loc[monthly["Monthly Result"].idxmin()]

    lines = [
        "# MNQ Chronological 2:1 Performance Summary",
        "",
        f"**Coverage:** {start_date} → {end_date}",
        "",
        "## Key Metrics",
        "",
        f"- Total trades: **{total_trades}**",
        f"- Wins / Losses: **{total_wins} / {total_losses}**",
        f"- Win rate: **{win_rate:.1f}%**",
        f"- Net result: **{net_points:+d}R**",
        f"- Max drawdown: **{max_drawdown}R**",
        f"- Double wins: **{double_wins}**  ·  Double losses: **{double_losses}**",
        f"- Longest winning streak: **{win_streak} days**",
        f"- Longest losing streak: **{loss_streak} days**",
        "",
        "## Weekly Momentum",
        "",
        dataframe_to_markdown(weekly),
        "",
        "## Monthly Scorecard",
        "",
        dataframe_to_markdown(monthly),
        "",
        "## Highlights",
        "",
        (
            f"- Strongest month: **{highlight_month['Month']}** with"
            f" **{int(highlight_month['Monthly Result']):+d}R**"
        ),
        (
            f"- Softest month: **{weakest_month['Month']}** with"
            f" **{int(weakest_month['Monthly Result']):+d}R**"
        ),
        (
            "- Momentum trend: cumulative curve climbed from"
            f" **{int(df.iloc[0]['Cumulative']):+d}R** to"
            f" **{int(df.iloc[-1]['Cumulative']):+d}R**"
        ),
        (
            "- Pattern distribution:"
            f" {double_wins} double-win sessions vs {double_losses} double-loss sessions"
        ),
    ]

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    files = discover_record_files(args.records_root)
    df = load_records(files, args.symbol)

    filtered = df[
        (df["tp_multiplier"].round(2) == round(args.tp_multiplier, 2))
        & (df["risk_pct"].round(4) == round(args.risk_pct, 4))
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No rows match the requested tp_multiplier and risk_pct configuration"
        )

    rows = build_daily_rows(filtered, start_direction=args.start_with, limit=args.limit)
    if not rows:
        raise RuntimeError("No daily outcomes could be derived from the records")

    summary_df = pd.DataFrame([row.as_dict() for row in rows])

    # Rolling cumulative counts for the pattern flags
    for column in ["Double Win", "Double Lose", "Single Win", "Single Lose"]:
        summary_df[f"{column} Cum"] = summary_df[column].cumsum()

    weekly_monthly = compute_weekly_monthly(summary_df)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_table(summary_df, output_dir / "mnq_chronological_log.png")

    write_markdown_summary(
        summary_df,
        weekly=weekly_monthly["weekly"],
        monthly=weekly_monthly["monthly"],
        output_path=output_dir / "mnq_chronological_summary.md",
    )

    csv_path = output_dir / "mnq_chronological_log.csv"
    summary_df.to_csv(csv_path, index=False)

    if args.preview:
        print("\n=== Chronological log (first 10 sessions) ===")
        print(summary_df.head(10).to_string(index=False))

        print("\n=== Cumulative pattern trackers (latest) ===")
        cum_cols = ["Date"] + PATTERN_CUM_COLUMNS
        print(summary_df[cum_cols].tail(1).to_string(index=False))

        print("\n=== Weekly rollup ===")
        print(weekly_monthly["weekly"].to_string(index=False))

        print("\n=== Monthly rollup ===")
        print(weekly_monthly["monthly"].to_string(index=False))

    print(f"\n✓ Saved chronological table: {csv_path}")
    print(f"✓ Saved markdown overview: {output_dir / 'mnq_chronological_summary.md'}")
    print(f"✓ Saved visual snapshot: {output_dir / 'mnq_chronological_log.png'}")


if __name__ == "__main__":
    main()
