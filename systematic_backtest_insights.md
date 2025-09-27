# Systematic ORB Backtest Highlights

`systematic_backtest.py` 
- iterates through ORB duration, take-profit multiplier, interval, and risk settings, calling the shared `BacktestEngine` for each scenario and storing per-symbol metrics such as win rate, profit factor, and return percentage.

- After the sweep completes, the script normalizes numeric columns and emits detailed as well as grouped CSV summaries that capture average returns, profit factors, and the share of profitable symbols for every parameter combination and instrument.


- `visualize_systematic_backtest.py` 
reloads the consolidated CSV, rebuilds the same combination and symbol aggregations, and prepares the data needed for downstream plotting.

- Its plotting helpers render heatmaps for each interval/risk pair, rank the parameter sets by the proportion of positive symbols, and chart the symbols with the strongest average returns, saving each figure under `reports/`.

- The heatmaps highlight that 5-minute data with a 1% risk budget consistently posts the strongest average returns (around 5% when TP is 0.5× or 1.0×), whereas 15-minute sessions trend negative across the tested multipliers.【


- Increasing risk to 2% on 5-minute bars still keeps results positive but trims the edge (average returns fall toward 3% for the 0.5× TP setting), suggesting diminishing benefits from higher leverage.

- Parameter sets combining 5-minute intervals, 0.5× TP, and 1% risk top the positive-symbol leaderboard, with more than half of symbols finishing green alongside ~5% average returns.

- Overall, shorter ORB durations paired with modest take-profit targets and restrained risk budgets deliver the broadest consistency, while longer ORB windows and aggressive multipliers underperform in this dataset.