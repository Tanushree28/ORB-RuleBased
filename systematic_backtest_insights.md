# Systematic ORB Backtest Highlights
# Systematic ORB Backtest Insights

## Key Insights

### 1. Top Performing Parameter Sets

-   The **5-minute ORB with low TP multipliers (0.5x--1.0x)**
    consistently outperformed others.
-   Risk at **1% per trade** gave more stable and higher returns than
    2%.
-   Best individual configuration: **5m \| TP 0.5x \| Risk 2%**, with
    \~5% average return.

### 2. Category-Level Returns

-   **Futures** led with \~15% average returns.
-   Commodities also showed positive performance (\~12%).
-   Equities and indices underperformed; some (e.g., GOOGL, AAPL, TSLA)
    had negative returns.

### 3. Symbol-Level Insights

-   **MNQ=F and NQ=F (Nasdaq futures)** were the strongest performers,
    each delivering \>10% average return.
-   **US equities (GOOGL, TSLA, AAPL, AMZN)** tended to be consistently
    negative under ORB rules.
-   CAD and Gold had small but positive returns.

### 4. Win Rate & Profitability

-   Distribution shows most strategies achieved **40--70% win rates**,
    with mean \~56.8%.
-   Profitability was more sensitive to **reward-to-risk ratio tuning**
    than just win rate.
-   Many strategies with win rates \>50% still had poor returns due to
    risk/TP imbalance.

### 5. Risk Sensitivity

-   **1% risk** setups generally outperformed **2% risk**, showing
    stability matters more than aggressiveness.
-   Higher risk amplified volatility without improving returns.

### 6. Robustness & Coverage

-   About **66.7% of runs were positive**, indicating good robustness.
-   Ten parameter-symbol combinations achieved both **positive returns
    and broad symbol coverage**, making them strong candidates for live
    strategies.

### 7. Trade Distribution

-   Most scenarios generated **60--80 trades**, ensuring enough sample
    size for statistical significance.
-   A few scenarios had \<40 trades, making their results less reliable.

------------------------------------------------------------------------

## Conclusion Summary

The systematic ORB sweep reveals that **shorter opening ranges (5m),
lower TP multipliers (0.5x--1.0x), and conservative risk (1%)** deliver
the most consistent and robust results. Futures (especially Nasdaq
futures MNQ=F, NQ=F) are the strongest asset class for ORB strategies,
with double-digit returns and stable performance, while large-cap US
equities generally underperform. Win rate alone isn't sufficient;
aligning TP/SL ratios with asset behavior is crucial. Overall, the
strategy is robust, with \~67% of tested combinations yielding positive
returns, but its edge is concentrated in futures rather than equities.


------------------------------------ Files With Cod ----------------------------------------

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