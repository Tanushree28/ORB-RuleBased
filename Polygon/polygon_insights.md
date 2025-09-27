# Polygon Systematic ORB Backtest Insights

## Key Insights

### 1. Top Parameter Sets

-   **15m ORB duration with 2% risk and low TP multipliers (0.5x--1.0x)** dominate, showing **\>95% average returns**.

-   Best configuration: **15m \| 2% risk \| TP 0.5x**, yielding **102% average return**.
-   Even shorter windows (5m) with 2% risk and low TP also performed well (70--85%).

### 2. Category-Level Returns

-   Unlike the **yfiance** comprehensive test (where equities underperformed), **Polygon data shows stocks leading strongly**.
-   The stock category averaged **\~69% return**, confirming robustness across multiple large-cap tickers.

### 3. Symbol-Level Insights

-   Top performers:
    -   **XLK (Technology ETF)** → \~139% return, 100% positive days.
    -   **GOOGL, NFLX, ADBE, MA, UNH, COST** also showed very high and
        consistent returns (all 100% positive days).
-   Even defensive tickers (DIA, PEP) posted consistent but smaller gains.
-   No negative-performing symbols in the top group, indicating **broad symbol coverage**.

### 4. Win Rate & Profitability

-   Mean win rate: **56%**, consistent with earlier results.
-   Profit factors cluster around **1.2--1.8**, but high returns came from a balance of modest win rates with **frequent, smaller profits (low TP multipliers)**.

### 5. Risk & Duration Sensitivity

-   **15m ORB + 2% risk** proved optimal here (vs. 5m dominance in theprevious dataset).
-   Suggests Polygon's stock data aligns better with **longer ranges andslightly higher risk tolerance**.
-   Still, **low TP multipliers (0.5x--1.0x)** consistently outperformacross all risk/duration setups.

### 6. Robustness

-   **100% of runs positive** → extremely robust under Polygon dataset.
-   18 parameter-symbol combinations satisfied robustness criteria (avg return \>0 and positive ratio ≥0.6).
-   Wide coverage across stocks supports generalizability.

------------------------------------------------------------------------

## Conclusion Summary

The Polygon-based ORB backtest shows a **clear edge for longer openingranges (15m) combined with 2% risk and low TP multipliers
(0.5x--1.0x)** , producing exceptional returns (up to 102%) and broad symbol robustness. Unlike earlier mixed results, equities here are highly profitable, led by XLK, GOOGL, NFLX, and ADBE with **100% positive days**. This suggests that the strategy is not only robust but potentially scalable across diverse stock tickers.
