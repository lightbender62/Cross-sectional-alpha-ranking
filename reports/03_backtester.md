# Backtester

`src/backtester.py` turns monthly scores into a net-of-cost P&L series and performance metrics.

## Timeline

- **Rebalance dates:** the last trading day of each calendar month. A final partial month (the last price date falls more than five days before month end) is dropped, so no period is shorter than a month.
- **Holding period:** each position is held from one month-end close to the next. Forward returns are `close(t+1) / close(t) − 1`.
- **No lookahead:** weights on date `t` use only scores for `t`, and the risk layer sees only daily returns dated before `t`. A score date that does not match the price calendar raises an error when `strict=True`.

## P&L

| Quantity | Definition |
|---|---|
| Portfolio return | Sum of weight × forward return over held names |
| Turnover | Sum of absolute weight changes versus the previous book, including names that left the universe |
| Transaction cost | Turnover × 0.0005 (5 bps per unit of turnover) |
| Net return | Portfolio return − transaction cost |
| NAV | Compounded net returns, starting at 1.0 |

A held name with no forward return (a data gap) counts as 0% and is recorded in `n_missing_returns`. `n_long` and `n_short` count only names that had a realised return.

## Benchmark

The default benchmark is the equal-weight mean return of all stocks multiplied by `target_net`. For a dollar-neutral book it is zero, so active return equals net return and the information ratio equals the Sharpe ratio. A custom `benchmark_returns` series can be passed instead.

## Metrics

All metrics use monthly net returns, annualised with a factor of 12, with a zero risk-free rate.

| Metric | Definition |
|---|---|
| CAGR | `NAV_final ** (12 / months spanned) − 1`, with months counted on the calendar |
| Sharpe | `mean × 12 / (std × √12)` |
| Sortino | `mean × 12 / (downside deviation × √12)` |
| Max drawdown | Worst peak-to-trough NAV decline, measured from the starting NAV of 1.0 |
| Calmar | CAGR / \|max drawdown\| |
| Rank IC | Spearman correlation of scores and forward returns over the whole scored cross-section |
| ICIR | Mean IC / standard deviation of IC |
| IC hit rate | Share of months with positive IC |

## Skipped dates

A date is skipped and logged in `result.skipped` (with a reason) if it has no scores, no finite scores, no valid tail selection, or no realised returns for the traded names.

## Risk hook

`risk_fn(weights, daily_returns, nav_history)` can resize the portfolio before returns are applied. It must return a Series with the same tickers. Rank IC depends only on the scores and realised returns, not on the weights, so risk management never changes it.