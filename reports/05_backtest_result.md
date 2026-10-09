# Backtest results

Monthly long/short backtest of the random forest (`rf`) scores on 91 NSE stocks. Evaluation covers 2023-01 to 2024-12 (23 monthly periods): 12 in 2023 (validation) and 11 in 2024 (test). Features: `ret_5, ret_20, ret_60, vol_20, relvol_20, rsi_14, macd_norm`. Costs are 5 bps per unit of turnover.

## By window (no risk layer)

| Metric | Validation 2023 | Test 2024 |
|---|---|---|
| Periods | 12 | 11 |
| Mean rank IC | +0.0058 | −0.0667 |
| ICIR | 0.0486 | −0.5699 |
| IC hit rate | 50.0% | 27.3% |
| Sharpe | 1.01 | −1.10 |
| Sortino | 2.50 | −1.26 |
| Max drawdown | −4.5% | −12.7% |
| Annualised return | +7.3% | −12.0% |
| Mean monthly turnover | 1.66 | 1.62 |

## Baseline versus risk-adjusted (full period)

| Metric | Baseline | Risk-adjusted |
|---|---|---|
| Annualised volatility | 0.0940 | 0.0935 |
| Max drawdown | −12.7% | −12.7% |
| Mean monthly turnover | 1.64 | 1.63 |
| Sharpe | −0.20 | −0.23 |
| Sortino | −0.28 | −0.31 |
| Mean rank IC | −0.0288 | −0.0288 |
| Final NAV | 0.956 | 0.952 |

Rank IC is identical with and without risk management, as it must be.

## Interpretation

The model shows no demonstrable edge, and 23 months cannot establish one either way.

- With 91 stocks, a signal with no skill still produces a monthly IC with a standard deviation of about `1 / √90 ≈ 0.105`. The monthly ICs, which range from about −0.27 to +0.26, are consistent with that.
- The mean IC has a t-statistic of `ICIR × √T`. That is about 0.2 for 2023 and −1.9 for 2024. The 2024 value is weak evidence at best, and it is one of two windows examined after the fact.
- The annualised Sharpe of a 12-month window has a standard error of about 1, so +1.01 and −1.10 are both within noise of zero. In 2023 the positive Sharpe came with an IC of about zero, which points to luck rather than ranking skill.
- Detecting a true IC of 0.03 at t ≈ 2 would need about 49 months, and an IC of 0.02 about 110.
- Transaction costs are small relative to the noise: about 1% a year at the observed turnover.

## Reproducing

`python run_pipeline.py` reproduces these tables, saves period and summary CSVs to `reports/`, and saves figures to `reports/figures/`. `python run_pipeline.py --eval-start 2018-01-01` runs the longer-window diagnostic, which adds a `pre_2023` window.