# Cross-sectional alpha ranking

Finalytics Problem 2. A supervised-learning model scores a universe of NSE large-cap stocks each month. The top-ranked stocks are held long and the bottom-ranked short, and the strategy is judged by rank IC and by a cost-adjusted long/short backtest.

## Result

On 2023–2024 (23 monthly periods, 91 stocks) the random forest shows no demonstrable edge. Rank IC was +0.006 in 2023 and −0.067 in 2024, and that is statistically indistinguishable from zero over so few months. The risk layer barely acted, so its effect is untested here. See [`reports/05_backtest_results.md`](reports/05_backtest_results.md) for the tables and the reasoning.

| | Validation 2023 | Test 2024 |
|---|---|---|
| Mean rank IC | +0.0058 | −0.0667 |
| Sharpe | 1.01 | −1.10 |
| Max drawdown | −4.5% | −12.7% |

## Pipeline

```
data.py -> features.py -> labels.py -> models.py (walk-forward) -> portfolio.py -> backtester.py (+ risk.py)
```

1. Download adjusted daily prices (`data/ohlcv.parquet`).
2. Build features (`ret_5, ret_20, ret_60, vol_20, relvol_20, rsi_14, macd_norm`) and forward-return labels at month-end rebalance dates.
3. Produce out-of-sample scores with a walk-forward model (`rf`, `logit` or `ridge`).
4. Rank scores, go long the top 10% and short the bottom 10%, equal-weighted and dollar-neutral.
5. Optionally resize positions with the risk layer, then compute net-of-cost performance (5 bps per unit of turnover).

## Quick start

Requires Python 3 and the packages below.

```
pip install numpy pandas scipy scikit-learn matplotlib yfinance pyarrow pytest
```

Run everything from the project root.

```
python src/data.py          # download prices to data/ohlcv.parquet
python run_pipeline.py      # features, model, backtest, reports
python -m pytest            # run the tests
```

`run_pipeline.py` options:

| Option | Default | Meaning |
|---|---|---|
| `--model` | `rf` | `rf`, `logit` or `ridge` |
| `--eval-start` | `2023-01-01` | Start of the evaluation window |
| `--eval-end` | `2024-12-31` | End of the evaluation window |

For example, `python run_pipeline.py --eval-start 2018-01-01` runs a longer diagnostic window. Results go to `reports/` (CSV tables) and `reports/figures/` (plots), tagged with the model and start year.

## Project structure

```
data/                  price data (ohlcv.parquet)
notebooks/
  01_eda_stats.ipynb         exploratory analysis
  02_model_comparison.ipynb  model comparison
  03_backtest_results.ipynb  backtest, risk layer and integration checks
reports/                 written reports, result tables and figures
src/
  data.py            download and load prices
  features.py        feature construction
  preprocess.py      feature preprocessing
  labels.py          rebalance dates and forward-return labels
  splits.py          train/validation/test splits
  models.py          walk-forward models and rank IC
  portfolio.py       scores -> long/short weights
  backtester.py      P&L, costs, NAV and performance metrics
  risk.py            position cap, volatility target, drawdown de-risk
  diagnostics.py     model diagnostics
tests/                   pytest suites
run_pipeline.py         end-to-end run
```

Notebooks are meant to be run from the project root or from `notebooks/`; they locate the project root themselves.

## Reports

| Report | Contents |
|---|---|
| [01_data_and_universe](reports/01_data_and_universe.md) | Data source, universe, caveats |
| [02_portfolio_construction](reports/02_portfolio_construction.md) | Ranking, tail selection, weighting |
| [03_backtester](reports/03_backtester.md) | Timeline, P&L, costs, metric definitions |
| [04_risk_layer](reports/04_risk_layer.md) | Controls and parameters |
| [05_backtest_results](reports/05_backtest_results.md) | Results and interpretation |
| [06_validation_and_limitations](reports/06_validation_and_limitations.md) | Tests, fixes, limitations, next steps |

## Limitations

- The universe is today's large-cap names applied back to 2014, so results are biased upward by survivorship.
- 23 evaluation months are too few to detect a small true IC. A longer walk-forward window is the first thing to try.
- Only proportional trading costs are modelled. Short-borrow fees and market impact are not.

Full detail is in [`reports/06_validation_and_limitations.md`](reports/06_validation_and_limitations.md).
