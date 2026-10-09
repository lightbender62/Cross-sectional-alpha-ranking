# Validation and limitations

## Tests

`tests/` holds pytest suites, including ones for the backtester, portfolio construction and risk modules. Run them from the project root with `python -m pytest`. They check:

- **Direction:** the top-ranked stock is held long, the bottom-ranked is held short, and a good score ranking makes money.
- **Accounting:** long plus short return equals portfolio return, net return is portfolio return minus cost, and NAV compounds net returns.
- **Exposure:** net and gross exposure match their targets, and invalid inputs are rejected.
- **Timing:** the risk hook only receives returns dated before the rebalance date, and misaligned score dates are rejected when `strict=True`.
- **Skips:** dates with no scores, or with all-NaN scores, are logged rather than crashing the run.

## Integration checks

Notebook `03_backtest_results.ipynb` also verifies two seams between the model and the backtester. The rank IC computed by the model code and by the backtester differ by 0, and the forward returns used as labels match the backtester's forward returns exactly. It further confirms that the risk layer leaves rank IC unchanged.

## Corrections made after review

An independent review of the backtester, portfolio and risk code led to these fixes:

- Active return now uses net return, and the default benchmark is scaled by `target_net`.
- Maximum drawdown now counts a loss in the first period.
- Turnover includes names that left the universe.
- Held names with no return are treated as 0% and counted in `n_missing_returns`.
- A last partial month is dropped, and CAGR counts calendar months so skipped months still count toward elapsed time.
- Quantile selection is robust to floating-point error and raises `ValueError` on overlap instead of failing silently under `python -O`.
- The volatility estimate no longer counts an all-NaN first row as a 0% day.
- A date with all-NaN scores is skipped and logged instead of stopping the run.

## Limitations

- **Survivorship bias.** The universe is today's large-cap names applied back to 2014, which flatters results.
- **Short evaluation window.** 23 months cannot separate a small true IC from noise (see the results report).
- **Risk layer barely tested.** It acted once in 23 periods, so its effect in a stressed market is untested.
- **Simplified costs.** Only proportional trading costs are modelled. Short-borrow fees, market impact and liquidity limits are not.
- **Excluded stock.** `TATAMOTORS.NS` returned no data and is missing from the universe.
- **Unresolved warning.** A divide-by-zero (log of zero) warning appears while features are built. The run completed without error and all 2,093 prediction rows were scored, but the source was not investigated.
- **Tie-breaking.** Tied scores are ranked by input order. This is unlikely to matter with continuous model scores.

## Possible next steps

- Run the 2018–2024 walk-forward diagnostic to get enough months for a meaningful IC test.
- Compute the IC of each feature on its own.
- Compare models on the longer window.
- Replace the remaining `assert` input checks with `ValueError`.