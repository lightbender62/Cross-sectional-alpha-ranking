# Data and universe

Daily price data for the cross-sectional alpha ranking project, downloaded with `src/data.py`.

## Source

| Item | Value |
|---|---|
| Provider | Yahoo Finance via `yfinance` |
| Frequency | Daily OHLCV |
| Adjustment | `auto_adjust=True` (split- and dividend-adjusted prices) |
| Period | 2014-06-02 to 2024-12-30 (the download end date, 2024-12-31, is exclusive) |
| Output | `data/ohlcv.parquet` with columns `date, ticker, open, high, low, close, volume` |

Rows with a missing close are dropped. Re-run `python src/data.py` to refresh the file.

## Universe

91 large-cap NSE stocks have data. `TATAMOTORS.NS` returned no data from Yahoo Finance and is excluded.

Stocks that listed after 2014 have shorter histories and join the cross-section once they have prices. For example, ADANIGREEN starts on 2018-06-18 and VBL on 2016-11-08. Stocks with a full history have 2,608 trading days.

## Use downstream

The backtester needs only `date`, `ticker` and `close`. Features and labels are built from the same table.

## Caveats

- The universe is today's large-cap names applied back to 2014. Stocks that dropped out of the index are absent, which biases results upward (survivorship bias).
- Returns come from adjusted closes. A split or dividend error in the source would show up as an extreme return, so the backtester warns when any daily move exceeds 50%.