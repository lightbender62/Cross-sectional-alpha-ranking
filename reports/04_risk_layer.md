# Risk layer

`src/risk.py` rescales the portfolio after construction and before returns are applied. It changes position sizes only. It never changes which stocks are held or their long/short direction.

## Controls, in order

1. **Position cap.** If the largest weight in a leg exceeds the cap, the whole leg is scaled down proportionally. Net exposure is then restored toward the target where the caps allow it.
2. **Volatility target.** Realised portfolio volatility is estimated from the last 60 trading days of daily returns, using the current weights and annualised with √252. If it exceeds the target, all weights are scaled by `target_vol / estimated_vol`. The layer never increases exposure above what it was given.
3. **Position cap again,** to remove any breach after scaling.
4. **Drawdown de-risk.** If the strategy's drawdown exceeds the threshold, weights are scaled by `threshold / |drawdown|`, with a floor on the scale.

## Parameters

| Parameter | Value |
|---|---|
| Position cap | 10% per name |
| Target volatility | 15% annualised |
| Volatility lookback | 60 trading days (at least 20 observations) |
| Gross exposure limit | 1.0 |
| Drawdown threshold | 10% |
| Minimum drawdown scale | 0.25 |

## Timing

The layer sees daily returns dated strictly before the rebalance date and the strategy NAV through that date.

## Behaviour in the 2023–2024 run

The layer changed sizing in 1 of 23 periods. Strategy volatility (about 9% a year) stayed well below the 15% target, the 10% cap was never close to binding with 5.6% positions, and the drawdown rule triggered only in the final month. Volatility moved from 0.0940 to 0.0935 and maximum drawdown was unchanged at −12.7%. This run therefore says little about whether the layer works. A longer sample that includes a stress period, such as 2020, would test it properly.

## Limits

- The target volatility is a one-sided cap. The layer reduces risk but never adds leverage.
- `risk_adapter` fixes `target_net` at 0.0. A run with a non-zero net target needs `functools.partial(risk_adapter, target_net=...)`.