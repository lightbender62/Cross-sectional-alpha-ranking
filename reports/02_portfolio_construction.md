# Portfolio construction

`src/portfolio.py` turns one date's model scores into long and short weights.

## Steps

1. **Validate scores.** Values are converted to numbers. NaN, infinite and non-numeric entries are dropped and counted (`n_dropped`). A date with no finite scores is skipped and logged.
2. **Rank.** The highest score gets rank 1. Ties are broken by input order.
3. **Select the tails.** The long leg is the top `top_pct` of names and the short leg the bottom `bottom_pct`, each `int(n × pct)` names with a minimum of one. A small epsilon protects against floating-point error (for example, 0.29 × 100 evaluates to 28.999…). If the two legs would overlap, the date is skipped and logged.
4. **Weight.** Names are equal-weighted within each leg. Long exposure is `(gross + net) / 2` and short exposure is `(gross − net) / 2`.
5. **Check.** `assert_exposure` confirms net and gross exposure match the targets within 1e-8.

## Settings used

| Setting | Value |
|---|---|
| Top and bottom fraction | 10% each |
| Gross exposure | 1.0 |
| Net exposure | 0.0 (dollar-neutral) |
| Rebalance | Monthly |

With 91 names, each leg holds 9 stocks at about 5.56% of capital each (50% long, 50% short).

## Properties

- Only the selected names carry weight; every other stock is zero.
- Weights depend only on the rank order of scores, not on their size.
- Turnover is high by design, because both legs are rebuilt from the new ranking each month (about 1.6 per month in the 2023–2024 run).