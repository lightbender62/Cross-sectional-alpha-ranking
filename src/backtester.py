from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.portfolio import (
    validate_scores,
    rank_cross_sectional,
    select_quantiles,
    assign_weights,
    assert_exposure,
)


RiskFn = Callable[
    [pd.Series, pd.DataFrame, pd.Series],
    pd.Series,
]

# Constants
TRANSACTION_COST_RATE = 0.0005
PERIODS_PER_YEAR = 12

PERIOD_COLUMNS = [
    "date",
    "portfolio_return",
    "benchmark_return",
    "active_return",
    "turnover",
    "transaction_cost",
    "net_return",
    "nav",
    "rank_ic",
    "long_return",
    "short_return",
    "n_long",
    "n_short",
    "n_scored",
    "n_dropped",
]

# Rebalance calendar
def month_end_trading_dates(dates) -> pd.DatetimeIndex:
    """Return the last available trading date of each calendar month."""
    dates = (
        pd.DatetimeIndex(pd.to_datetime(dates))
        .drop_duplicates()
        .sort_values()
    )

    if len(dates) == 0:
        raise ValueError(
            "dates must contain at least one date"
        )

    s = pd.Series(dates)

    result = pd.DatetimeIndex(
        s.groupby(
            [s.dt.year, s.dt.month]
        ).max().to_numpy()
    )

    return result


# Forward returns
def compute_forward_returns(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute return from each rebalance date t to the next rebalance date.

    The final rebalance date has no subsequent period and is therefore
    excluded from the returned DataFrame.
    """
    required = {"date", "ticker", "close"}
    missing = required - set(prices.columns)

    if missing:
        raise ValueError(
            f"prices missing columns: {sorted(missing)}"
        )

    px = prices.copy()
    px["date"] = pd.to_datetime(
        px["date"]
    )

    close = (
        px.pivot(
            index="date",
            columns="ticker",
            values="close",
        )
        .sort_index()
    )

    rb = pd.DatetimeIndex(
        pd.to_datetime(rebalance_dates)
    )

    available = rb.intersection(
        close.index
    )

    if len(available) < 2:
        raise ValueError(
            "need at least two rebalance dates with available prices"
        )

    fwd = (
        close.loc[available].shift(-1)
        / close.loc[available]
        - 1.0
    )

    # Last date has no next rebalance period.
    fwd = fwd.iloc[:-1]

    fwd.index.name = "date"

    return fwd

# Turnover
def compute_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series,
    universe,
) -> float:
    
    universe = pd.Index(universe)

    current = current_weights.reindex(
        universe,
        fill_value=0.0,
    )

    previous = previous_weights.reindex(
        universe,
        fill_value=0.0,
    )

    return float(
        (current - previous).abs().sum()
    )


# Transaction costs

def compute_transaction_cost(
    turnover: float,
    rate: float = TRANSACTION_COST_RATE,
) -> float:
    """Transaction cost = turnover * cost rate."""
    if turnover < 0:
        raise ValueError(
            "turnover cannot be negative"
        )

    if rate < 0:
        raise ValueError(
            "transaction cost rate cannot be negative"
        )

    return float(
        turnover * rate
    )


# Rank IC

def compute_rank_ic(
    scores: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """Spearman rank IC between scores and realized forward returns.
    The calculation uses the full scored cross-section, not only
    traded names. Invalid pairs are removed pairwise.
    """
    x = pd.to_numeric(
        scores,
        errors="coerce",
    )

    y = pd.to_numeric(
        forward_returns,
        errors="coerce",
    )

    data = pd.concat(
        [
            x.rename("score"),
            y.rename("return"),
        ],
        axis=1,
    )

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(data) < 2:
        return np.nan

    return float(
        spearmanr(
            data["score"],
            data["return"],
        ).statistic
    )


# Result container
@dataclass
class BacktestResult:
    """Container for backtest output."""

    periods: pd.DataFrame
    skipped: pd.DataFrame
    params: dict = field(default_factory=dict)

    def summary(
        self,
        periods_per_year: int = PERIODS_PER_YEAR,
    ) -> pd.Series:
        return summarize_performance(
            self.periods,
            periods_per_year,
        )

    def by_window(
        self,
        windows: dict[str, tuple[str, str]],
        periods_per_year: int = PERIODS_PER_YEAR,
    ) -> pd.DataFrame:
        return summarize_by_window(
            self.periods,
            windows,
            periods_per_year,
        )


# Empty result helper

def _empty_periods() -> pd.DataFrame:
    """Return an empty periods DataFrame with the standard schema."""
    return pd.DataFrame(
        columns=PERIOD_COLUMNS
    )


# Main backtest

def run_backtest(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    rebalance_dates: pd.DatetimeIndex | None = None,
    benchmark_returns: pd.Series | None = None,
    eval_start: str | pd.Timestamp | None = None,
    eval_end: str | pd.Timestamp | None = None,
    top_pct: float = 0.10,
    bottom_pct: float = 0.10,
    target_gross: float = 1.0,
    target_net: float = 0.0,
    transaction_cost_rate: float = TRANSACTION_COST_RATE,
    strict: bool = True,
    risk_fn: RiskFn | None = None,
) -> BacktestResult:
    
    # Validate score schema
    required_score_columns = {
        "date",
        "ticker",
        "score",
    }

    missing_score_columns = (
        required_score_columns
        - set(scores.columns)
    )

    if missing_score_columns:
        raise ValueError(
            "scores missing columns: "
            f"{sorted(missing_score_columns)}"
        )

    # Copy inputs and normalize dates
    prices = prices.copy()
    scores = scores.copy()

    prices["date"] = pd.to_datetime(
        prices["date"]
    )

    scores["date"] = pd.to_datetime(
        scores["date"]
    )
    # Establish rebalance calendar
    if rebalance_dates is None:
        rebalance_dates = pd.DatetimeIndex(
            sorted(
                scores["date"].unique()
            )
        )
    else:
        rebalance_dates = (
            pd.DatetimeIndex(
                pd.to_datetime(rebalance_dates)
            )
            .drop_duplicates()
            .sort_values()
        )

    # Compute forward returns using canonical price calendar
    price_calendar = month_end_trading_dates(
        prices["date"]
    )

    forward_returns = compute_forward_returns(
        prices,
        price_calendar,
    )

    # Daily returns for the risk layer.
    # These are raw historical close-to-close returns. At rebalance date t only rows with date < t are passed to risk_fn.
    close = (
        prices
        .pivot(
            index="date",
            columns="ticker",
            values="close",
        )
        .sort_index()
    )

    daily_returns = close.pct_change(
        fill_method=None
    )

    # Alignment check
    tradable = rebalance_dates.intersection(
        forward_returns.index
    )

    unmatched = rebalance_dates.difference(
        forward_returns.index
    )

    # The final price-calendar date legitimately has no forward return because there is no next rebalance.
    legit_last = price_calendar[-1:]

    bad = unmatched.difference(
        legit_last
    )

    if len(bad) > 0 and strict:
        examples = [
            d.date().isoformat()
            for d in bad[:3]
        ]

        raise ValueError(
            f"{len(bad)} score date(s) do not align with "
            f"tradable rebalance dates, e.g. {examples}. "
            "Ensure the model and backtester use the same "
            "month_end_trading_dates calendar. "
            "Pass strict=False to trade only the aligned subset."
        )

    # Evaluation window
    if eval_start is not None:
        tradable = tradable[
            tradable >= pd.Timestamp(eval_start)
        ]

    if eval_end is not None:
        tradable = tradable[
            tradable <= pd.Timestamp(eval_end)
        ]

    tradable = tradable.sort_values()

    # Benchmark
    if benchmark_returns is None:
        benchmark_returns = forward_returns.mean(
            axis=1
        )
    else:
        benchmark_returns = benchmark_returns.copy()

        benchmark_returns.index = pd.to_datetime(
            benchmark_returns.index
        )

    # Backtest state
    records: list[dict] = []
    skips: list[dict] = []

    previous_weights = pd.Series(
        dtype=float,
        name="weight",
    )

    nav = 1.0
    # Seed the running NAV at the beginning of the available price calendar. It is updated only after realized backtest P&L.
    nav_history = pd.Series(
        [nav],
        index=pd.DatetimeIndex(
            [price_calendar[0]]
        ),
        dtype=float,
        name="nav",
    )

    # Rebalance loop
    for date in tradable:

        ds = (
            scores.loc[
                scores["date"] == date,
                ["ticker", "score"],
            ]
            .drop_duplicates("ticker")
        )

        if ds.empty:
            skips.append(
                {
                    "date": date,
                    "reason": "no scores",
                }
            )
            continue

        clean_scores, n_dropped = validate_scores(
            ds.set_index("ticker")["score"]
        )

        # Rank
        ranks = rank_cross_sectional(
            clean_scores
        )

        # Selecting tails
        try:
            long_names, short_names = select_quantiles(
                ranks,
                top_pct,
                bottom_pct,
            )

        except AssertionError as exc:
            skips.append(
                {
                    "date": date,
                    "reason": f"selection failed: {exc}",
                }
            )
            continue

        # Build raw portfolio
        weights = assign_weights(
            long_names,
            short_names,
            target_gross=target_gross,
            target_net=target_net,
        )

        assert_exposure(
            weights,
            target_net=target_net,
            target_gross=target_gross,
        )

        # Optional risk layer
        # The risk function sees only daily returns dated strictly before t and NAV generated by periods completed before t.
        if risk_fn is not None:
            pre_t_daily_returns = daily_returns.loc[
                daily_returns.index < date
            ]

            transformed_weights = risk_fn(
                weights.copy(),
                pre_t_daily_returns.copy(),
                nav_history.copy(),
            )

            if not isinstance(
                transformed_weights,
                pd.Series,
            ):
                raise TypeError(
                    "risk_fn must return a pandas Series"
                )

            if not transformed_weights.index.equals(
                weights.index
            ):
                raise ValueError(
                    "risk_fn must preserve the portfolio ticker index"
                )

            transformed_weights = pd.to_numeric(
                transformed_weights,
                errors="coerce",
            )

            if not np.isfinite(
                transformed_weights
            ).all():
                raise ValueError(
                    "risk_fn returned non-finite weights"
                )

            weights = transformed_weights.astype(float)

        stock_returns = forward_returns.loc[
            date
        ]

        aligned = stock_returns.reindex(
            weights.index
        )

        if not aligned.notna().any():
            skips.append(
                {
                    "date": date,
                    "reason": (
                        "no realized returns "
                        "for traded names"
                    ),
                }
            )
            continue

        aligned = aligned.dropna()

        rw = weights.reindex(
            aligned.index
        )
        # Portfolio P&L
        portfolio_return = float(
            (rw * aligned).sum()
        )

        long_weights = rw[
            rw > 0
        ]

        short_weights = rw[
            rw < 0
        ]

        long_return = float(
            (
                long_weights
                * aligned.reindex(
                    long_weights.index
                )
            ).sum()
        )

        short_return = float(
            (
                short_weights
                * aligned.reindex(
                    short_weights.index
                )
            ).sum()
        )

        # Turnover
        universe = pd.Index(
            prices.loc[
                prices["date"] == date,
                "ticker",
            ]
            .dropna()
            .unique()
        )

        turnover = compute_turnover(
            current_weights=weights,
            previous_weights=previous_weights,
            universe=universe,
        )

        # Transaction costs
        transaction_cost = compute_transaction_cost(
            turnover,
            rate=transaction_cost_rate,
        )

        # Net return and NAV
        net_return = (
            portfolio_return
            - transaction_cost
        )

        nav *= (
            1.0 + net_return
        )

        # The forward return from date -> next_date has now been realized.Therefore NAV through next_date is available to the next rebalance's risk function.
        forward_dates = forward_returns.index

        if risk_fn is not None:
            forward_dates = forward_returns.index
            loc = forward_dates.get_loc(date)
            if loc + 1 < len(forward_dates):
                next_date = forward_dates[loc + 1]
                nav_history.loc[next_date] = nav

        rank_ic = compute_rank_ic(
            clean_scores,
            stock_returns.reindex(
                clean_scores.index
            ),
        )

        benchmark_return = benchmark_returns.get(
            date,
            np.nan,
        )

        if pd.isna(
            benchmark_return
        ):
            benchmark_return = np.nan
        else:
            benchmark_return = float(
                benchmark_return
            )

        records.append(
            {
                "date": date,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "active_return": (
                    portfolio_return
                    - benchmark_return
                    if not np.isnan(
                        benchmark_return
                    )
                    else np.nan
                ),
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "nav": nav,
                "rank_ic": rank_ic,
                "long_return": long_return,
                "short_return": short_return,
                "n_long": len(long_names),
                "n_short": len(short_names),
                "n_scored": len(clean_scores),
                "n_dropped": n_dropped,
            }
        )

        previous_weights = weights.copy()

    
    if not records:

        if len(tradable) == 0:
            raise ValueError(
                "No tradable rebalance dates after "
                "alignment/window filtering — "
                "check the calendar and eval window."
            )

        return BacktestResult(
            periods=_empty_periods(),
            skipped=(
                pd.DataFrame(skips)
                if skips
                else pd.DataFrame(
                    columns=[
                        "date",
                        "reason",
                    ]
                )
            ),
            params={
                "top_pct": top_pct,
                "bottom_pct": bottom_pct,
                "target_gross": target_gross,
                "target_net": target_net,
                "transaction_cost_rate": (
                    transaction_cost_rate
                ),
                "eval_start": eval_start,
                "eval_end": eval_end,
            },
        )

    periods = (
        pd.DataFrame(records)
        .sort_values("date")
        .reset_index(drop=True)
    )

    skipped = (
        pd.DataFrame(skips)
        .sort_values("date")
        .reset_index(drop=True)
        if skips
        else pd.DataFrame(
            columns=[
                "date",
                "reason",
            ]
        )
    )

    return BacktestResult(
        periods=periods,
        skipped=skipped,
        params={
            "top_pct": top_pct,
            "bottom_pct": bottom_pct,
            "target_gross": target_gross,
            "target_net": target_net,
            "transaction_cost_rate": (
                transaction_cost_rate
            ),
            "eval_start": eval_start,
            "eval_end": eval_end,
        },
    )

# Performance summary

def summarize_performance(
    periods: pd.DataFrame,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Calculate headline performance metrics.

    Risk-free rate is assumed to be zero.
    """

    if periods.empty:
        raise ValueError(
            "no periods to summarize"
        )

    r = periods[
        "net_return"
    ].astype(float)

    n = len(r)

    nav_final = float(
        periods["nav"].iloc[-1]
    )

    cagr = (
        nav_final
        ** (periods_per_year / n)
        - 1.0
    )

    ann_vol = (
        r.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    sharpe = (
        (r.mean() * periods_per_year)
        / ann_vol
        if ann_vol > 0
        else np.nan
    )

    downside = (
        np.sqrt(
            (
                np.minimum(r, 0.0)
                ** 2
            ).mean()
        )
        * np.sqrt(periods_per_year)
    )

    sortino = (
        (r.mean() * periods_per_year)
        / downside
        if downside > 0
        else np.nan
    )

    nav = periods[
        "nav"
    ].astype(float)

    max_drawdown = float(
        (
            nav / nav.cummax()
            - 1.0
        ).min()
    )

    calmar = (
        cagr / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )

    ic = (
        periods["rank_ic"]
        .astype(float)
        .dropna()
    )

    mean_ic = (
        float(ic.mean())
        if len(ic)
        else np.nan
    )

    icir = (
        float(
            ic.mean()
            / ic.std(ddof=1)
        )
        if (
            len(ic) > 1
            and ic.std(ddof=1) > 0
        )
        else np.nan
    )

    ic_hit_rate = (
        float(
            (ic > 0).mean()
        )
        if len(ic)
        else np.nan
    )

    mean_turnover = float(
        periods[
            "turnover"
        ].mean()
    )

    ann_cost_drag = float(
        periods[
            "transaction_cost"
        ].mean()
        * periods_per_year
    )

    active = (
        periods[
            "active_return"
        ]
        .astype(float)
        .dropna()
    )

    ann_active_return = (
        float(
            active.mean()
            * periods_per_year
        )
        if len(active)
        else np.nan
    )

    tracking_error = (
        float(
            active.std(ddof=1)
            * np.sqrt(periods_per_year)
        )
        if len(active) > 1
        else np.nan
    )

    information_ratio = (
        ann_active_return
        / tracking_error
        if (
            not np.isnan(tracking_error)
            and tracking_error > 0
        )
        else np.nan
    )

    return pd.Series(
        {
            "n_periods": n,
            "CAGR": cagr,
            "ann_return": (
                r.mean()
                * periods_per_year
            ),
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "mean_IC": mean_ic,
            "ICIR": icir,
            "IC_hit_rate": ic_hit_rate,
            "mean_turnover": mean_turnover,
            "ann_cost_drag": ann_cost_drag,
            "mean_long_return": float(
                periods[
                    "long_return"
                ].mean()
            ),
            "mean_short_return": float(
                periods[
                    "short_return"
                ].mean()
            ),
            "ann_active_return": (
                ann_active_return
            ),
            "information_ratio": (
                information_ratio
            ),
            "final_nav": nav_final,
        }
    )



def summarize_by_window(
    periods: pd.DataFrame,
    windows: dict[str, tuple[str, str]],
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
   
    out = {}

    for name, (lo, hi) in windows.items():

        sub = periods[
            (
                periods["date"]
                >= pd.Timestamp(lo)
            )
            & (
                periods["date"]
                <= pd.Timestamp(hi)
            )
        ].copy()

        if sub.empty:
            continue

        sub["nav"] = (
            1.0 + sub["net_return"]
        ).cumprod()

        out[name] = summarize_performance(
            sub,
            periods_per_year,
        )

    return pd.DataFrame(out)