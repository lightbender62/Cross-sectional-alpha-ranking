import numpy as np
import pandas as pd


def validate_scores(scores: pd.Series) -> tuple[pd.Series, int]:
    assert isinstance(scores, pd.Series), "scores must be a pandas Series"
    assert scores.index.is_unique, "score index must contain unique tickers"
    # Coerce to numeric at the boundary; non-numeric -> NaN -> dropped below.
    scores = pd.to_numeric(scores, errors="coerce")
    original_n = len(scores)
    scores = scores[np.isfinite(scores)]
    n_dropped = original_n - len(scores)
    assert not scores.empty, "no finite scores remain after cleaning"
    return scores, n_dropped


def rank_cross_sectional(scores: pd.Series) -> pd.Series:
    
    assert isinstance(scores, pd.Series), "scores must be a pandas Series"
    assert not scores.empty, "scores must be non-empty"
    assert scores.index.is_unique, "score index must contain unique tickers"
    assert np.isfinite(scores).all(), "scores must contain only finite values"

    return scores.rank(method="first", ascending=False)


def select_quantiles(
    ranks: pd.Series,
    top_pct: float,
    bottom_pct: float,
) -> tuple[list, list]:
   
    assert isinstance(ranks, pd.Series), "ranks must be a pandas Series"
    assert not ranks.empty, "ranks must be non-empty"
    assert ranks.index.is_unique, "rank index must contain unique tickers"

    assert 0 < top_pct <= 1, "top_pct must be in (0, 1]"
    assert 0 < bottom_pct <= 1, "bottom_pct must be in (0, 1]"

    n = len(ranks)

    # Small epsilon so float error (e.g. 0.29 * 100 = 28.999999999999996) does not truncate down.
    top_n = max(1, int(n * top_pct + 1e-9))
    bottom_n = max(1, int(n * bottom_pct + 1e-9))

    # ValueError, not assert: the backtester relies on this to skip a date, and asserts vanish under python -O.
    if top_n + bottom_n > n:
        raise ValueError(
            "top and bottom quantile selections overlap; "
            "reduce top_pct and/or bottom_pct"
        )

    long_names = ranks.nsmallest(top_n).index.tolist()
    short_names = ranks.nlargest(bottom_n).index.tolist()

    assert not set(long_names) & set(short_names), (
        "long and short selections must not overlap"
    )

    return long_names, short_names


def assign_weights(
    long_names,
    short_names,
    target_gross: float = 1.0,
    target_net: float = 0.0,
) -> pd.Series:
    
    assert target_gross >= 0, "target_gross must be non-negative"
    assert abs(target_net) <= target_gross, (
        "target_net magnitude cannot exceed target_gross"
    )

    long_names = list(long_names)
    short_names = list(short_names)

    assert long_names, "long_names must be non-empty"
    assert short_names, "short_names must be non-empty"
    assert len(set(long_names)) == len(long_names), (
        "long_names must contain unique tickers"
    )
    assert len(set(short_names)) == len(short_names), (
        "short_names must contain unique tickers"
    )
    assert not set(long_names) & set(short_names), (
        "long and short names must be disjoint"
    )

    long_exposure = (target_gross + target_net) / 2
    short_exposure = (target_gross - target_net) / 2

    long_weight = long_exposure / len(long_names)
    short_weight = -short_exposure / len(short_names)

    weights = pd.Series(
        long_weight,
        index=long_names,
        dtype=float,
        name="weight",
    )

    short_weights = pd.Series(
        short_weight,
        index=short_names,
        dtype=float,
        name="weight",
    )

    weights = pd.concat([weights, short_weights])

    return weights


def assert_exposure(
    weights: pd.Series,
    target_net: float = 0.0,
    target_gross: float = 1.0,
    tol: float = 1e-8,
) -> None:
   
    assert isinstance(weights, pd.Series), "weights must be a pandas Series"
    assert not weights.empty, "weights must be non-empty"
    assert weights.index.is_unique, "weight index must contain unique tickers"
    assert np.isfinite(weights).all(), "weights must contain only finite values"
    assert tol >= 0, "tol must be non-negative"

    net_exposure = weights.sum()
    gross_exposure = weights.abs().sum()

    assert np.isclose(net_exposure, target_net, atol=tol, rtol=0), (
        f"net exposure {net_exposure:.12f} != "
        f"target {target_net:.12f}"
    )

    assert np.isclose(gross_exposure, target_gross, atol=tol, rtol=0), (
        f"gross exposure {gross_exposure:.12f} != "
        f"target {target_gross:.12f}"
    )