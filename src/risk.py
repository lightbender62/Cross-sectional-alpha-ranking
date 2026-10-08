from __future__ import annotations
import numpy as np
import pandas as pd


DEFAULT_POSITION_CAP = 0.10
DEFAULT_TARGET_VOL = 0.15
DEFAULT_VOL_LOOKBACK = 60
DEFAULT_MIN_VOL_OBS = 20
DEFAULT_MAX_GROSS = 1.0
DEFAULT_DRAWDOWN_THRESHOLD = 0.10
DEFAULT_MIN_DRAWDOWN_SCALE = 0.25
TRADING_DAYS_PER_YEAR = 252


def _validate_weights(weights: pd.Series) -> pd.Series:
    assert isinstance(weights, pd.Series), "weights must be a pandas Series"
    assert weights.index.is_unique, "weight index must contain unique tickers"
    assert not weights.empty, "weights must be non-empty"

    weights = pd.to_numeric(weights, errors="coerce")
    assert np.isfinite(weights).all(), "weights must contain only finite values"

    return weights.astype(float)


def _validate_position_cap(max_position: float) -> None:
    assert np.isfinite(max_position), "max_position must be finite"
    assert 0 < max_position <= 1.0, "max_position must be in (0, 1]"


def _cap_one_leg(
    leg: pd.Series,
    max_position: float,
) -> pd.Series:
    if leg.empty:
        return leg.copy()

    max_abs_weight = float(leg.abs().max())

    if max_abs_weight <= max_position + 1e-12:
        return leg.copy()

    scale = max_position / max_abs_weight
    return leg * scale


def apply_position_cap(
    weights: pd.Series,
    max_position: float = DEFAULT_POSITION_CAP,
    target_net: float = 0.0,
) -> pd.Series:
    weights = _validate_weights(weights)
    _validate_position_cap(max_position)

    assert np.isfinite(target_net), "target_net must be finite"

    longs = weights[weights > 0].copy()
    shorts = weights[weights < 0].copy()

    capped_longs = _cap_one_leg(
        longs,
        max_position=max_position,
    )
    capped_shorts = _cap_one_leg(
        shorts,
        max_position=max_position,
    )

    long_capacity = float(capped_longs.sum())
    short_capacity = float(-capped_shorts.sum())

    # Feasible net range: -short_capacity <= net <= long_capacity
    feasible_net = float(
        np.clip(
            target_net,
            -short_capacity,
            long_capacity,
        )
    )

    if feasible_net >= 0:
        # long - short = feasible_net
        final_short = min(
            short_capacity,
            long_capacity - feasible_net,
        )
        final_long = final_short + feasible_net

    else:
        # long - short = feasible_net
        final_long = min(
            long_capacity,
            short_capacity + feasible_net,
        )
        final_short = final_long - feasible_net

    result = pd.Series(
        0.0,
        index=weights.index,
        dtype=float,
        name=weights.name,
    )

    if long_capacity > 0 and final_long > 0:
        long_scale = final_long / long_capacity
        result.loc[capped_longs.index] = (
            capped_longs * long_scale
        )

    if short_capacity > 0 and final_short > 0:
        short_scale = final_short / short_capacity
        result.loc[capped_shorts.index] = (
            capped_shorts * short_scale
        )

    assert np.isfinite(result).all()
    assert (
        result.abs() <= max_position + 1e-10
    ).all()

    return result


def estimate_portfolio_vol(
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    lookback: int = DEFAULT_VOL_LOOKBACK,
    min_obs: int = DEFAULT_MIN_VOL_OBS,
) -> float:
    """Estimate annualized realized portfolio volatility."""
    weights = _validate_weights(weights)

    assert isinstance(
        daily_returns,
        pd.DataFrame,
    ), "daily_returns must be a pandas DataFrame"

    assert lookback > 0, "lookback must be positive"
    assert min_obs > 1, "min_obs must be greater than 1"

    available = weights.index.intersection(
        daily_returns.columns
    )

    assert len(available) > 0, (
        "no overlapping tickers between weights and returns"
    )

    returns = daily_returns.loc[:, available].copy()
    returns = returns.apply(
        pd.to_numeric,
        errors="coerce",
    )

    aligned_weights = weights.loc[available]

    portfolio_returns = returns.mul(
        aligned_weights,
        axis=1,
    ).sum(axis=1)

    portfolio_returns = (
        portfolio_returns
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    portfolio_returns = portfolio_returns.tail(lookback)

    if len(portfolio_returns) < min_obs:
        raise ValueError(
            f"insufficient observations for volatility estimate: "
            f"{len(portfolio_returns)} < {min_obs}"
        )

    daily_vol = float(
        portfolio_returns.std(ddof=1)
    )

    if not np.isfinite(daily_vol):
        raise ValueError(
            "estimated daily volatility is not finite"
        )

    return float(
        daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def apply_vol_target(
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    target_vol: float = DEFAULT_TARGET_VOL,
    lookback: int = DEFAULT_VOL_LOOKBACK,
    min_obs: int = DEFAULT_MIN_VOL_OBS,
    max_gross: float = DEFAULT_MAX_GROSS,
) -> pd.Series:
    """
    Scale portfolio toward target volatility.

    Exposure is never increased beyond the current exposure
    and max_gross.
    """
    weights = _validate_weights(weights)

    assert np.isfinite(target_vol), (
        "target_vol must be finite"
    )
    assert target_vol > 0, (
        "target_vol must be positive"
    )

    assert np.isfinite(max_gross), (
        "max_gross must be finite"
    )
    assert max_gross >= 0, (
        "max_gross must be non-negative"
    )

    current_gross = float(
        weights.abs().sum()
    )

    if current_gross <= 0:
        return weights.copy()

    if current_gross > max_gross:
        gross_scale = max_gross / current_gross
        weights = weights * gross_scale

    estimated_vol = estimate_portfolio_vol(
        weights=weights,
        daily_returns=daily_returns,
        lookback=lookback,
        min_obs=min_obs,
    )

    if estimated_vol <= 0:
        return weights.copy()

    vol_scale = min(
        1.0,
        target_vol / estimated_vol,
    )

    scaled = weights * vol_scale

    gross = float(scaled.abs().sum())

    if gross > max_gross + 1e-12:
        scaled *= max_gross / gross

    return scaled


def current_drawdown(nav: pd.Series) -> float:
    assert isinstance(nav, pd.Series), (
        "nav must be a pandas Series"
    )
    assert not nav.empty, "nav must be non-empty"

    nav = pd.to_numeric(nav, errors="coerce")
    nav = (
        nav
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    assert not nav.empty, (
        "nav contains no finite observations"
    )
    assert (nav > 0).all(), (
        "nav must be strictly positive"
    )

    running_peak = nav.cummax()
    drawdowns = nav / running_peak - 1.0

    return float(drawdowns.iloc[-1])


def drawdown_scale(
    drawdown: float,
    threshold: float = DEFAULT_DRAWDOWN_THRESHOLD,
    min_scale: float = DEFAULT_MIN_DRAWDOWN_SCALE,
) -> float:
    
    assert np.isfinite(drawdown), (
        "drawdown must be finite"
    )
    assert drawdown <= 0, (
        "drawdown must be non-positive"
    )

    assert np.isfinite(threshold), (
        "threshold must be finite"
    )
    assert 0 < threshold < 1.0, (
        "threshold must be in (0, 1)"
    )

    assert np.isfinite(min_scale), (
        "min_scale must be finite"
    )
    assert 0 < min_scale <= 1.0, (
        "min_scale must be in (0, 1]"
    )

    if abs(drawdown) <= threshold:
        return 1.0

    scale = threshold / abs(drawdown)

    return float(
        max(
            min_scale,
            min(1.0, scale),
        )
    )


def apply_drawdown_de_risk(
    weights: pd.Series,
    nav: pd.Series,
    threshold: float = DEFAULT_DRAWDOWN_THRESHOLD,
    min_scale: float = DEFAULT_MIN_DRAWDOWN_SCALE,
) -> pd.Series:
    weights = _validate_weights(weights)

    dd = current_drawdown(nav)

    scale = drawdown_scale(
        drawdown=dd,
        threshold=threshold,
        min_scale=min_scale,
    )

    return weights * scale


def apply_risk_controls(
    raw_weights: pd.Series,
    daily_returns: pd.DataFrame,
    nav: pd.Series,
    target_net: float = 0.0,
    max_position: float = DEFAULT_POSITION_CAP,
    target_vol: float = DEFAULT_TARGET_VOL,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    min_vol_obs: int = DEFAULT_MIN_VOL_OBS,
    max_gross: float = DEFAULT_MAX_GROSS,
    drawdown_threshold: float = DEFAULT_DRAWDOWN_THRESHOLD,
    min_drawdown_scale: float = DEFAULT_MIN_DRAWDOWN_SCALE,
) -> pd.Series:
    raw_weights = _validate_weights(raw_weights)

    assert np.isfinite(target_net), (
        "target_net must be finite"
    )

    weights = apply_position_cap(
        raw_weights,
        max_position=max_position,
        target_net=target_net,
    )

    weights = apply_vol_target(
        weights,
        daily_returns=daily_returns,
        target_vol=target_vol,
        lookback=vol_lookback,
        min_obs=min_vol_obs,
        max_gross=max_gross,
    )

    weights = apply_position_cap(
        weights,
        max_position=max_position,
        target_net=target_net,
    )

    weights = apply_drawdown_de_risk(
        weights,
        nav=nav,
        threshold=drawdown_threshold,
        min_scale=min_drawdown_scale,
    )

    assert np.isfinite(weights).all()
    assert (
        weights.abs() <= max_position + 1e-10
    ).all()

    return weights


def risk_adapter(
    weights: pd.Series,
    daily_returns: pd.DataFrame,
    nav: pd.Series,
) -> pd.Series:
    return apply_risk_controls(
        raw_weights=weights,
        daily_returns=daily_returns,
        nav=nav,
        target_net=0.0,
    )