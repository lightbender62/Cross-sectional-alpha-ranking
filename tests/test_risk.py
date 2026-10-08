import numpy as np
import pandas as pd
import pytest

from src.risk import (
    DEFAULT_DRAWDOWN_THRESHOLD,
    DEFAULT_MIN_DRAWDOWN_SCALE,
    DEFAULT_POSITION_CAP,
    DEFAULT_TARGET_VOL,
    apply_drawdown_de_risk,
    apply_position_cap,
    apply_risk_controls,
    apply_vol_target,
    current_drawdown,
    drawdown_scale,
    estimate_portfolio_vol,
)


def test_apply_position_cap_preserves_neutrality():
    weights = pd.Series(
        {
            "A": 0.40,
            "B": 0.40,
            "C": 0.20,
            "D": -0.25,
            "E": -0.25,
            "F": -0.25,
            "G": -0.25,
        }
    )

    result = apply_position_cap(
        weights,
        max_position=0.20,
        target_net=0.0,
    )

    assert np.isclose(result.sum(), 0.0)
    assert (result.abs() <= 0.20 + 1e-12).all()


def test_apply_position_cap_preserves_feasible_nonzero_target_net():
    weights = pd.Series(
        {
            "A": 0.30,
            "B": 0.20,
            "C": 0.10,
            "D": -0.15,
            "E": -0.15,
        }
    )

    result = apply_position_cap(
        weights,
        max_position=0.20,
        target_net=0.10,
    )

    assert np.isclose(result.sum(), 0.10)
    assert (result.abs() <= 0.20 + 1e-12).all()


def test_apply_position_cap_clips_infeasible_positive_target_net():
    weights = pd.Series(
        {
            "A": 0.40,
            "B": 0.40,
            "C": 0.20,
            "D": -0.125,
            "E": -0.125,
            "F": -0.125,
            "G": -0.125,
        }
    )

    result = apply_position_cap(
        weights,
        max_position=0.10,
        target_net=0.50,
    )

    assert np.isclose(result.sum(), 0.25)
    assert (result.abs() <= 0.10 + 1e-12).all()


def test_apply_position_cap_clips_infeasible_negative_target_net():
    weights = pd.Series(
        {
            "A": 0.20,
            "B": 0.20,
            "C": -0.40,
            "D": -0.20,
        }
    )

    result = apply_position_cap(
        weights,
        max_position=0.10,
        target_net=-0.50,
    )

    assert np.isclose(result.sum(), -0.15)
    assert (result.abs() <= 0.10 + 1e-12).all()


def test_apply_position_cap_never_violates_cap():
    weights = pd.Series(
        {
            "A": 10.0,
            "B": 5.0,
            "C": 2.0,
            "D": -8.0,
            "E": -4.0,
        }
    )

    result = apply_position_cap(
        weights,
        max_position=0.10,
        target_net=0.0,
    )

    assert (result.abs() <= 0.10 + 1e-12).all()


def test_apply_position_cap_handles_already_capped_weights():
    weights = pd.Series(
        {
            "A": 0.10,
            "B": 0.10,
            "C": -0.10,
            "D": -0.10,
        }
    )

    result = apply_position_cap(
        weights,
        max_position=0.10,
        target_net=0.0,
    )

    pd.testing.assert_series_equal(result, weights)


def test_apply_position_cap_preserves_relative_long_composition():
    weights = pd.Series(
        {
            "A": 0.40,
            "B": 0.20,
            "C": -0.30,
            "D": -0.30,
        }
    )

    result = apply_position_cap(
        weights,
        max_position=0.10,
        target_net=0.0,
    )

    assert np.isclose(result["A"] / result["B"], 2.0)
    assert np.isclose(result["C"] / result["D"], 1.0)


def test_apply_position_cap_rejects_invalid_cap():
    weights = pd.Series(
        {"A": 0.5, "B": -0.5}
    )

    with pytest.raises(AssertionError):
        apply_position_cap(
            weights,
            max_position=0.0,
        )

    with pytest.raises(AssertionError):
        apply_position_cap(
            weights,
            max_position=1.1,
        )


def test_estimate_portfolio_vol():
    rng = np.random.default_rng(0)

    returns = pd.DataFrame(
        rng.normal(
            0,
            0.01,
            size=(60, 2),
        ),
        columns=["A", "B"],
    )

    weights = pd.Series(
        {
            "A": 0.5,
            "B": -0.5,
        }
    )

    vol = estimate_portfolio_vol(
        weights,
        returns,
        lookback=60,
        min_obs=20,
    )

    assert np.isfinite(vol)
    assert vol >= 0


def test_estimate_portfolio_vol_requires_sufficient_observations():
    returns = pd.DataFrame(
        {
            "A": [0.01] * 10,
            "B": [0.01] * 10,
        }
    )

    weights = pd.Series(
        {
            "A": 0.5,
            "B": -0.5,
        }
    )

    with pytest.raises(ValueError):
        estimate_portfolio_vol(
            weights,
            returns,
            lookback=60,
            min_obs=20,
        )


def test_apply_vol_target_reduces_high_volatility_portfolio():
    rng = np.random.default_rng(1)

    returns = pd.DataFrame(
        rng.normal(
            0,
            0.03,
            size=(60, 2),
        ),
        columns=["A", "B"],
    )

    weights = pd.Series(
        {
            "A": 0.5,
            "B": -0.5,
        }
    )

    result = apply_vol_target(
        weights,
        daily_returns=returns,
        target_vol=0.15,
        lookback=60,
        min_obs=20,
        max_gross=1.0,
    )

    assert result.abs().sum() <= (
        weights.abs().sum() + 1e-12
    )
    assert np.isfinite(result).all()


def test_apply_vol_target_never_increases_gross():
    rng = np.random.default_rng(2)

    returns = pd.DataFrame(
        rng.normal(
            0,
            0.01,
            size=(60, 2),
        ),
        columns=["A", "B"],
    )

    weights = pd.Series(
        {
            "A": 0.5,
            "B": -0.5,
        }
    )

    result = apply_vol_target(
        weights,
        daily_returns=returns,
        target_vol=0.50,
        lookback=60,
        min_obs=20,
        max_gross=1.0,
    )

    assert result.abs().sum() <= (
        weights.abs().sum() + 1e-12
    )
    assert result.abs().sum() <= (
        1.0 + 1e-12
    )


def test_current_drawdown():
    nav = pd.Series(
        [1.0, 1.10, 1.05, 1.00]
    )

    dd = current_drawdown(nav)

    assert np.isclose(
        dd,
        1.00 / 1.10 - 1.0,
    )


def test_current_drawdown_at_peak_is_zero():
    nav = pd.Series(
        [1.0, 1.05, 1.10]
    )

    assert np.isclose(
        current_drawdown(nav),
        0.0,
    )


def test_drawdown_scale_before_threshold():
    scale = drawdown_scale(
        drawdown=-0.05,
        threshold=0.10,
        min_scale=0.25,
    )

    assert np.isclose(scale, 1.0)


def test_drawdown_scale_beyond_threshold():
    scale = drawdown_scale(
        drawdown=-0.20,
        threshold=0.10,
        min_scale=0.25,
    )

    assert np.isclose(scale, 0.50)


def test_drawdown_scale_respects_floor():
    scale = drawdown_scale(
        drawdown=-0.90,
        threshold=0.10,
        min_scale=0.25,
    )

    assert np.isclose(scale, 0.25)


def test_apply_drawdown_de_risk_reduces_exposure():
    weights = pd.Series(
        {
            "A": 0.10,
            "B": -0.10,
        }
    )

    nav = pd.Series(
        [1.0, 1.10, 0.80]
    )

    result = apply_drawdown_de_risk(
        weights,
        nav,
        threshold=0.10,
        min_scale=0.25,
    )

    assert result.abs().sum() < (
        weights.abs().sum()
    )
    assert np.sign(result["A"]) == np.sign(
        weights["A"]
    )
    assert np.sign(result["B"]) == np.sign(
        weights["B"]
    )


def test_apply_risk_controls_neutrality():
    rng = np.random.default_rng(3)

    returns = pd.DataFrame(
        rng.normal(
            0,
            0.015,
            size=(60, 4),
        ),
        columns=["A", "B", "C", "D"],
    )

    raw_weights = pd.Series(
        {
            "A": 0.25,
            "B": 0.25,
            "C": -0.25,
            "D": -0.25,
        }
    )

    nav = pd.Series(
        np.linspace(1.0, 0.95, 60)
    )

    result = apply_risk_controls(
        raw_weights=raw_weights,
        daily_returns=returns,
        nav=nav,
        target_net=0.0,
        max_position=DEFAULT_POSITION_CAP,
        target_vol=DEFAULT_TARGET_VOL,
        vol_lookback=60,
        min_vol_obs=20,
        max_gross=1.0,
        drawdown_threshold=DEFAULT_DRAWDOWN_THRESHOLD,
        min_drawdown_scale=DEFAULT_MIN_DRAWDOWN_SCALE,
    )

    assert np.isfinite(result).all()
    assert np.isclose(
        result.sum(),
        0.0,
        atol=1e-10,
    )
    assert (
        result.abs()
        <= DEFAULT_POSITION_CAP + 1e-10
    ).all()
    assert result.abs().sum() <= (
        1.0 + 1e-10
    )


def test_apply_risk_controls_preserves_direction():
    rng = np.random.default_rng(4)

    returns = pd.DataFrame(
        rng.normal(
            0,
            0.01,
            size=(60, 4),
        ),
        columns=["A", "B", "C", "D"],
    )

    raw_weights = pd.Series(
        {
            "A": 0.25,
            "B": 0.25,
            "C": -0.25,
            "D": -0.25,
        }
    )

    nav = pd.Series(
        np.ones(60)
    )

    result = apply_risk_controls(
        raw_weights=raw_weights,
        daily_returns=returns,
        nav=nav,
        target_net=0.0,
    )

    assert result["A"] >= 0
    assert result["B"] >= 0
    assert result["C"] <= 0
    assert result["D"] <= 0


def test_apply_risk_controls_reduces_on_drawdown():
    rng = np.random.default_rng(5)

    returns = pd.DataFrame(
        rng.normal(
            0,
            0.01,
            size=(60, 4),
        ),
        columns=["A", "B", "C", "D"],
    )

    raw_weights = pd.Series(
        {
            "A": 0.25,
            "B": 0.25,
            "C": -0.25,
            "D": -0.25,
        }
    )

    nav = pd.Series(
        np.concatenate(
            [
                np.linspace(1.0, 1.20, 50),
                np.linspace(1.20, 0.90, 10),
            ]
        )
    )

    result = apply_risk_controls(
        raw_weights=raw_weights,
        daily_returns=returns,
        nav=nav,
        target_net=0.0,
        drawdown_threshold=0.10,
        min_drawdown_scale=0.25,
    )

    capped_before_dd = apply_position_cap(
        raw_weights,
        max_position=DEFAULT_POSITION_CAP,
        target_net=0.0,
    )

    assert result.abs().sum() <= (
        capped_before_dd.abs().sum() + 1e-12
    )