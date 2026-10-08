import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    validate_scores,
    rank_cross_sectional,
    select_quantiles,
    assign_weights,
    assert_exposure,
)


def make_scores(n=100):
    """Deterministic single-date score series."""
    tickers = [f"STOCK_{i:03d}" for i in range(n)]
    values = np.linspace(1.0, -1.0, n)

    return pd.Series(values, index=tickers, name="score")


def test_validate_scores_accepts_valid_scores():
    scores = make_scores(10)

    clean, n_dropped = validate_scores(scores)

    pd.testing.assert_series_equal(clean, scores)
    assert n_dropped == 0


def test_validate_scores_coerces_object_values_to_numeric():
    scores = pd.Series(
        ["1.0", "0.5", "bad", "-0.2"],
        index=["A", "B", "C", "D"],
        name="score",
    )

    clean, n_dropped = validate_scores(scores)

    expected = pd.Series(
        [1.0, 0.5, -0.2],
        index=["A", "B", "D"],
        name="score",
    )

    pd.testing.assert_series_equal(clean, expected)
    assert n_dropped == 1


def test_validate_scores_drops_nan_and_infinities():
    scores = pd.Series(
        [1.0, np.nan, np.inf, -np.inf, 0.5],
        index=["A", "B", "C", "D", "E"],
        name="score",
    )

    clean, n_dropped = validate_scores(scores)

    expected = pd.Series(
        [1.0, 0.5],
        index=["A", "E"],
        name="score",
    )

    pd.testing.assert_series_equal(clean, expected)
    assert n_dropped == 3


def test_validate_scores_raises_on_duplicate_tickers():
    scores = pd.Series(
        [1.0, 0.5, -0.2],
        index=["A", "A", "B"],
    )

    with pytest.raises(AssertionError):
        validate_scores(scores)


def test_validate_scores_raises_if_no_finite_scores_remain():
    scores = pd.Series(
        [np.nan, np.inf, -np.inf],
        index=["A", "B", "C"],
    )

    with pytest.raises(AssertionError):
        validate_scores(scores)


def test_validate_scores_raises_for_non_series_input():
    with pytest.raises(AssertionError):
        validate_scores([1.0, 2.0, 3.0])


def test_rank_cross_sectional_ranks_highest_score_first():
    scores = pd.Series(
        [0.2, 0.9, -0.1, 0.5],
        index=["A", "B", "C", "D"],
    )

    ranks = rank_cross_sectional(scores)

    expected = pd.Series(
        [3.0, 1.0, 4.0, 2.0],
        index=["A", "B", "C", "D"],
    )

    pd.testing.assert_series_equal(ranks, expected)


def test_rank_cross_sectional_uses_deterministic_first_tie_break():
    scores = pd.Series(
        [0.5, 0.5, 0.2],
        index=["A", "B", "C"],
    )

    ranks = rank_cross_sectional(scores)

    assert ranks["A"] == 1.0
    assert ranks["B"] == 2.0
    assert ranks["C"] == 3.0


def test_rank_cross_sectional_rejects_non_finite_scores():
    scores = pd.Series(
        [1.0, np.inf, 0.5],
        index=["A", "B", "C"],
    )

    with pytest.raises(AssertionError):
        rank_cross_sectional(scores)


def test_select_quantiles_selects_correct_deciles_for_100_names():
    ranks = pd.Series(
        np.arange(1, 101),
        index=[f"STOCK_{i:03d}" for i in range(100)],
    )

    long_names, short_names = select_quantiles(
        ranks,
        top_pct=0.10,
        bottom_pct=0.10,
    )

    assert len(long_names) == 10
    assert len(short_names) == 10

    assert set(long_names) == {f"STOCK_{i:03d}" for i in range(10)}
    assert set(short_names) == {f"STOCK_{i:03d}" for i in range(90, 100)}


def test_select_quantiles_floors_fractional_counts():
    # 73 * 10% = 7.3 -> 7 names per leg
    ranks = pd.Series(
        np.arange(1, 74),
        index=[f"STOCK_{i:03d}" for i in range(73)],
    )

    long_names, short_names = select_quantiles(
        ranks,
        top_pct=0.10,
        bottom_pct=0.10,
    )

    assert len(long_names) == 7
    assert len(short_names) == 7


def test_select_quantiles_selects_at_least_one_name_per_leg():
    # 9 * 10% = 0.9 -> int() = 0 -> max(1, 0) = 1
    ranks = pd.Series(
        np.arange(1, 10),
        index=[f"STOCK_{i:03d}" for i in range(9)],
    )

    long_names, short_names = select_quantiles(
        ranks,
        top_pct=0.10,
        bottom_pct=0.10,
    )

    assert len(long_names) == 1
    assert len(short_names) == 1


def test_select_quantiles_rejects_overlapping_quantiles():
    ranks = pd.Series(
        np.arange(1, 11),
        index=[f"STOCK_{i:03d}" for i in range(10)],
    )

    with pytest.raises(AssertionError):
        select_quantiles(
            ranks,
            top_pct=0.60,
            bottom_pct=0.60,
        )


def test_select_quantiles_returns_disjoint_sets():
    ranks = make_scores(100).rank(
        method="first",
        ascending=False,
    )

    long_names, short_names = select_quantiles(
        ranks,
        top_pct=0.10,
        bottom_pct=0.10,
    )

    assert set(long_names).isdisjoint(short_names)


def test_select_quantiles_rejects_invalid_percentages():
    ranks = pd.Series(
        np.arange(1, 11),
        index=[f"STOCK_{i:03d}" for i in range(10)],
    )

    with pytest.raises(AssertionError):
        select_quantiles(
            ranks,
            top_pct=0.0,
            bottom_pct=0.10,
        )

    with pytest.raises(AssertionError):
        select_quantiles(
            ranks,
            top_pct=0.10,
            bottom_pct=0.0,
        )

    with pytest.raises(AssertionError):
        select_quantiles(
            ranks,
            top_pct=1.1,
            bottom_pct=0.10,
        )

def test_assign_weights_uses_traded_names_only():
    long_names = ["A", "B"]
    short_names = ["C", "D"]

    weights = assign_weights(long_names, short_names)

    assert set(weights.index) == {"A", "B", "C", "D"}
    assert not any(weight == 0.0 for weight in weights)


def test_assign_weights_equal_weights_within_each_leg():
    long_names = ["A", "B", "C"]
    short_names = ["D", "E"]

    weights = assign_weights(long_names, short_names)

    assert np.allclose(weights.loc[long_names], 0.5 / 3)
    assert np.allclose(weights.loc[short_names], -0.5 / 2)


def test_assign_weights_has_target_net_and_gross_exposure():
    long_names = ["A", "B", "C"]
    short_names = ["D", "E", "F", "G"]

    weights = assign_weights(long_names, short_names)

    assert_exposure(weights)


def test_assign_weights_handles_unequal_leg_sizes():
    long_names = ["A", "B"]
    short_names = ["C", "D", "E", "F", "G"]

    weights = assign_weights(long_names, short_names)

    assert np.isclose(weights.loc[long_names].sum(), 0.5)
    assert np.isclose(weights.loc[short_names].sum(), -0.5)

    assert np.isclose(weights.sum(), 0.0)
    assert np.isclose(weights.abs().sum(), 1.0)


def test_assign_weights_supports_nonzero_target_net():
    long_names = ["A", "B"]
    short_names = ["C", "D"]

    # Gross = 1, Net = 0.2
    # Long = 0.6, Short = -0.4
    weights = assign_weights(
        long_names,
        short_names,
        target_gross=1.0,
        target_net=0.2,
    )

    assert np.isclose(weights.loc[long_names].sum(), 0.6)
    assert np.isclose(weights.loc[short_names].sum(), -0.4)

    assert_exposure(
        weights,
        target_net=0.2,
        target_gross=1.0,
    )


def test_assign_weights_rejects_overlapping_legs():
    with pytest.raises(AssertionError):
        assign_weights(
            ["A", "B"],
            ["B", "C"],
        )


def test_assign_weights_rejects_duplicate_long_names():
    with pytest.raises(AssertionError):
        assign_weights(
            ["A", "A"],
            ["B"],
        )

def test_assign_weights_rejects_duplicate_short_names():
    with pytest.raises(AssertionError):
        assign_weights(
            ["A"],
            ["B", "B"],
        )

def test_assign_weights_rejects_empty_long_leg():
    with pytest.raises(AssertionError):
        assign_weights([], ["A"])


def test_assign_weights_rejects_empty_short_leg():
    with pytest.raises(AssertionError):
        assign_weights(["A"], [])


def test_assign_weights_rejects_invalid_net_exposure():
    with pytest.raises(AssertionError):
        assign_weights(
            ["A"],
            ["B"],
            target_gross=1.0,
            target_net=1.1,
        )

def test_assert_exposure_passes_for_valid_gross_one_market_neutral_book():
    weights = pd.Series(
        [0.25, 0.25, -0.25, -0.25],
        index=["A", "B", "C", "D"],
    )

    assert_exposure(
        weights,
        target_net=0.0,
        target_gross=1.0,
    )


def test_assert_exposure_checks_net_exposure():
    weights = pd.Series(
        [0.35, 0.35, -0.35, -0.35],
        index=["A", "B", "C", "D"],
    )

    # Gross = 1.4 and net = 0.
    # Net alone would pass, but gross must fail.
    with pytest.raises(AssertionError):
        assert_exposure(
            weights,
            target_net=0.0,
            target_gross=1.0,
        )


def test_assert_exposure_checks_gross_exposure():
    weights = pd.Series(
        [0.60, -0.40],
        index=["A", "B"],
    )

    # Net = 0.2, gross = 1.0.
    with pytest.raises(AssertionError):
        assert_exposure(
            weights,
            target_net=0.0,
            target_gross=1.0,
        )


def test_assert_exposure_respects_tolerance():
    weights = pd.Series(
        [0.5 + 1e-10, -0.5],
        index=["A", "B"],
    )

    assert_exposure(
        weights,
        target_net=1e-10,
        target_gross=0.9999999999,
        tol=1e-8,
    )


def test_assert_exposure_rejects_non_finite_weights():
    weights = pd.Series(
        [0.5, np.inf, -0.5],
        index=["A", "B", "C"],
    )

    with pytest.raises(AssertionError):
        assert_exposure(weights)


def test_assert_exposure_rejects_duplicate_tickers():
    weights = pd.Series(
        [0.5, -0.5],
        index=["A", "A"],
    )

    with pytest.raises(AssertionError):
        assert_exposure(weights)



def test_pipeline_direction_end_to_end():
    scores = pd.Series(
        {
            "BEST": 0.9,
            "MID": 0.1,
            "BAD": -0.2,
            "WORST": -0.8,
            "X": 0.0,
        }
    )

    clean, _ = validate_scores(scores)
    ranks = rank_cross_sectional(clean)
    longs, shorts = select_quantiles(
        ranks,
        0.2,
        0.2,
    )

    weights = assign_weights(longs, shorts)

    assert weights["BEST"] > 0
    assert weights["WORST"] < 0