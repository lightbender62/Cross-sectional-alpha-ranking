import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    month_end_trading_dates,
    compute_forward_returns,
    compute_turnover,
    compute_transaction_cost,
    compute_rank_ic,
    run_backtest,
    summarize_performance,
    summarize_by_window,
)

def make_monthly_prices():
    """Simple 3-month, 4-stock price dataset."""
    dates = pd.to_datetime(
        [
            "2023-01-31",
            "2023-02-28",
            "2023-03-31",
        ]
    )

    prices = {
        "A": [100.0, 110.0, 121.0],
        "B": [100.0, 105.0, 110.25],
        "C": [100.0, 95.0, 90.25],
        "D": [100.0, 90.0, 81.0],
    }

    rows = []

    for ticker, values in prices.items():
        for date, close in zip(dates, values):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "close": close,
                }
            )

    return pd.DataFrame(rows)


def make_scores():
    """Scores that deliberately rank A highest and D lowest."""
    dates = pd.to_datetime(
        [
            "2023-01-31",
            "2023-02-28",
        ]
    )

    rows = []

    for date in dates:
        rows.extend(
            [
                {
                    "date": date,
                    "ticker": "A",
                    "score": 4.0,
                },
                {
                    "date": date,
                    "ticker": "B",
                    "score": 3.0,
                },
                {
                    "date": date,
                    "ticker": "C",
                    "score": 2.0,
                },
                {
                    "date": date,
                    "ticker": "D",
                    "score": 1.0,
                },
            ]
        )

    return pd.DataFrame(rows)


# month end trading dates

def test_month_end_trading_dates_returns_last_date_of_each_month():
    dates = pd.to_datetime(
        [
            "2023-01-02",
            "2023-01-30",
            "2023-01-31",
            "2023-02-01",
            "2023-02-27",
            "2023-02-28",
        ]
    )

    result = month_end_trading_dates(dates)

    expected = pd.DatetimeIndex(
        [
            "2023-01-31",
            "2023-02-28",
        ]
    )

    pd.testing.assert_index_equal(result, expected)


def test_month_end_trading_dates_handles_non_calendar_month_end():
    dates = pd.to_datetime(
        [
            "2023-04-03",
            "2023-04-28",
            "2023-05-01",
            "2023-05-31",
        ]
    )

    result = month_end_trading_dates(dates)

    expected = pd.DatetimeIndex(
        [
            "2023-04-28",
            "2023-05-31",
        ]
    )

    pd.testing.assert_index_equal(result, expected)


def test_month_end_trading_dates_rejects_empty_input():
    with pytest.raises(ValueError):
        month_end_trading_dates([])


def test_compute_forward_returns_uses_next_rebalance_date():
    prices = make_monthly_prices()

    rb_dates = pd.DatetimeIndex(
        [
            "2023-01-31",
            "2023-02-28",
            "2023-03-31",
        ]
    )

    result = compute_forward_returns(
        prices,
        rb_dates,
    )

    # January -> February
    assert np.isclose(
        result.loc["2023-01-31", "A"],
        0.10,
    )

    assert np.isclose(
        result.loc["2023-01-31", "D"],
        -0.10,
    )

    # February -> March
    assert np.isclose(
        result.loc["2023-02-28", "A"],
        0.10,
    )

    assert np.isclose(
        result.loc["2023-02-28", "D"],
        -0.10,
    )


def test_compute_forward_returns_excludes_last_rebalance():
    prices = make_monthly_prices()

    rb_dates = pd.DatetimeIndex(
        [
            "2023-01-31",
            "2023-02-28",
            "2023-03-31",
        ]
    )

    result = compute_forward_returns(
        prices,
        rb_dates,
    )

    assert list(result.index) == [
        pd.Timestamp("2023-01-31"),
        pd.Timestamp("2023-02-28"),
    ]


def test_compute_forward_returns_rejects_missing_columns():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-31"]
            ),
            "ticker": ["A"],
        }
    )

    with pytest.raises(ValueError):
        compute_forward_returns(
            prices,
            pd.DatetimeIndex(
                [
                    "2023-01-31",
                    "2023-02-28",
                ]
            ),
        )


def test_compute_forward_returns_requires_two_dates():
    prices = make_monthly_prices()

    with pytest.raises(ValueError):
        compute_forward_returns(
            prices,
            pd.DatetimeIndex(
                ["2023-01-31"]
            ),
        )


def test_compute_turnover_from_zero_initial_book():
    current = pd.Series(
        [0.25, 0.25, -0.25, -0.25],
        index=["A", "B", "C", "D"],
    )

    previous = pd.Series(
        dtype=float
    )

    turnover = compute_turnover(
        current,
        previous,
        ["A", "B", "C", "D"],
    )

    assert np.isclose(
        turnover,
        1.0,
    )


def test_compute_turnover_matches_absolute_weight_changes():
    current = pd.Series(
        [0.25, -0.25],
        index=["A", "B"],
    )

    previous = pd.Series(
        [0.50, -0.50],
        index=["A", "B"],
    )

    turnover = compute_turnover(
        current,
        previous,
        ["A", "B"],
    )

    assert np.isclose(
        turnover,
        0.50,
    )


def test_compute_turnover_treats_missing_names_as_zero():
    current = pd.Series(
        [0.50],
        index=["A"],
    )

    previous = pd.Series(
        [-0.50],
        index=["B"],
    )

    turnover = compute_turnover(
        current,
        previous,
        ["A", "B"],
    )

    assert np.isclose(
        turnover,
        1.0,
    )


def test_compute_turnover_counts_names_outside_universe():
    current = pd.Series(
        [0.50],
        index=["A"],
    )

    # C was held before but is no longer in the priced universe.
    previous = pd.Series(
        [-0.50],
        index=["C"],
    )

    turnover = compute_turnover(
        current,
        previous,
        ["A"],
    )

    assert np.isclose(
        turnover,
        1.0,
    )


# ---------------------------------------------------------------------------
# compute_transaction_cost
# ---------------------------------------------------------------------------

def test_compute_transaction_cost_uses_turnover_times_rate():
    cost = compute_transaction_cost(
        turnover=2.0,
        rate=0.0005,
    )

    assert np.isclose(
        cost,
        0.001,
    )


def test_compute_transaction_cost_zero_turnover():
    cost = compute_transaction_cost(
        turnover=0.0,
        rate=0.0005,
    )

    assert np.isclose(
        cost,
        0.0,
    )


def test_compute_transaction_cost_rejects_negative_turnover():
    with pytest.raises(ValueError):
        compute_transaction_cost(-1.0)


def test_compute_transaction_cost_rejects_negative_rate():
    with pytest.raises(ValueError):
        compute_transaction_cost(
            1.0,
            rate=-0.0005,
        )


def test_compute_rank_ic_is_one_for_perfect_positive_ranking():
    scores = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=["A", "B", "C", "D"],
    )

    returns = pd.Series(
        [0.01, 0.02, 0.03, 0.04],
        index=["A", "B", "C", "D"],
    )

    ic = compute_rank_ic(
        scores,
        returns,
    )

    assert np.isclose(
        ic,
        1.0,
    )


def test_compute_rank_ic_is_negative_one_for_reversed_ranking():
    scores = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=["A", "B", "C", "D"],
    )

    returns = pd.Series(
        [0.04, 0.03, 0.02, 0.01],
        index=["A", "B", "C", "D"],
    )

    ic = compute_rank_ic(
        scores,
        returns,
    )

    assert np.isclose(
        ic,
        -1.0,
    )


def test_compute_rank_ic_drops_non_finite_pairs():
    scores = pd.Series(
        [1.0, 2.0, 3.0],
        index=["A", "B", "C"],
    )

    returns = pd.Series(
        [0.01, np.nan, 0.03],
        index=["A", "B", "C"],
    )

    ic = compute_rank_ic(
        scores,
        returns,
    )

    assert np.isfinite(ic)


def test_compute_rank_ic_returns_nan_with_fewer_than_two_pairs():
    scores = pd.Series(
        [1.0, np.nan],
        index=["A", "B"],
    )

    returns = pd.Series(
        [0.01, 0.02],
        index=["A", "B"],
    )

    ic = compute_rank_ic(
        scores,
        returns,
    )

    assert np.isnan(ic)


def test_summarize_performance_compounds_nav():
    periods = pd.DataFrame(
        {
            "net_return": [0.01, 0.02],
            "nav": [1.01, 1.01 * 1.02],
            "rank_ic": [0.2, 0.4],
            "turnover": [1.0, 0.5],
            "transaction_cost": [0.0005, 0.00025],
            "long_return": [0.03, 0.04],
            "short_return": [0.02, 0.01],
            "benchmark_return": [0.0, 0.0],
            "active_return": [0.01, 0.02],
        }
    )

    summary = summarize_performance(
        periods
    )

    assert summary["n_periods"] == 2

    assert np.isclose(
        summary["final_nav"],
        1.01 * 1.02,
    )

    assert np.isclose(
        summary["mean_IC"],
        0.3,
    )

    assert np.isclose(
        summary["mean_turnover"],
        0.75,
    )


def test_summarize_performance_max_drawdown():
    periods = pd.DataFrame(
        {
            "net_return": [0.10, -0.20, 0.05],
            "nav": [1.10, 0.88, 0.924],
            "rank_ic": [0.1, 0.1, 0.1],
            "turnover": [1.0, 1.0, 1.0],
            "transaction_cost": [0.0, 0.0, 0.0],
            "long_return": [0.0, 0.0, 0.0],
            "short_return": [0.0, 0.0, 0.0],
            "benchmark_return": [0.0, 0.0, 0.0],
            "active_return": [0.0, 0.0, 0.0],
        }
    )

    summary = summarize_performance(
        periods
    )

    assert np.isclose(
        summary["max_drawdown"],
        -0.20,
    )


def test_summarize_performance_max_drawdown_counts_first_period_loss():
    periods = pd.DataFrame(
        {
            "net_return": [-0.10, 0.10],
            "nav": [0.90, 0.99],
            "rank_ic": [0.1, 0.1],
            "turnover": [1.0, 1.0],
            "transaction_cost": [0.0, 0.0],
            "long_return": [0.0, 0.0],
            "short_return": [0.0, 0.0],
            "benchmark_return": [0.0, 0.0],
            "active_return": [0.0, 0.0],
        }
    )

    summary = summarize_performance(
        periods
    )

    # Measured from the starting NAV of 1.0, not from the first period's NAV.
    assert np.isclose(
        summary["max_drawdown"],
        -0.10,
    )


def test_summarize_by_window_returns_requested_windows():
    periods = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-31",
                    "2023-02-28",
                    "2024-01-31",
                    "2024-02-29",
                ]
            ),
            "net_return": [0.01, 0.02, 0.03, -0.01],
            "nav": [1.01, 1.0302, 1.060, 1.0494],
            "rank_ic": [0.1, 0.2, 0.3, 0.4],
            "turnover": [1.0, 1.0, 1.0, 1.0],
            "transaction_cost": [0.0, 0.0, 0.0, 0.0],
            "long_return": [0.01, 0.02, 0.03, 0.01],
            "short_return": [0.00, 0.00, 0.00, -0.02],
            "benchmark_return": [0.0, 0.0, 0.0, 0.0],
            "active_return": [0.01, 0.02, 0.03, -0.01],
        }
    )

    windows = {
        "validation_2023": (
            "2023-01-01",
            "2023-12-31",
        ),
        "test_2024": (
            "2024-01-01",
            "2024-12-31",
        ),
    }

    result = summarize_by_window(
        periods,
        windows,
    )

    assert "validation_2023" in result.columns
    assert "test_2024" in result.columns


def test_run_backtest_produces_expected_period_columns():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
    )

    expected_columns = {
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
        "n_missing_returns",
    }

    assert expected_columns.issubset(
        result.periods.columns
    )


def test_run_backtest_uses_correct_long_short_direction():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
    )

    first = result.periods.iloc[0]

    # A is long and rises 10%.
    # D is short and falls 10%.
    # Gross = 1, so each leg has 50% exposure.
    # Portfolio return = 0.5*0.10 + (-0.5)*(-0.10) = 0.10.
    assert np.isclose(
        first["portfolio_return"],
        0.10,
    )

    assert first["long_return"] > 0
    assert first["short_return"] > 0


def test_run_backtest_first_period_turnover_is_one():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0005,
    )

    first = result.periods.iloc[0]

    assert np.isclose(
        first["turnover"],
        1.0,
    )

    assert np.isclose(
        first["transaction_cost"],
        0.0005,
    )


def test_run_backtest_cost_is_subtracted_from_gross_return():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0005,
    )

    first = result.periods.iloc[0]

    assert np.isclose(
        first["net_return"],
        first["portfolio_return"]
        - first["transaction_cost"],
    )


def test_run_backtest_nav_compounds_net_returns():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
    )

    expected_nav_1 = (
        1.0
        + result.periods.iloc[0]["net_return"]
    )

    assert np.isclose(
        result.periods.iloc[0]["nav"],
        expected_nav_1,
    )

    expected_nav_2 = (
        expected_nav_1
        * (
            1.0
            + result.periods.iloc[1]["net_return"]
        )
    )

    assert np.isclose(
        result.periods.iloc[1]["nav"],
        expected_nav_2,
    )


def test_run_backtest_default_benchmark_is_scaled_by_target_net():
    dates = pd.to_datetime(
        [
            "2023-01-31",
            "2023-02-28",
            "2023-03-31",
        ]
    )

    closes = {
        "A": [100.0, 120.0, 132.0],
        "B": [100.0, 110.0, 115.5],
        "C": [100.0, 100.0, 95.0],
        "D": [100.0, 90.0, 81.0],
    }

    prices = pd.DataFrame(
        [
            {"date": d, "ticker": t, "close": c}
            for t, values in closes.items()
            for d, c in zip(dates, values)
        ]
    )

    kwargs = dict(
        prices=prices,
        scores=make_scores(),
        rebalance_dates=pd.DatetimeIndex(dates),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
    )

    neutral = run_backtest(**kwargs)
    tilted = run_backtest(**kwargs, target_net=0.2)

    # Equal-weight Jan -> Feb return = (20% + 10% + 0% - 10%) / 4 = 5%.
    # A dollar-neutral book has no market exposure, so its benchmark is 0.
    assert np.isclose(
        neutral.periods.iloc[0]["benchmark_return"],
        0.0,
    )

    assert np.isclose(
        tilted.periods.iloc[0]["benchmark_return"],
        0.2 * 0.05,
    )


def test_run_backtest_active_return_is_net_return_minus_benchmark():
    benchmark = pd.Series(
        [0.03, 0.01],
        index=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
            ]
        ),
    )

    result = run_backtest(
        prices=make_monthly_prices(),
        scores=make_scores(),
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        benchmark_returns=benchmark,
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0005,
    )

    periods = result.periods

    assert np.allclose(
        periods["active_return"],
        periods["net_return"]
        - periods["benchmark_return"],
    )


def test_run_backtest_counts_held_names_with_missing_returns():
    prices = make_monthly_prices()

    # D is the short name. Dropping its February price makes its January return missing.
    prices = prices[
        ~(
            (prices["ticker"] == "D")
            & (prices["date"] == pd.Timestamp("2023-02-28"))
        )
    ]

    result = run_backtest(
        prices=prices,
        scores=make_scores(),
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
    )

    first = result.periods.iloc[0]

    # The missing return counts as 0%, so only A's 50% * 10% contributes.
    assert first["n_missing_returns"] == 1
    assert first["n_short"] == 0
    assert np.isclose(
        first["portfolio_return"],
        0.05,
    )


def test_run_backtest_computes_positive_rank_ic_for_good_scores():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
    )

    assert (
        result.periods.iloc[0]["rank_ic"]
        > 0
    )


def test_run_backtest_records_no_skipped_dates_for_valid_input():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
    )

    assert result.skipped.empty


def test_run_backtest_rejects_misaligned_dates_in_strict_mode():
    prices = make_monthly_prices()

    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-01-31",
                    "2023-02-28",
                ]
            ),
            "ticker": [
                "A",
                "A",
                "A",
            ],
            "score": [
                1.0,
                1.0,
                1.0,
            ],
        }
    )

    with pytest.raises(ValueError):
        run_backtest(
            prices=prices,
            scores=scores,
            rebalance_dates=pd.DatetimeIndex(
                [
                    "2023-01-15",
                    "2023-01-31",
                    "2023-02-28",
                ]
            ),
            strict=True,
        )


def test_run_backtest_strict_false_trades_aligned_subset():
    prices = make_monthly_prices()

    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-31",
                    "2023-01-31",
                    "2023-01-31",
                    "2023-01-31",
                ]
            ),
            "ticker": [
                "A",
                "B",
                "C",
                "D",
            ],
            "score": [
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-15",
                "2023-01-31",
                "2023-02-28",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        strict=False,
    )

    assert len(result.periods) == 1

    assert (
        result.periods.iloc[0]["date"]
        == pd.Timestamp("2023-01-31")
    )

    # February is aligned but has no scores, so the skip log must explain it.
    assert len(result.skipped) == 1

    assert (
        result.skipped.iloc[0]["date"]
        == pd.Timestamp("2023-02-28")
    )

    assert (
        result.skipped.iloc[0]["reason"]
        == "no scores"
    )


def test_run_backtest_logs_no_score_dates_in_skipped():
    prices = make_monthly_prices()

    # January has a valid cross-section.
    # February has no scores and must therefore be logged as skipped.
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-31",
                    "2023-01-31",
                    "2023-01-31",
                    "2023-01-31",
                ]
            ),
            "ticker": [
                "A",
                "B",
                "C",
                "D",
            ],
            "score": [
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        strict=True,
    )

    assert len(result.periods) == 1
    assert len(result.skipped) == 1

    assert (
        result.skipped.iloc[0]["date"]
        == pd.Timestamp("2023-02-28")
    )

    assert (
        result.skipped.iloc[0]["reason"]
        == "no scores"
    )


def test_run_backtest_records_n_dropped_for_nan_score():
    prices = make_monthly_prices()

    scores = make_scores()

    mask = (
        (scores["date"] == pd.Timestamp("2023-01-31"))
        & (scores["ticker"] == "B")
    )

    scores.loc[mask, "score"] = np.nan

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
    )

    first = result.periods.iloc[0]

    assert first["n_dropped"] == 1
    assert first["n_scored"] == 3


def test_run_backtest_rejects_missing_score_column():
    prices = make_monthly_prices()

    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-31"]
            ),
            "ticker": ["A"],
        }
    )

    with pytest.raises(ValueError):
        run_backtest(
            prices=prices,
            scores=scores,
        )


def test_run_backtest_eval_window_restricts_periods():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        eval_start="2023-02-01",
        eval_end="2023-02-28",
        top_pct=0.25,
        bottom_pct=0.25,
    )

    assert len(result.periods) == 1

    assert (
        result.periods.iloc[0]["date"]
        == pd.Timestamp("2023-02-28")
    )

def test_backtest_direction_guard_end_to_end():
    prices = make_monthly_prices()

    # Only January scores are required because February is the forward-return endpoint.
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-31",
                    "2023-01-31",
                    "2023-01-31",
                    "2023-01-31",
                ]
            ),
            "ticker": [
                "A",
                "B",
                "C",
                "D",
            ],
            "score": [
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
    )

    first = result.periods.iloc[0]

    # A is the best score and rises.
    # D is the worst score and falls.
    # Therefore the long/short strategy must make money.
    assert first["portfolio_return"] > 0
    assert first["rank_ic"] > 0


def test_backtest_long_plus_short_equals_portfolio_return():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
    )

    periods = result.periods

    assert np.allclose(
        periods["long_return"]
        + periods["short_return"],
        periods["portfolio_return"],
    )


def test_backtest_net_return_never_exceeds_gross_return_due_to_cost():
    prices = make_monthly_prices()
    scores = make_scores()

    result = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0005,
    )

    periods = result.periods

    assert np.all(
        periods["net_return"]
        <= periods["portfolio_return"] + 1e-12
    )
    

def test_run_backtest_risk_hook_identity_matches_baseline():
    prices = make_monthly_prices()
    scores = make_scores()

    rebalance_dates = pd.DatetimeIndex(
        [
            "2023-01-31",
            "2023-02-28",
            "2023-03-31",
        ]
    )

    baseline = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=rebalance_dates,
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
        risk_fn=None,
    )

    def identity_risk(weights, daily_returns, nav):
        return weights.copy()

    hooked = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=rebalance_dates,
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
        risk_fn=identity_risk,
    )

    pd.testing.assert_frame_equal(
        baseline.periods,
        hooked.periods,
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_run_backtest_risk_hook_changes_sizing_but_not_rank_ic():
    prices = make_monthly_prices()
    scores = make_scores()

    rebalance_dates = pd.DatetimeIndex(
        [
            "2023-01-31",
            "2023-02-28",
            "2023-03-31",
        ]
    )

    baseline = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=rebalance_dates,
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
        risk_fn=None,
    )

    def half_risk(weights, daily_returns, nav):
        return weights * 0.5

    risk_managed = run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=rebalance_dates,
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
        risk_fn=half_risk,
    )

    # The scoring cross-section is unchanged, so IC must be identical.
    np.testing.assert_allclose(
        baseline.periods["rank_ic"].to_numpy(),
        risk_managed.periods["rank_ic"].to_numpy(),
        atol=1e-12,
        rtol=1e-12,
    )

    
    assert not np.allclose(
        baseline.periods["portfolio_return"].to_numpy(),
        risk_managed.periods["portfolio_return"].to_numpy(),
        atol=1e-12,
        rtol=1e-12,
    )


def test_run_backtest_risk_hook_receives_only_pre_t_returns():
    prices = make_monthly_prices()
    scores = make_scores()

    rebalance_dates = pd.DatetimeIndex(
        [
            "2023-01-31",
            "2023-02-28",
            "2023-03-31",
        ]
    )

    observed = []

    def inspecting_risk(weights, daily_returns, nav):
        current_date = rebalance_dates[len(observed)]

        if not daily_returns.empty:
            assert daily_returns.index.max() < current_date

        observed.append(
            {
                "date": current_date,
                "max_return_date": (
                    daily_returns.index.max()
                    if not daily_returns.empty
                    else None
                ),
                "nav_last": nav.iloc[-1],
            }
        )

        return weights.copy()

    run_backtest(
        prices=prices,
        scores=scores,
        rebalance_dates=rebalance_dates,
        top_pct=0.25,
        bottom_pct=0.25,
        transaction_cost_rate=0.0,
        risk_fn=inspecting_risk,
    )

    assert len(observed) == 2

def test_run_backtest_skips_date_when_all_scores_are_nan():
    scores = make_scores()

    scores.loc[
        scores["date"] == pd.Timestamp("2023-02-28"),
        "score",
    ] = np.nan

    result = run_backtest(
        prices=make_monthly_prices(),
        scores=scores,
        rebalance_dates=pd.DatetimeIndex(
            [
                "2023-01-31",
                "2023-02-28",
                "2023-03-31",
            ]
        ),
        top_pct=0.25,
        bottom_pct=0.25,
    )

    assert len(result.periods) == 1
    assert len(result.skipped) == 1

    assert (
        result.skipped.iloc[0]["date"]
        == pd.Timestamp("2023-02-28")
    )