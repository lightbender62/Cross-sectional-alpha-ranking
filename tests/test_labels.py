import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from labels import make_labels, rebalance_dates  # noqa: E402

TICKERS = [f"T{i:02d}" for i in range(25)]


@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    frames = []
    for t in TICKERS:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(dates))))
        frames.append(pd.DataFrame({"date": dates, "ticker": t, "close": close}))
    return pd.concat(frames).reset_index(drop=True)


@pytest.fixture(scope="module")
def rb(prices):
    return rebalance_dates(prices["date"].unique())


def test_rebalance_dates_are_last_trading_day_of_each_month(prices, rb):
    dates = pd.DatetimeIndex(prices["date"].unique())
    assert rb.isin(dates).all()
    assert rb.is_monotonic_increasing
    assert len(rb) == 36
    assert len(set(zip(rb.year, rb.month))) == len(rb)
    for d in rb:
        later = dates[(dates > d) & (dates.month == d.month) & (dates.year == d.year)]
        assert len(later) == 0


def test_rebalance_skips_weekend_month_end():
    dates = pd.bdate_range("2021-01-01", "2021-03-31")
    rb = rebalance_dates(dates)
    assert pd.Timestamp("2021-01-29") in rb
    assert pd.Timestamp("2021-01-31") not in rb


def test_columns_and_last_date_dropped(prices, rb):
    lab = make_labels(prices, rb)
    assert list(lab.index.names) == ["date", "ticker"]
    for c in ["label_end", "fwd_ret", "fwd_ret_z", "beat_median"]:
        assert c in lab.columns
    dates = lab.index.get_level_values("date")
    assert rb[-1] not in dates
    assert rb[0] in dates
    assert lab[["fwd_ret", "fwd_ret_z", "beat_median", "label_end"]].notna().all().all()


def test_fwd_ret_matches_hand_calc(prices, rb):
    lab = make_labels(prices, rb)
    g = prices[prices.ticker == "T03"].set_index("date")["close"]
    t0, t1 = rb[5], rb[6]
    expected = g[t1] / g[t0] - 1
    assert lab.loc[(t0, "T03"), "fwd_ret"] == pytest.approx(expected)
    assert lab.loc[(t0, "T03"), "label_end"] == t1


def test_label_end_is_next_rebalance_date(prices, rb):
    lab = make_labels(prices, rb)
    d = lab.index.get_level_values("date")
    nxt = dict(zip(rb[:-1], rb[1:]))
    assert (lab["label_end"].to_numpy() == pd.DatetimeIndex(d.map(nxt)).to_numpy()).all()


def test_fwd_ret_z_is_cross_sectional(prices, rb):
    lab = make_labels(prices, rb)
    g = lab.groupby(level="date")["fwd_ret_z"]
    assert np.allclose(g.mean(), 0, atol=1e-9)
    assert np.allclose(g.std(), 1, atol=1e-9)


def test_beat_median_splits_cross_section(prices, rb):
    lab = make_labels(prices, rb)
    counts = lab.groupby(level="date")["beat_median"].sum()
    assert (counts == len(TICKERS) // 2).all()
    assert set(lab["beat_median"].unique()) <= {0.0, 1.0}


def test_label_uses_only_the_two_rebalance_closes(prices, rb):
    base = make_labels(prices, rb)
    bad = prices.copy()
    keep = bad["date"].isin(rb)
    bad.loc[~keep, "close"] = bad.loc[~keep, "close"] * 7
    pd.testing.assert_frame_equal(base, make_labels(bad, rb), check_exact=False, rtol=1e-12)


def test_label_k_ignores_prices_after_window(prices, rb):
    base = make_labels(prices, rb)
    bad = prices.copy()
    late = bad["date"] > rb[10]
    bad.loc[late, "close"] = bad.loc[late, "close"] * 5
    pert = make_labels(bad, rb)
    d = base.index.get_level_values("date")
    early = d < rb[9]
    pd.testing.assert_frame_equal(base[early], pert[early], check_exact=False, rtol=1e-12)