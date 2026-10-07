import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from features import build_features  # noqa: E402

COLS = ["ret_5", "ret_20", "ret_60", "vol_20", "relvol_20", "rsi_14", "macd_norm"]


@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    frames = []
    for t in ["AAA", "BBB", "CCC"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
        frames.append(pd.DataFrame({
            "date": dates,
            "ticker": t,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 100_000, len(dates)).astype(float),
        }))
    return pd.concat(frames).sort_values(["ticker", "date"]).reset_index(drop=True)


def test_shape_and_columns(prices):
    f = build_features(prices)
    assert list(f.index.names) == ["date", "ticker"]
    for c in COLS:
        assert c in f.columns
    assert len(f) == len(prices)


def test_nans_only_in_warmup(prices):
    f = build_features(prices)
    for _, g in f.groupby(level="ticker"):
        valid = g[COLS].notna().all(axis=1).to_numpy()
        first = valid.argmax()
        assert valid[first:].all()
        assert first <= 70


def test_ranges(prices):
    f = build_features(prices).dropna()
    assert f["rsi_14"].between(0, 100).all()
    assert (f["vol_20"] > 0).all()


def test_ret_20_matches_hand_calc(prices):
    f = build_features(prices)
    g = prices[prices.ticker == "AAA"].reset_index(drop=True)
    expected = g["close"].iloc[100] / g["close"].iloc[80] - 1
    got = f.xs("AAA", level="ticker")["ret_20"].iloc[100]
    assert got == pytest.approx(expected)


def test_per_ticker_isolation(prices):
    full = build_features(prices)
    alone = build_features(prices[prices.ticker == "BBB"].reset_index(drop=True))
    a = full.xs("BBB", level="ticker")[COLS]
    b = alone.xs("BBB", level="ticker")[COLS]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_no_lookahead(prices):
    base = build_features(prices)
    cutoff = prices["date"].unique()[200]
    bad = prices.copy()
    late = bad["date"] > cutoff
    for c in ["open", "high", "low", "close", "volume"]:
        bad.loc[late, c] = bad.loc[late, c] * 10
    pert = build_features(bad)
    d = base.index.get_level_values("date")
    pd.testing.assert_frame_equal(
        base[d <= cutoff][COLS], pert[d <= cutoff][COLS], check_exact=False, rtol=1e-9
    )