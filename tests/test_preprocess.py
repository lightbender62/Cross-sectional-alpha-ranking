import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import preprocess, winsorize_cs, zscore_cs  # noqa: E402

COLS = ["f1", "f2", "f3"]


@pytest.fixture(scope="module")
def feats():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=40)
    tickers = [f"T{i:02d}" for i in range(25)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    data = rng.standard_t(df=3, size=(len(idx), len(COLS)))
    return pd.DataFrame(data, index=idx, columns=COLS)


def test_shape_index_columns(feats):
    for fn in (winsorize_cs, zscore_cs, preprocess):
        out = fn(feats)
        assert out.shape == feats.shape
        assert out.index.equals(feats.index)
        assert list(out.columns) == COLS


def test_zscore_mean_zero_std_one_per_date(feats):
    out = zscore_cs(feats)
    g = out.groupby(level="date")
    assert np.allclose(g.mean().to_numpy(), 0, atol=1e-9)
    assert np.allclose(g.std().to_numpy(), 1, atol=1e-9)


def test_zscore_is_cross_sectional_shift_invariant(feats):
    dates = feats.index.get_level_values("date")
    shift = pd.Series(np.linspace(-5, 5, dates.nunique()), index=dates.unique())
    shifted = feats.add(shift.reindex(dates).to_numpy()[:, None])
    pd.testing.assert_frame_equal(zscore_cs(feats), zscore_cs(shifted), check_exact=False, atol=1e-9)


def test_zscore_is_cross_sectional_scale_invariant(feats):
    dates = feats.index.get_level_values("date")
    scale = pd.Series(np.linspace(0.5, 4, dates.nunique()), index=dates.unique())
    scaled = feats.mul(scale.reindex(dates).to_numpy()[:, None])
    pd.testing.assert_frame_equal(zscore_cs(feats), zscore_cs(scaled), check_exact=False, atol=1e-9)


def test_no_cross_date_leakage(feats):
    base = preprocess(feats)
    bad = feats.copy()
    target = feats.index.get_level_values("date").unique()[10]
    mask = bad.index.get_level_values("date") == target
    noise = np.random.default_rng(5).normal(0, 3, mask.sum())
    bad.loc[mask, "f1"] = bad.loc[mask, "f1"].to_numpy() + noise
    pert = preprocess(bad)
    other = ~mask
    pd.testing.assert_frame_equal(base[other], pert[other], check_exact=False, atol=1e-9)
    assert not np.allclose(base[mask]["f1"], pert[mask]["f1"])


def test_row_order_does_not_matter(feats):
    shuffled = feats.sample(frac=1, random_state=0)
    a = preprocess(feats)
    b = preprocess(shuffled).reindex(feats.index)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_winsorize_clips_to_per_date_quantiles(feats):
    lo, hi = 0.05, 0.95
    out = winsorize_cs(feats, lo, hi)
    for _, g in feats.groupby(level="date"):
        o = out.loc[g.index]
        for c in COLS:
            assert o[c].max() <= g[c].quantile(hi) + 1e-12
            assert o[c].min() >= g[c].quantile(lo) - 1e-12
    inside = (feats >= feats.groupby(level="date").transform("quantile", lo)) & (
        feats <= feats.groupby(level="date").transform("quantile", hi)
    )
    pd.testing.assert_frame_equal(out.where(inside), feats.where(inside))


def test_outlier_is_tamed_by_preprocess(feats):
    bad = feats.copy()
    first = bad.index.get_level_values("date").unique()[0]
    key = (first, "T00")
    bad.loc[key, "f1"] = 1e6
    plain = zscore_cs(bad).loc[key, "f1"]
    robust = preprocess(bad).loc[key, "f1"]
    assert robust < plain
    assert robust < 5


def test_nan_stays_nan_and_does_not_poison_date(feats):
    bad = feats.copy()
    first = bad.index.get_level_values("date").unique()[0]
    bad.loc[(first, "T03"), "f2"] = np.nan
    out = preprocess(bad)
    assert np.isnan(out.loc[(first, "T03"), "f2"])
    rest = out.xs(first, level="date")["f2"].drop("T03")
    assert rest.notna().all()
    assert np.isfinite(rest).all()


def test_degenerate_dates_give_nan_not_inf():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=2), ["A", "B", "C"]], names=["date", "ticker"]
    )
    df = pd.DataFrame({"f1": [1.0, 1.0, 1.0, 1.0, 2.0, 3.0]}, index=idx)
    out = zscore_cs(df)
    first = out.xs(pd.Timestamp("2020-01-01"), level="date")["f1"]
    assert first.isna().all()
    assert not np.isinf(out["f1"]).any()