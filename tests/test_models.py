import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from models import (  # noqa: E402
    ic_summary,
    make_dataset,
    make_model,
    oos_accuracy,
    oos_r2,
    rank_ic,
    walk_forward_predict,
)

COLS = ["f1", "f2", "f3"]
N = 30
MONTHS = 48
RF = dict(n_estimators=30, min_samples_leaf=5, n_jobs=1)


def build(signal, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=MONTHS, freq="ME")
    tickers = [f"T{i:02d}" for i in range(N)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    f = rng.normal(size=(len(idx), 3))
    ret = 0.02 * (signal * f[:, 0] + rng.normal(size=len(idx)))
    df = pd.DataFrame(f, index=idx, columns=COLS)
    df["fwd_ret"] = ret
    g = df.groupby(level="date")["fwd_ret"]
    df["fwd_ret_z"] = (df["fwd_ret"] - g.transform("mean")) / g.transform("std")
    df["beat_median"] = (df["fwd_ret"] > g.transform("median")).astype(float)
    df = df[df.index.get_level_values("date") < dates[-1]]
    label_end = pd.Series(dates[1:], index=dates[:-1])
    return df, label_end, dates


@pytest.fixture(scope="module")
def signal_data():
    return build(signal=1.0)


@pytest.fixture(scope="module")
def noise_data():
    return build(signal=0.0, seed=3)


def run(data, name, start=24, end=40, **params):
    df, label_end, dates = data
    return walk_forward_predict(df, COLS, label_end, name, dates[start], dates[end], min_train=12, **params)


def test_make_dataset_inner_join_and_dropna():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-31", "2020-02-28"]), ["A", "B"]], names=["date", "ticker"])
    feats = pd.DataFrame({"f1": [1.0, np.nan, 3.0, 4.0]}, index=idx)
    labels = pd.DataFrame({"fwd_ret": [0.1, 0.2, 0.3]}, index=idx[:3])
    out = make_dataset(feats, labels)
    assert list(out.index.names) == ["date", "ticker"]
    assert len(out) == 2
    assert not out.isna().any().any()


def test_make_model_names_and_unknown():
    for name in ("ridge", "logit", "rf"):
        assert hasattr(make_model(name), "fit")
    with pytest.raises(ValueError):
        make_model("xgboost")
    assert make_model("ridge", alpha=3.0).alpha == 3.0


@pytest.mark.parametrize("name", ["ridge", "logit", "rf"])
def test_predictions_cover_window_only(signal_data, name):
    df, label_end, dates = signal_data
    params = RF if name == "rf" else {}
    pred, log = run(signal_data, name, **params)
    got = pred.index.get_level_values("date").unique()
    assert got.min() >= dates[24] and got.max() <= dates[40]
    assert len(got) == 17
    assert len(pred) == 17 * N
    assert np.isfinite(pred["score"]).all()
    assert list(log.index) == list(got)


@pytest.mark.parametrize("name", ["logit", "rf"])
def test_classifier_scores_are_probabilities(signal_data, name):
    pred, _ = run(signal_data, name, **(RF if name == "rf" else {}))
    assert pred["score"].between(0, 1).all()


def test_n_train_matches_expanding_window(signal_data):
    _, log = run(signal_data, "ridge")
    expected = [(24 + i) * N for i in range(len(log))]
    assert list(log["n_train"]) == expected


@pytest.mark.parametrize("name", ["ridge", "rf"])
def test_no_lookahead_in_training(signal_data, name):
    df, label_end, dates = signal_data
    params = RF if name == "rf" else {}
    cutoff = dates[30]
    base, _ = walk_forward_predict(df, COLS, label_end, name, dates[24], dates[40], **params)

    bad = df.copy()
    d = bad.index.get_level_values("date")
    rng = np.random.default_rng(9)
    late_lab = d >= cutoff
    for c in ["fwd_ret_z", "beat_median", "fwd_ret"]:
        bad.loc[late_lab, c] = rng.permutation(bad.loc[late_lab, c].to_numpy())
    late_feat = d > cutoff
    bad.loc[late_feat, COLS] = rng.normal(size=(late_feat.sum(), 3)) * 50
    pert, _ = walk_forward_predict(bad, COLS, label_end, name, dates[24], dates[40], **params)

    pd.testing.assert_series_equal(
        base["score"][base.index.get_level_values("date") <= cutoff],
        pert["score"][pert.index.get_level_values("date") <= cutoff],
        check_exact=False,
        atol=1e-9,
    )


def test_random_forest_is_deterministic(signal_data):
    a, _ = run(signal_data, "rf", **RF)
    b, _ = run(signal_data, "rf", **RF)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("name,floor", [("ridge", 0.25), ("logit", 0.25), ("rf", 0.15)])
def test_recovers_known_signal(signal_data, name, floor):
    pred, _ = run(signal_data, name, **(RF if name == "rf" else {}))
    assert ic_summary(rank_ic(pred))["mean_ic"] > floor


@pytest.mark.parametrize("name", ["ridge", "logit"])
def test_no_signal_gives_no_ic(noise_data, name):
    pred, _ = run(noise_data, name)
    assert abs(ic_summary(rank_ic(pred))["mean_ic"]) < 0.12


def test_ridge_shrinks_with_alpha(signal_data):
    _, small = run(signal_data, "ridge", alpha=1e-6)
    _, large = run(signal_data, "ridge", alpha=1e6)
    imp = [f"imp_{c}" for c in COLS]
    assert large[imp].abs().to_numpy().max() < 0.01
    assert small[imp].abs().to_numpy().max() > 0.1


def test_ridge_finds_the_true_feature(signal_data):
    _, log = run(signal_data, "ridge")
    mean_abs = log[[f"imp_{c}" for c in COLS]].abs().mean()
    assert mean_abs.idxmax() == "imp_f1"


def test_fit_log_has_in_sample_score(signal_data):
    _, log = run(signal_data, "ridge")
    assert {"n_train", "train_score"} <= set(log.columns)
    assert log["train_score"].between(-1, 1).all()


def test_rank_ic_matches_scipy_and_perfect_score():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-31", "2020-02-29"]), list("ABCDE")], names=["date", "ticker"])
    rng = np.random.default_rng(0)
    pred = pd.DataFrame({"score": rng.normal(size=10), "fwd_ret": rng.normal(size=10)}, index=idx)
    ic = rank_ic(pred)
    for d, g in pred.groupby(level="date"):
        assert ic.loc[d] == pytest.approx(spearmanr(g["score"], g["fwd_ret"])[0])
    perfect = pred.assign(score=pred["fwd_ret"] * 3 + 1)
    assert np.allclose(rank_ic(perfect), 1.0)


def test_ic_summary_fields():
    ic = pd.Series([0.1, 0.2, -0.05, 0.15, 0.05])
    s = ic_summary(ic)
    assert s["mean_ic"] == pytest.approx(ic.mean())
    assert s["ic_ir"] == pytest.approx(ic.mean() / ic.std())
    assert s["t_stat"] == pytest.approx(ic.mean() / (ic.std() / np.sqrt(5)))
    assert s["hit_rate"] == pytest.approx(0.8)
    assert s["n_dates"] == 5


def test_oos_metrics_perfect_predictions():
    reg = pd.DataFrame({"y": [1.0, -1.0, 0.5, 2.0], "score": [1.0, -1.0, 0.5, 2.0]})
    assert oos_r2(reg) == pytest.approx(1.0)
    clf = pd.DataFrame({"y": [1.0, 0.0, 1.0, 0.0], "score": [0.9, 0.1, 0.8, 0.2]})
    assert oos_accuracy(clf) == pytest.approx(1.0)