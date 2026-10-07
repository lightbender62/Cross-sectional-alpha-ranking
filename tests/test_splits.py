import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from splits import walk_forward_splits  # noqa: E402


@pytest.fixture(scope="module")
def label_end():
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    s = pd.Series(dates)
    rb = pd.DatetimeIndex(s.groupby([s.dt.year, s.dt.month]).max().to_numpy())
    return pd.Series(rb[1:], index=rb[:-1])


def test_train_strictly_before_test_and_labels_known(label_end):
    n = 0
    for train, t in walk_forward_splits(label_end, "2023-01-01", "2023-12-31"):
        n += 1
        assert (train < t).all()
        assert (label_end.loc[train] <= t).all()
    assert n > 0


def test_no_2023_label_in_validation_training_boundary(label_end):
    train, t = next(walk_forward_splits(label_end, "2023-01-01", "2023-12-31"))
    assert t >= pd.Timestamp("2023-01-01")
    assert label_end.loc[train].max() <= t


def test_test_dates_in_window_ascending_and_complete(label_end):
    got = [t for _, t in walk_forward_splits(label_end, "2024-01-01", "2024-12-31")]
    expected = [d for d in label_end.index if pd.Timestamp("2024-01-01") <= d <= pd.Timestamp("2024-12-31")]
    assert got == expected
    assert got == sorted(got)


def test_train_window_is_expanding(label_end):
    prev = None
    for train, _ in walk_forward_splits(label_end, "2023-01-01", "2024-12-31"):
        if prev is not None:
            assert set(prev) <= set(train)
            assert len(train) >= len(prev)
        prev = train


def test_min_train_respected(label_end):
    for train, _ in walk_forward_splits(label_end, "2015-01-01", "2016-12-31", min_train=12):
        assert len(train) >= 12
    first = next(walk_forward_splits(label_end, "2015-01-01", "2016-12-31", min_train=12))
    assert len(first[0]) == 12


def test_purges_overlapping_labels(label_end):
    long_end = pd.Series(label_end.index + pd.DateOffset(months=3), index=label_end.index)
    for train, t in walk_forward_splits(long_end, "2023-01-01", "2023-12-31"):
        assert (long_end.loc[train] <= t).all()
        assert len(train) < len(label_end.index[label_end.index < t])