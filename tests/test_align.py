import numpy as np
import pandas as pd

from evaluation.align import align_on_dates


def test_alignment_uses_dates_not_positions():
    idx = pd.bdate_range("2021-01-04", periods=10)
    target = pd.Series(np.arange(10.0), index=idx)
    late = pd.Series(np.arange(5.0) + 100, index=idx[5:])   # starts 5 days later (warm-up)
    full = pd.Series(np.arange(10.0) * 2, index=idx)
    out = align_on_dates(target, {"late": late, "full": full})
    assert list(out.index) == list(idx[5:])
    # position-based truncation would have paired late[0] with target[0]
    assert out["target"].iloc[0] == 5.0 and out["late"].iloc[0] == 100.0


def test_alignment_drops_nans():
    idx = pd.RangeIndex(5)
    t = pd.Series([1, 2, 3, 4, 5.0], index=idx)
    p = pd.Series([1, np.nan, 3, 4, 5.0], index=idx)
    assert len(align_on_dates(t, {"m": p})) == 4
