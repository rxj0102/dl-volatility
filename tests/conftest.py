import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_spy(n=3300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n, name="Date")  # yfinance names the index "Date"
    sig = 0.01 * np.exp(0.5 * np.cumsum(rng.normal(0, 0.05, n)) * 0.2)
    r = rng.normal(0, 1, n) * sig
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, sig / 2)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, sig / 2)))
    spy = pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close,
                        "Volume": rng.integers(5_000_000, 9_000_000, n).astype(float)}, index=idx)
    vix = pd.Series(15 + 100 * sig, index=idx)
    return spy, vix


@pytest.fixture(scope="session")
def spy_vix():
    return make_spy()


@pytest.fixture(scope="session")
def feat_df(spy_vix):
    pytest.importorskip("yfinance")
    from data.features import engineer_features

    return engineer_features(*spy_vix)
