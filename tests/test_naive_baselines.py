"""Naive forecasts must use only information through t-1."""
import numpy as np
import pandas as pd

from evaluation.evaluate import get_naive_preds


def test_naive_preds_are_point_in_time():
    idx = pd.bdate_range("2021-01-01", periods=100)
    df = pd.DataFrame({"target": np.random.default_rng(0).uniform(1e-5, 1e-4, 100)}, index=idx)
    base = get_naive_preds(df)
    bumped = df.copy()
    bumped.iloc[60:, 0] *= 100  # change day 60 and later
    alt = get_naive_preds(bumped)
    for k in base:
        pd.testing.assert_series_equal(base[k].iloc[:61], alt[k].iloc[:61])
    assert np.isclose(base["Random walk"].iloc[10], df["target"].iloc[9])
