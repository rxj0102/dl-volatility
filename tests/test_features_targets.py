import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yfinance")
from data.features import FEATURE_COLS, engineer_features  # noqa: E402


def _gk(spy):
    hl, co = np.log(spy["High"] / spy["Low"]), np.log(spy["Close"] / spy["Open"])
    return 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2


def test_target_is_same_day_variance(spy_vix, feat_df):
    spy, _ = spy_vix
    gk = _gk(spy)
    d = feat_df.index[500]
    assert feat_df.loc[d, "target"] == pytest.approx(gk.loc[d])


def test_features_exclude_the_targets_own_day(spy_vix):
    spy, vix = spy_vix
    base = engineer_features(spy, vix)
    d = base.index[500]
    bumped = spy.copy()
    bumped.loc[d, "High"] *= 1.10  # change day d's range only
    alt = engineer_features(bumped, vix)
    # day d's features (information through d-1) are untouched, its target moves
    pd.testing.assert_series_equal(base.loc[d, FEATURE_COLS], alt.loc[d, FEATURE_COLS])
    assert base.loc[d, "target"] != pytest.approx(alt.loc[d, "target"])
    # the next row's features do see day d
    nxt = base.index[501]
    assert not np.allclose(base.loc[nxt, ["gk_5", "gk_10", "gk_22"]], alt.loc[nxt, ["gk_5", "gk_10", "gk_22"]])


def test_features_point_in_time(spy_vix, feat_df):
    spy, vix = spy_vix
    part = engineer_features(spy.iloc[:1500], vix.iloc[:1500])
    pd.testing.assert_frame_equal(part, feat_df.loc[part.index])


def test_vix_as_one_column_frame_matches_series(spy_vix):
    """Newer yfinance returns Close as a one-column DataFrame named ^VIX."""
    spy, vix = spy_vix
    as_frame = vix.to_frame(name="^VIX")
    pd.testing.assert_frame_equal(engineer_features(spy, vix), engineer_features(spy, as_frame))
