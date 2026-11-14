import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
from models.baseline import BaselineForecaster, TRAIN_END, VAL_END  # noqa: E402


@pytest.fixture(scope="module")
def fitted(feat_df):
    bf = BaselineForecaster().fit(feat_df)
    test_df = feat_df[feat_df.index > VAL_END].dropna()
    return bf, test_df


def test_garch_forecasts_cover_the_test_period_not_training(fitted):
    bf, test_df = fitted
    g = bf.predict_garch(test_df)
    assert g.index.min() >= test_df.index[0]          # regression: used to return training dates
    assert g.index.max() <= test_df.index[-1]
    assert len(g) >= len(test_df) - 1
    assert g.notna().all() and (g > 0).all()


def test_garch_forecast_is_point_in_time(fitted):
    bf, test_df = fitted
    full = bf.predict_garch(test_df)
    part = bf.predict_garch(test_df.iloc[:200])
    np.testing.assert_allclose(full.loc[part.index].values, part.values, rtol=1e-8)


def test_garch_reacts_to_test_period_shock(fitted):
    bf, test_df = fitted
    shocked = test_df.copy()
    shocked.iloc[100, shocked.columns.get_loc("log_ret")] = 0.12
    a, b = bf.predict_garch(test_df), bf.predict_garch(shocked)
    d = test_df.index[101]
    assert b.loc[d] > a.loc[d] * 1.5                    # filtered variance picks up the shock
    pd.testing.assert_series_equal(a.iloc[:100], b.iloc[:100])


def test_xgb_predictions_are_date_indexed(fitted):
    bf, test_df = fitted
    x = bf.predict_xgb(test_df)
    assert isinstance(x, pd.Series) and x.index.isin(test_df.index).all()
    assert len(x) < len(test_df)                        # lag features drop the first rows
