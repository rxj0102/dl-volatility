"""
Baseline models: GJR-GARCH(1,1) and XGBoost.

GJR-GARCH is fit with the arch library; XGBoost uses lagged vol features.
Both produce next-day conditional variance forecasts on the test set.
"""

import numpy as np
import pandas as pd
from arch import arch_model
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from data.features import FEATURE_COLS, download_and_engineer, garman_klass_vol

TRAIN_END = "2018-12-31"
VAL_END = "2020-12-31"


# ── GJR-GARCH ────────────────────────────────────────────────────────────────

def fit_gjr_garch(returns: pd.Series):
    """Fit GJR-GARCH(1,1) on return series. Returns fitted model result."""
    scaled = returns * 100  # arch library works better with scaled returns
    am = arch_model(scaled, vol="GARCH", p=1, o=1, q=1, dist="Normal")
    res = am.fit(disp="off", show_warning=False)
    return res


def garch_forecasts(res, n_ahead: int) -> np.ndarray:
    """Produce n_ahead one-step-ahead variance forecasts (rolling)."""
    forecasts = res.forecast(horizon=1, reindex=False)
    # residual variance in (return*100)^2 units → convert to daily var
    var_pct = forecasts.variance.values[-1, 0]
    return var_pct / (100**2)


def rolling_garch_forecasts(train_returns: pd.Series, test_returns: pd.Series) -> np.ndarray:
    """
    Rolling 1-step GARCH forecasts over the test period.
    Re-fits once on train+val and then produces expanding forecasts.
    """
    all_ret = pd.concat([train_returns])
    res = fit_gjr_garch(all_ret)

    preds = []
    expanding = train_returns.copy()
    for r in test_returns:
        res = fit_gjr_garch(expanding)
        fc = res.forecast(horizon=1, reindex=False)
        var_pct = fc.variance.values[-1, 0]
        preds.append(var_pct / (100**2))
        expanding = pd.concat([expanding, pd.Series([r])])

    return np.array(preds)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def build_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra lag features for XGBoost on top of the standard feature set."""
    out = df[FEATURE_COLS].copy()
    for lag in [1, 2, 3, 5]:
        for col in ["gk_5", "gk_10", "gk_22", "log_ret"]:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)
    return out.dropna()


class BaselineForecaster:
    """Fit GJR-GARCH and XGBoost on train data; predict on test."""

    def __init__(self):
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.garch_res = None

    def fit(self, df: pd.DataFrame):
        train = df[df.index <= TRAIN_END]
        val = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
        train_val = pd.concat([train, val])

        # GJR-GARCH on log returns
        self.garch_res = fit_gjr_garch(train_val["log_ret"].dropna())

        # XGBoost
        feat_df = build_xgb_features(train_val)
        feat_df = feat_df.loc[feat_df.index.isin(train_val.index)]
        common = feat_df.index.intersection(train_val["target"].dropna().index)
        X = feat_df.loc[common].values
        y = train_val.loc[common, "target"].values

        X_scaled = self.scaler.fit_transform(X)
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
        )
        self.xgb_model.fit(X_scaled, y)
        return self

    def predict_garch(self, test_df: pd.DataFrame) -> np.ndarray:
        """GARCH 1-step-ahead variance for each test day."""
        fc = self.garch_res.forecast(
            horizon=1, start=test_df.index[0], reindex=True
        )
        var_pct = fc.variance.iloc[: len(test_df), 0].values
        return var_pct / (100**2)

    def predict_xgb(self, test_df: pd.DataFrame) -> np.ndarray:
        feat_df = build_xgb_features(test_df)
        common = feat_df.index.intersection(test_df.index)
        X = feat_df.loc[common].values
        X_scaled = self.scaler.transform(X)
        return self.xgb_model.predict(X_scaled)

    def get_test_targets(self, df: pd.DataFrame) -> pd.Series:
        test = df[df.index > VAL_END]
        return test["target"].dropna()


def load_or_fit_baselines(df: pd.DataFrame = None) -> BaselineForecaster:
    if df is None:
        df = download_and_engineer()
    bf = BaselineForecaster()
    bf.fit(df)
    return bf
