import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("yfinance")
import data.dataset as ds  # noqa: E402
from data.features import FEATURE_COLS  # noqa: E402


def test_window_is_paired_with_its_own_rows_target():
    n, seq = 100, 10
    data = np.arange(n, dtype=float).reshape(-1, 1)
    targets = np.arange(n, dtype=float) * 10
    d = ds.VolDataset(data, targets, seq_len=seq)
    assert len(d) == n - seq + 1
    x, y = d[0]
    assert x[-1, 0] == 9 and y == 90            # window ends at row 9 -> target of row 9
    x, y = d[len(d) - 1]
    assert x[-1, 0] == n - 1 and y == (n - 1) * 10   # last window uses the final row


def test_tft_scaler_is_fit_on_train_only(monkeypatch, feat_df):
    monkeypatch.setattr(ds, "download_and_engineer", lambda ticker="SPY": feat_df)
    scaled_df, train_mask, _, _, scaler = ds.build_tft_dataframes()
    train = feat_df[feat_df.index <= ds.TRAIN_END]
    np.testing.assert_allclose(scaler.mean_, train[FEATURE_COLS].mean().values)
    raw = feat_df.loc[scaled_df["date"].values, FEATURE_COLS].values
    np.testing.assert_allclose(scaled_df[FEATURE_COLS].values, (raw - scaler.mean_) / scaler.scale_)


def test_tft_target_is_lagged_one_row(monkeypatch, feat_df):
    """TFT row j forecasts target[j-1], dated at row j-1, so encoder rows
    (strictly before j) hold features through the forecast date, as for the LSTM."""
    monkeypatch.setattr(ds, "download_and_engineer", lambda ticker="SPY": feat_df)
    scaled_df, train_mask, val_mask, test_mask, _ = ds.build_tft_dataframes()
    assert len(scaled_df) == len(feat_df) - 1
    np.testing.assert_allclose(scaled_df["tft_target"].values, feat_df["target"].values[:-1])
    assert (scaled_df["forecast_date"].values == feat_df.index.values[:-1]).all()
    assert (scaled_df["date"].values == feat_df.index.values[1:]).all()
    # the three periods partition the rows by forecast date
    assert (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int) == 1).all()
    assert scaled_df.loc[test_mask, "forecast_date"].min() > pd.Timestamp(ds.VAL_END)


def test_tft_datasets_score_every_day(monkeypatch, feat_df):
    """Validation and test sets hold one rolling window per period day, not the
    single window that predict=True produced before."""
    pytest.importorskip("pytorch_forecasting")
    from models.tft_model import build_tft_datasets, build_tft_test_dataset, SEQ_LEN

    monkeypatch.setattr(ds, "download_and_engineer", lambda ticker="SPY": feat_df)
    scaled_df, train_mask, val_mask, test_mask, _ = ds.build_tft_dataframes()
    training, validation = build_tft_datasets(scaled_df, train_mask, val_mask)
    test = build_tft_test_dataset(training, scaled_df, test_mask)

    assert len(training) == int(train_mask.sum()) - SEQ_LEN
    assert len(validation) == int(val_mask.sum())
    assert len(test) == int(test_mask.sum())
