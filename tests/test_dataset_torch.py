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
    np.testing.assert_allclose(scaled_df.loc[train_mask, FEATURE_COLS].mean().values, 0.0, atol=1e-8)
