"""PyTorch Dataset and DataLoader for 60-day rolling window sequences."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from data.features import FEATURE_COLS, download_and_engineer

SEQ_LEN = 60
TRAIN_END = "2018-12-31"
VAL_END = "2020-12-31"


def get_splits(df: pd.DataFrame):
    train = df[df.index <= TRAIN_END]
    val = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test = df[df.index > VAL_END]
    return train, val, test


class VolDataset(Dataset):
    """Rolling 60-day window → next-day realized variance."""

    def __init__(self, data: np.ndarray, targets: np.ndarray, seq_len: int = SEQ_LEN):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        # windows end at rows seq_len-1 .. n-1
        return max(len(self.data) - self.seq_len + 1, 0)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        # target[t] is already the NEXT-day variance observed from row t, so the
        # window ending at row idx+seq_len-1 is paired with target[idx+seq_len-1]
        y = self.targets[idx + self.seq_len - 1]
        return x, y


def build_loaders(
    batch_size: int = 64,
    ticker: str = "SPY",
    num_workers: int = 0,
):
    """Return (train_loader, val_loader, test_loader, scaler, df)."""
    df = download_and_engineer(ticker=ticker)

    train_df, val_df, test_df = get_splits(df)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df[FEATURE_COLS].values)
    val_X = scaler.transform(val_df[FEATURE_COLS].values)
    test_X = scaler.transform(test_df[FEATURE_COLS].values)

    train_y = train_df["target"].values
    val_y = val_df["target"].values
    test_y = test_df["target"].values

    train_ds = VolDataset(train_X, train_y)
    val_ds = VolDataset(val_X, val_y)
    test_ds = VolDataset(test_X, test_y)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=False, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, scaler, df


def build_tft_dataframes(ticker: str = "SPY"):
    """Return train/val/test DataFrames formatted for pytorch-forecasting."""
    df = download_and_engineer(ticker=ticker)

    scaler = StandardScaler()
    train_df, val_df, test_df = get_splits(df)
    all_X = pd.concat([train_df, val_df, test_df])

    scaler.fit(train_df[FEATURE_COLS].values)  # train-only statistics (no leakage)
    scaled = scaler.transform(all_X[FEATURE_COLS].values)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLS, index=all_X.index)
    scaled_df["target"] = all_X["target"].values
    scaled_df["vix_raw"] = all_X["vix"].values  # kept for regime analysis

    # pytorch-forecasting requires integer time_idx and group id
    scaled_df = scaled_df.reset_index()
    scaled_df.rename(columns={"Date": "date"}, inplace=True)
    # TFT encoders only see rows BEFORE the decoder step. To give the TFT the same
    # information set as the LSTM (features through day t for target[t]), the TFT
    # target at row j is target[j-1], and its forecast belongs to the date of row j-1.
    scaled_df["tft_target"] = scaled_df["target"].shift(1)
    scaled_df["forecast_date"] = scaled_df["date"].shift(1)
    scaled_df = scaled_df.iloc[1:].reset_index(drop=True)
    scaled_df["time_idx"] = np.arange(len(scaled_df))
    scaled_df["group"] = "SPY"

    fd = scaled_df["forecast_date"]
    train_mask = fd <= pd.Timestamp(TRAIN_END)
    val_mask = (fd > pd.Timestamp(TRAIN_END)) & (fd <= pd.Timestamp(VAL_END))
    test_mask = fd > pd.Timestamp(VAL_END)

    return scaled_df, train_mask, val_mask, test_mask, scaler
