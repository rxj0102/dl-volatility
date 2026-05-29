"""Feature engineering: Garman-Klass volatility, VIX, log returns, volume z-score."""

import numpy as np
import pandas as pd
import yfinance as yf


def garman_klass_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Annualised Garman-Klass realized volatility over a rolling window."""
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(gk_daily.rolling(window).mean() * 252)


def download_and_engineer(
    ticker: str = "SPY",
    vix_ticker: str = "^VIX",
    start: str = "2010-01-01",
    end: str = None,
) -> pd.DataFrame:
    """
    Download OHLCV + VIX and return a feature DataFrame with:
      gk_5, gk_10, gk_22  – rolling Garman-Klass vol
      vix                  – VIX close
      log_ret              – daily log return
      vol_zscore           – 22-day volume z-score
      dow                  – day-of-week (0-4)
      month                – month (1-12)
      target               – next-day GK daily variance (for QLIKE)
    All features are lagged 1 day to avoid look-ahead.
    """
    spy = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    vix = yf.download(vix_ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    vix.name = "vix"

    df = spy.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["gk_5"] = garman_klass_vol(df, 5)
    df["gk_10"] = garman_klass_vol(df, 10)
    df["gk_22"] = garman_klass_vol(df, 22)

    vol_mean = df["Volume"].rolling(22).mean()
    vol_std = df["Volume"].rolling(22).std()
    df["vol_zscore"] = (df["Volume"] - vol_mean) / vol_std

    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()

    # next-day GK daily variance as target
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk_daily_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    df["target"] = gk_daily_var.shift(-1)  # next day

    feature_cols = ["gk_5", "gk_10", "gk_22", "vix", "log_ret", "vol_zscore", "dow", "month"]

    # lag features 1 day to avoid look-ahead
    for col in feature_cols:
        df[col] = df[col].shift(1)

    df = df.dropna(subset=feature_cols + ["target"])
    return df[feature_cols + ["target", "Close"]]


FEATURE_COLS = ["gk_5", "gk_10", "gk_22", "vix", "log_ret", "vol_zscore", "dow", "month"]
