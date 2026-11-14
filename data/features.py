"""Feature engineering for next-day volatility forecasting.

Default data is open and free (no Yahoo Finance):
  SPY stand in : Ken French US market total return (Mkt-RF + RF), daily
  VIX          : FRED VIXCLS

These sources only provide closes, so the default (close-only) mode uses
close-to-close realized volatility and a squared-return variance target.
If OHLCV data is supplied, engineer_features() uses Garman-Klass instead.

Feature columns are named generically so both modes share FEATURE_COLS:
  vol_5, vol_10, vol_22 : rolling annualised volatility (GK or close-to-close)
  vix                   : VIX close
  log_ret               : daily log return
  aux_z                 : 22-day z-score of volume (OHLCV) or of the daily VIX log change (close-only)
  dow, month            : calendar
  target                : variance of day t (GK or squared log return)
All features are lagged 1 day, so row t uses information through t-1.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = ["vol_5", "vol_10", "vol_22", "vix", "log_ret", "aux_z", "dow", "month"]
VOL_COLS = ["vol_5", "vol_10", "vol_22"]
TARGET_FLOOR = 2.5e-9  # Ken French returns are rounded to 0.01%, so a zero return means |r| < 0.5e-4;
                       # floor the squared return at that resolution (QLIKE needs a positive target)


def garman_klass_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Annualised Garman-Klass realized volatility over a rolling window."""
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Close"] / df["Open"])
    gk_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(gk_daily.rolling(window).mean() * 252)


def load_open_data(start: str = "2010-01-01", end: str = None):
    """Return (prices, vix) from open sources: Ken French market index and FRED VIX."""
    from data.opendata import fred, french

    ff = french("F-F_Research_Data_Factors_daily", start=start, end=end)
    mkt = ff["Mkt-RF"] + ff["RF"]
    prices = pd.DataFrame({"Close": 100 * (1 + mkt).cumprod()})
    prices.index.name = "Date"
    vix = fred("VIXCLS", start, end)["VIXCLS"].dropna()
    return prices, vix


def download_and_engineer(
    ticker: str = "SPY",
    vix_ticker: str = "^VIX",
    start: str = "2010-01-01",
    end: str = None,
) -> pd.DataFrame:
    """Load open data for the SPY stand in and VIX and build the feature frame.

    ticker and vix_ticker are kept for API compatibility; the open data loader
    always uses the Ken French US market return and FRED VIXCLS.
    """
    prices, vix = load_open_data(start, end)
    return engineer_features(prices, vix)


def engineer_features(spy: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Pure feature construction (no network). Row t holds information through the
    close of day t-1 and the target is the realised variance of day t, i.e. a
    genuine one-step-ahead forecast problem.

    Uses Garman-Klass when Open/High/Low/Volume are present, otherwise close-only.
    """
    vix = vix.copy()
    vix.name = "vix"
    df = spy.copy()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    ohlcv = {"Open", "High", "Low", "Volume"}.issubset(df.columns)

    if ohlcv:
        for w in (5, 10, 22):
            df[f"vol_{w}"] = garman_klass_vol(df, w)
        aux = df["Volume"]
    else:
        for w in (5, 10, 22):
            df[f"vol_{w}"] = df["log_ret"].rolling(w).std() * np.sqrt(252)

    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.join(vix, how="left")
    df["vix"] = df["vix"].ffill()

    if not ohlcv:
        aux = np.log(df["vix"]).diff()
    df["aux_z"] = (aux - aux.rolling(22).mean()) / aux.rolling(22).std()

    # target: realised variance of day t. Features are lagged one day below.
    if ohlcv:
        log_hl = np.log(df["High"] / df["Low"])
        log_co = np.log(df["Close"] / df["Open"])
        df["target"] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    else:
        df["target"] = (df["log_ret"] ** 2).clip(lower=TARGET_FLOOR)

    for col in FEATURE_COLS:
        df[col] = df[col].shift(1)

    df = df.dropna(subset=FEATURE_COLS + ["target"])
    return df[FEATURE_COLS + ["target", "Close"]]
