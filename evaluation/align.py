"""Date-based alignment of model forecasts and realised targets."""

from __future__ import annotations

import pandas as pd


def align_on_dates(targets: pd.Series, preds: dict[str, pd.Series]) -> pd.DataFrame:
    """Return a frame with ``target`` and one column per model on the COMMON dates.

    Models start forecasting on different dates (sequence warm-up, lag features),
    so truncating arrays by position compares forecasts with the wrong days.
    """
    cols = [targets.rename("target")] + [p.rename(k) for k, p in preds.items()]
    return pd.concat(cols, axis=1, join="inner").dropna()
