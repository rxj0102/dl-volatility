"""QLIKE, RMSE-log-vol, and Diebold-Mariano test (HLN correction)."""

import numpy as np
import pandas as pd
from scipy import stats


def qlike(sigma2: np.ndarray, h: np.ndarray, eps: float = 1e-8) -> float:
    h = np.clip(h, eps, None)
    ratio = sigma2 / h
    return float(np.mean(ratio - np.log(ratio) - 1))


def rmse_log_vol(sigma2: np.ndarray, h: np.ndarray, eps: float = 1e-8) -> float:
    log_actual = np.log(np.sqrt(np.clip(sigma2, eps, None)))
    log_pred = np.log(np.sqrt(np.clip(h, eps, None)))
    return float(np.sqrt(np.mean((log_actual - log_pred) ** 2)))


def loss_differential(sigma2: np.ndarray, h1: np.ndarray, h2: np.ndarray, eps=1e-8):
    """Element-wise QLIKE loss differential d_t = L(h1_t) - L(h2_t)."""
    h1 = np.clip(h1, eps, None)
    h2 = np.clip(h2, eps, None)
    l1 = sigma2 / h1 - np.log(sigma2 / h1) - 1
    l2 = sigma2 / h2 - np.log(sigma2 / h2) - 1
    return l1 - l2


def diebold_mariano(
    sigma2: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    h: int = 1,
    eps: float = 1e-8,
) -> dict:
    """
    Harvey-Leybourne-Newbold corrected Diebold-Mariano test.

    H0: equal predictive accuracy (E[d_t] = 0).
    Positive DM stat → h1 is worse than h2.

    Returns dict with keys: dm_stat, p_value, n.
    """
    d = loss_differential(sigma2, h1, h2, eps=eps)
    n = len(d)
    d_bar = np.mean(d)

    # Long-run variance via Newey-West with h-1 lags
    gamma0 = np.var(d, ddof=1)
    nw_var = gamma0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        nw_var += 2 * (1 - k / h) * gamma_k

    sigma_d = np.sqrt(max(nw_var / n, 1e-20))
    dm_stat = d_bar / sigma_d

    # HLN correction factor
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_hln = dm_stat * hln_factor

    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_hln), df=n - 1))
    return {"dm_stat": float(dm_stat_hln), "p_value": float(p_value), "n": n}


def regime_qlike(
    sigma2: np.ndarray,
    h: np.ndarray,
    vix: np.ndarray,
    low_thresh: float = 20.0,
    high_thresh: float = 25.0,
) -> dict:
    """QLIKE in low-vol (VIX < 20) and high-vol (VIX > 25) regimes."""
    low_mask = vix < low_thresh
    high_mask = vix > high_thresh
    results = {}
    results["low_vol_qlike"] = qlike(sigma2[low_mask], h[low_mask]) if low_mask.sum() > 0 else np.nan
    results["high_vol_qlike"] = qlike(sigma2[high_mask], h[high_mask]) if high_mask.sum() > 0 else np.nan
    results["n_low"] = int(low_mask.sum())
    results["n_high"] = int(high_mask.sum())
    return results
