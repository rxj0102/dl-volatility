"""QLIKE loss used for training and evaluation."""

import torch
import torch.nn as nn
import numpy as np


class QLIKELoss(nn.Module):
    """
    QLIKE: mean( sigma^2 / h - log(sigma^2 / h) - 1 )
    where sigma^2 is the realized variance and h is the forecast.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, h: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        h = h.clamp(min=self.eps)
        ratio = sigma2 / h
        return (ratio - torch.log(ratio) - 1).mean()


def qlike_numpy(sigma2: np.ndarray, h: np.ndarray, eps: float = 1e-8) -> float:
    h = np.clip(h, eps, None)
    ratio = sigma2 / h
    return float(np.mean(ratio - np.log(ratio) - 1))


def rmse_log_vol(sigma2: np.ndarray, h: np.ndarray, eps: float = 1e-8) -> float:
    log_actual = np.log(np.sqrt(np.clip(sigma2, eps, None)))
    log_pred = np.log(np.sqrt(np.clip(h, eps, None)))
    return float(np.sqrt(np.mean((log_actual - log_pred) ** 2)))
