"""2-layer LSTM for next-day realized variance forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        init_log_var: float | None = None,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # The head predicts LOG variance. Daily variance is ~1e-4, while a fresh
        # Softplus head outputs ~0.7; that four-orders-of-magnitude start made
        # training unstable (4 of 5 seeds never improved on epoch 1). Starting
        # the bias at the log of the mean training variance removes the gap.
        if init_log_var is not None:
            nn.init.constant_(self.head[-1].bias, init_log_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) → (batch,)"""
        _, (h_n, _) = self.lstm(x)
        log_var = self.head(h_n[-1]).squeeze(-1)  # last layer hidden state
        return torch.exp(log_var.clamp(-30.0, 5.0))  # positive variance


def build_lstm(input_size: int = 8, device: str = "cpu",
               init_log_var: float | None = None) -> LSTMForecaster:
    model = LSTMForecaster(input_size=input_size, init_log_var=init_log_var)
    return model.to(device)
