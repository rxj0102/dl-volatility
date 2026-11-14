"""2-layer LSTM for next-day realized variance forecasting."""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        log_var_init: float = -9.0,
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
        # Predict log-variance around a fixed offset (log of the mean training
        # target) and exponentiate, so the forecast is positive and starts on
        # the right scale. A Softplus head starts near 1, about four orders of
        # magnitude above daily variance (~1e-4), and training stalls.
        self.register_buffer("log_var_offset", torch.tensor(float(log_var_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) → (batch,)"""
        _, (h_n, _) = self.lstm(x)
        raw = self.head(h_n[-1]).squeeze(-1)  # last layer hidden state
        return torch.exp((raw + self.log_var_offset).clamp(-30.0, 5.0))


def build_lstm(input_size: int = 8, device: str = "cpu", log_var_init: float = -9.0) -> LSTMForecaster:
    model = LSTMForecaster(input_size=input_size, log_var_init=log_var_init)
    return model.to(device)
