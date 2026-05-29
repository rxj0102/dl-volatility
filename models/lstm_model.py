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
            nn.Softplus(),  # ensures positive output (variance)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) → (batch,)"""
        _, (h_n, _) = self.lstm(x)
        out = self.head(h_n[-1])  # last layer hidden state
        return out.squeeze(-1)


def build_lstm(input_size: int = 8, device: str = "cpu") -> LSTMForecaster:
    model = LSTMForecaster(input_size=input_size)
    return model.to(device)
