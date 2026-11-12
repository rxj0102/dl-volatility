import math

import pytest

torch = pytest.importorskip("torch")
from models.lstm_model import build_lstm  # noqa: E402


def test_fresh_lstm_starts_at_mean_training_variance():
    """With the head bias at log(mean variance), an untrained model already
    forecasts on the right scale (the old Softplus head started near 0.7)."""
    torch.manual_seed(0)
    model = build_lstm(input_size=8, init_log_var=math.log(1e-4)).eval()
    with torch.no_grad():
        out = model(torch.randn(64, 60, 8))
    assert (out > 0).all()
    ratio = out / 1e-4
    assert ratio.min() > 0.5 and ratio.max() < 2.0


def test_output_is_positive_without_init():
    model = build_lstm(input_size=4).eval()
    with torch.no_grad():
        assert (model(torch.randn(8, 60, 4)) > 0).all()
