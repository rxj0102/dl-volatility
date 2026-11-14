# dl-volatility: LSTM vs. Temporal Fusion Transformer for Realized Volatility Forecasting

[![CI](https://github.com/rxj0102/dl-volatility/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/dl-volatility/actions/workflows/ci.yml)

Benchmark deep-learning volatility models (LSTM, TFT) against classical baselines (GJR-GARCH, XGBoost) on daily US equity market data from 2010 to September 2026, using free open data (Ken French Data Library, FRED). Evaluation uses QLIKE loss, RMSE of log-vol, Diebold-Mariano significance tests, and regime-conditional analysis.

---

## Architecture Summary

### LSTM
| Component | Detail |
|-----------|--------|
| Layers | 2-layer LSTM |
| Hidden size | 128 |
| Dropout | 0.2 |
| Output head | Linear → Softplus (ensures positive variance) |
| Optimizer | Adam, lr=1e-3, ReduceLROnPlateau |
| Input | 60-day rolling window, 8 features |

### Temporal Fusion Transformer (TFT)
| Component | Detail |
|-----------|--------|
| Hidden size | 64 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Hidden continuous size | 32 |
| Encoder length | 60 days |
| Framework | pytorch-forecasting |
| Loss | RMSE (regression mode) |

### Baselines
- **GJR-GARCH(1,1)** – asymmetric GARCH via the `arch` library; captures leverage effect.
- **XGBoost** – gradient-boosted trees on lagged realized vol, VIX, and return features.

---

## Data

| Series | Source |
|--------|--------|
| SPY stand in | Ken French Data Library, US market total return (Mkt-RF + RF), daily, 2010 to 2026-09-30 |
| VIX | FRED VIXCLS |

These sources publish closes only (no open, high, low or volume), so the default pipeline is close-only. If you pass OHLCV data to `engineer_features`, it switches back to Garman-Klass features and target.

## Features
| Feature | Close-only (default) | With OHLCV input |
|---------|-------------|-------------|
| `vol_5/10/22` | Close-to-close annualized vol (5/10/22-day window) | Garman-Klass annualized vol |
| `vix` | VIX close | VIX close |
| `log_ret` | Daily log return | Daily log return |
| `aux_z` | 22-day z-score of the daily VIX log change | 22-day volume z-score |
| `dow` | Day-of-week (0-4) | Day-of-week (0-4) |
| `month` | Month (1-12) | Month (1-12) |

All features are lagged 1 day. Target: realized variance of day t, which is the squared daily log return in close-only mode (floored at 1e-10) and Garman-Klass variance with OHLCV.

---

## Data Splits
| Split | Period |
|-------|--------|
| Train | 2010-01-01 – 2018-12-31 |
| Val | 2019-01-01 – 2020-12-31 |
| Test | 2021-01-01 – present |

---

## Test-Set Results

Test set: 2021-01-04 to 2026-09-30, scored on the dates where all three models have a forecast (the LSTM needs a 60 day window, so it starts later). Numbers are copied from `outputs/model_comparison.csv` and `outputs/regime_performance.csv`, produced by `python -m training.run_all --skip-tft`.

| Model | QLIKE | RMSE (log-vol) | DM vs GARCH stat | DM p-value |
|-------|-------|----------------|------------------|------------|
| GJR-GARCH | 1.5142 | 1.3625 | 0.000 | 1.000 |
| XGBoost | 1.5237 | 1.3718 | 0.450 | 0.653 |
| LSTM | 3.9137 | 2.7048 | 44.686 | 0.000 |
| TFT | not run | not run | not run | not run |

Positive DM statistics mean the model has higher QLIKE loss than GARCH. XGBoost is statistically indistinguishable from GJR-GARCH. The LSTM is far worse: on this noisy squared return target its validation loss was best after the first epoch and early stopping ended training at epoch 16 (`outputs/training_log_lstm.csv`), so it did not learn a useful mapping. TFT was not run here because one epoch takes about five minutes on CPU; run `python -m training.run_all` on a GPU to add it.

---

## Regime-Conditional QLIKE

| Model | Low-vol QLIKE (VIX < 20, 948 days) | High-vol QLIKE (VIX > 25, 179 days) |
|-------|--------------------------|---------------------------|
| GJR-GARCH | 1.5370 | 1.3314 |
| XGBoost | 1.5304 | 1.4677 |
| LSTM | 4.3600 | 2.4814 |
| TFT | not run | not run |

XGBoost edges GARCH in calm markets and loses to it in stressed markets.

---

## Project Structure

```
dl-volatility/
├── data/
│   ├── features.py        # Realized vol (close-only or Garman-Klass), VIX, returns
│   ├── opendata.py        # FRED and Ken French downloaders (cached)
│   └── dataset.py         # PyTorch Dataset, DataLoaders, TFT-compatible splits
├── models/
│   ├── lstm_model.py      # 2-layer LSTM forecaster
│   ├── tft_model.py       # TFT via pytorch-forecasting
│   └── baseline.py        # GJR-GARCH + XGBoost wrappers
├── training/
│   ├── losses.py          # QLIKE loss (PyTorch + NumPy)
│   ├── train_lstm.py      # LSTM train loop, early stopping, checkpointing
│   ├── train_tft.py       # TFT Lightning trainer
│   └── run_all.py         # End-to-end runner
├── evaluation/
│   ├── metrics.py         # QLIKE, RMSE, DM test (HLN), regime split
│   └── evaluate.py        # Full evaluation pipeline + plots
├── outputs/
│   ├── checkpoints/       # Best model weights
│   ├── model_comparison.csv
│   ├── regime_performance.csv
│   ├── forecast_comparison.png
│   └── tft_attention_heatmap.png
├── notebooks/
│   └── dl_volatility_analysis.ipynb
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Train both models and evaluate (GPU optional)
python -m training.run_all

# Train LSTM only (faster)
python -m training.run_all --skip-tft

# Evaluation only (if checkpoints exist)
python -m evaluation.evaluate outputs/checkpoints/tft_best.ckpt
```

---

## Evaluation notes (important)

- **Target and alignment fixed.** Features were lagged one day *and* the target shifted `-1`, so the "next-day" target was really two days ahead, and the LSTM dataset paired each window with the target one row too late. Rows now hold information through `t-1` and the target is the realised GK variance of day `t`; each window is paired with its own last row's target.
- **GARCH baseline fixed.** `predict_garch` previously returned *training-period* forecasts (arch ignores `start` beyond the fitted sample). It now fixes the train+val parameters and filters through the test returns, so forecasts are genuinely out of sample.
- **Date-based comparison.** `evaluate.py` used to truncate LSTM, GARCH and XGBoost forecasts by position, which compared forecasts with the wrong days (the LSTM starts 60 days into the test set, XGBoost a few days in). Models are now aligned on common dates.
- **TFT scaler** is fit on the training split only.
- **Results regenerated after these fixes** on open data (Nov 2026). The TFT evaluation path is still experimental (it is not date-aligned).
