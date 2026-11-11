# dl-volatility: LSTM vs. Temporal Fusion Transformer for Realized Volatility Forecasting

[![CI](https://github.com/rxj0102/dl-volatility/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/dl-volatility/actions/workflows/ci.yml)

Benchmark deep-learning volatility models (LSTM, TFT) against classical baselines (GJR-GARCH, XGBoost) on SPY daily data from 2010 to present. Evaluation uses QLIKE loss, RMSE of log-vol, Diebold-Mariano significance tests, and regime-conditional analysis.

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
- **XGBoost** – gradient-boosted trees on lagged Garman-Klass vol, VIX, and return features.

---

## Features
| Feature | Description |
|---------|-------------|
| `gk_5/10/22` | Garman-Klass annualized realized vol (5/10/22-day window) |
| `vix` | CBOE VIX close |
| `log_ret` | Daily log return |
| `vol_zscore` | 22-day volume z-score |
| `dow` | Day-of-week (0–4) |
| `month` | Month (1–12) |

All features are lagged 1 day. Target: next-day Garman-Klass daily variance.

---

## Data Splits
| Split | Period |
|-------|--------|
| Train | 2010-01-01 – 2018-12-31 |
| Val | 2019-01-01 – 2020-12-31 |
| Test | 2021-01-01 – present |

---

## Test-Set Results

From `python -m training.train_lstm` then `python -m evaluation.evaluate`, run on 2026-11-11 (CPU, one training run, early stopping at epoch 90). Test set: 1,413 common trading days, 2021-03-30 to 2026-11-11, after the 2010 to 2018 training and 2019 to 2020 validation periods. Target: the Garman-Klass variance of day t, forecast from information through t-1. Lower is better. Source files: `outputs/model_comparison.csv`, `outputs/regime_performance.csv`, `outputs/training_log_lstm.csv`.

The random walk (yesterday's Garman-Klass variance) and the 22-day mean are naive benchmarks on the same information set. DM is the Diebold-Mariano statistic on QLIKE losses; negative means the row's model has lower loss than the comparison model.

| Model | QLIKE | RMSE (log-vol) | DM vs GARCH | p | DM vs random walk | p |
|-------|-------|----------------|-------------|---|-------------------|---|
| LSTM | **0.3061** | **0.4253** | -9.70 | 0.0000 | -8.10 | 0.0000 |
| GARCH | 0.4097 | 0.5584 | 0.00 | 1.0000 | -4.97 | 0.0000 |
| XGB | 0.3640 | 0.4824 | -4.01 | 0.0001 | -6.24 | 0.0000 |
| Random walk | 0.6356 | 0.4774 | 4.97 | 0.0000 | 0.00 | 1.0000 |
| 22d mean | 0.4399 | 0.4834 | 1.23 | 0.2186 | -4.55 | 0.0000 |

The TFT is not included: its evaluation path is not date-aligned (see Evaluation notes), so it was not trained for this run.

**Reading this honestly.**
- The LSTM has the lowest QLIKE and RMSE, and beats GARCH, XGBoost and both naive benchmarks with Diebold-Mariano p-values below 0.001.
- GARCH beats the random walk on QLIKE but loses to it on RMSE. GARCH forecasts close-to-close variance while the target is range-based, so it is on a different scale; that is why the naive rows are in the table.
- This is one training run with one seed. The size of the LSTM edge across seeds has not been measured.

## Regime-Conditional QLIKE

VIX at the prior close below 20 (973 days) or above 25 (181 days); days in between are in neither column.

| Model | Low-vol QLIKE (VIX < 20) | High-vol QLIKE (VIX > 25) |
|-------|--------------------------|---------------------------|
| LSTM | 0.3206 | 0.2557 |
| GARCH | 0.4307 | 0.3766 |
| XGB | 0.3857 | 0.2913 |
| Random walk | 0.6707 | 0.5555 |
| 22d mean | 0.4176 | 0.5555 |

---

## Project Structure

```
dl-volatility/
├── data/
│   ├── features.py        # Garman-Klass vol, VIX, log returns, volume z-score
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

- **Target and alignment fixed.** Features were lagged one day *and* the target shifted `-1`, so the next-day target was really two days ahead, and the LSTM dataset paired each window with the target one row too late. Rows now hold information through `t-1` and the target is the realised GK variance of day `t`; each window is paired with its own last row's target.
- **GARCH baseline fixed.** `predict_garch` previously returned *training-period* forecasts (arch ignores `start` beyond the fitted sample). It now fixes the train+val parameters and filters through the test returns, so forecasts are genuinely out of sample.
- **Date-based comparison.** `evaluate.py` used to truncate LSTM, GARCH and XGBoost forecasts by position, which compared forecasts with the wrong days (the LSTM starts 60 days into the test set, XGBoost a few days in). Models are now aligned on common dates.
- **TFT scaler** is fit on the training split only.
- **Results regenerated on 2026-11-11** after these fixes and two more: VIX is now read correctly from newer yfinance (it returns a one-column DataFrame, which previously raised KeyError), and downloads are cached per run so evaluation makes two Yahoo requests instead of four. The TFT evaluation path is still experimental (it is not date-aligned).
