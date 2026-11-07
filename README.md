# dl-volatility: LSTM vs. Temporal Fusion Transformer for Realized Volatility Forecasting

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

| Model | QLIKE | RMSE (log-vol) | DM vs GARCH stat | DM p-value |
|-------|-------|----------------|------------------|------------|
| LSTM | — | — | — | — |
| TFT | — | — | — | — |
| GARCH | — | — | 0.000 | 1.000 |
| XGB | — | — | — | — |

> Run `python -m training.run_all` to populate this table with actual results.

---

## Regime-Conditional QLIKE

| Model | Low-vol QLIKE (VIX < 20) | High-vol QLIKE (VIX > 25) |
|-------|--------------------------|---------------------------|
| LSTM | — | — |
| TFT | — | — |
| GARCH | — | — |
| XGB | — | — |

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

## Resume Bullets

- Built end-to-end volatility forecasting pipeline comparing **LSTM** and **Temporal Fusion Transformer** (TFT) against GJR-GARCH and XGBoost on 14+ years of SPY daily data.
- Engineered Garman-Klass realized volatility features (5/10/22-day), VIX, log returns, and volume z-scores with rigorous look-ahead prevention.
- Implemented **QLIKE loss** as training objective and evaluation metric; validated model differences with **Diebold-Mariano tests** (Harvey-Leybourne-Newbold correction).
- Conducted **regime-conditional analysis** splitting test set by VIX quintile to assess model robustness in low vs. high volatility environments.
- Extracted and visualized **TFT encoder attention weights** to interpret which historical time steps drive next-day variance forecasts.
- Achieved early stopping with patience scheduling and full checkpoint/log persistence for reproducibility.

## Evaluation notes (important)

- **Target and alignment fixed.** Features were lagged one day *and* the target shifted `-1`, so the "next-day" target was really two days ahead, and the LSTM dataset paired each window with the target one row too late. Rows now hold information through `t-1` and the target is the realised GK variance of day `t`; each window is paired with its own last row's target.
- **GARCH baseline fixed.** `predict_garch` previously returned *training-period* forecasts (arch ignores `start` beyond the fitted sample). It now fixes the train+val parameters and filters through the test returns, so forecasts are genuinely out of sample.
- **Date-based comparison.** `evaluate.py` used to truncate LSTM, GARCH and XGBoost forecasts by position, which compared forecasts with the wrong days (the LSTM starts 60 days into the test set, XGBoost a few days in). Models are now aligned on common dates.
- **TFT scaler** is fit on the training split only.
- **Results in `outputs/` and any README numbers predate these fixes and must be regenerated** (`python -m training.train_lstm`, then `python -m evaluation.evaluate`). The TFT evaluation path is still experimental (it is not date-aligned).
