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

From `python -m training.run_all` then `python -m evaluation.evaluate outputs/checkpoints/tft_best.ckpt`, run on 2026-11-11 on CPU. Both networks use seed 0; the LSTM early-stopped at epoch 29 and the TFT at epoch 36. Test set: 1,412 common trading days, 2021-03-30 to 2026-11-10, after the 2010 to 2018 training and 2019 to 2020 validation periods. Target: the Garman-Klass variance of day t, forecast from information through t-1. Lower is better. Source files: `outputs/model_comparison.csv`, `outputs/regime_performance.csv`, `outputs/training_log_lstm.csv`.

The random walk (yesterday's Garman-Klass variance) and the 22-day mean are naive benchmarks on the same information set. DM is the Diebold-Mariano statistic on QLIKE losses; negative means the row's model has lower loss than the comparison model.

| Model | QLIKE | RMSE (log-vol) | DM vs GARCH | p | DM vs random walk | p |
|-------|-------|----------------|-------------|---|-------------------|---|
| LSTM | **0.3092** | 0.4251 | -8.08 | 0.0000 | -8.01 | 0.0000 |
| TFT | 0.3347 | **0.3943** | -4.06 | 0.0001 | -7.73 | 0.0000 |
| XGB | 0.3639 | 0.4822 | -3.92 | 0.0001 | -6.25 | 0.0000 |
| GARCH | 0.4084 | 0.5570 | 0.00 | 1.0000 | -5.00 | 0.0000 |
| 22d mean | 0.4398 | 0.4833 | 1.28 | 0.2001 | -4.55 | 0.0000 |
| Random walk | 0.6358 | 0.4773 | 5.00 | 0.0000 | 0.00 | 1.0000 |

### Seed stability (LSTM)

From `python -m evaluation.seed_study`, which retrains the LSTM with seeds 0 to 4 and scores every run on the same 1,413 test days (2021-03-30 to 2026-11-11; one more day than above, because the TFT is not in this comparison). Source: `outputs/lstm_seed_study.csv`.

| | QLIKE |
|---|---|
| LSTM, mean of 5 seeds | 0.3042 |
| LSTM, best and worst seed | 0.3013 to 0.3091 |
| LSTM, standard deviation across seeds | 0.0030 |
| XGB (best non-LSTM model on these days) | 0.3636 |

Every seed beats XGBoost on QLIKE with a Diebold-Mariano p-value below 0.001 (DM statistics -4.57 to -6.12).

**Reading this honestly.**
- The LSTM has the lowest QLIKE, and the seed study shows the edge is not a lucky draw: the spread across five seeds (0.0030) is small next to the gap to XGBoost (about 0.06).
- The TFT is second on QLIKE but has the lowest RMSE, so the ranking of the two networks depends on the loss. The TFT was trained once, with one seed, and no test of LSTM against TFT was run.
- GARCH beats the random walk on QLIKE but loses to it on RMSE. GARCH forecasts close-to-close variance while the target is range-based, so it is on a different scale; that is why the naive rows are in the table.

## Regime-Conditional QLIKE

VIX at the prior close below 20 (972 days) or above 25 (181 days); days in between are in neither column.

| Model | Low-vol QLIKE (VIX < 20) | High-vol QLIKE (VIX > 25) |
|-------|--------------------------|---------------------------|
| LSTM | 0.3290 | 0.2353 |
| TFT | 0.3575 | 0.2580 |
| XGB | 0.3855 | 0.2913 |
| GARCH | 0.4288 | 0.3766 |
| 22d mean | 0.4175 | 0.5555 |
| Random walk | 0.6709 | 0.5555 |

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

# LSTM seed stability (retrains 5 times)
python -m evaluation.seed_study
```

---

## Evaluation notes (important)

- **Target and alignment fixed.** Features were lagged one day *and* the target shifted `-1`, so the next-day target was really two days ahead, and the LSTM dataset paired each window with the target one row too late. Rows now hold information through `t-1` and the target is the realised GK variance of day `t`; each window is paired with its own last row's target.
- **GARCH baseline fixed.** `predict_garch` previously returned *training-period* forecasts (arch ignores `start` beyond the fitted sample). It now fixes the train+val parameters and filters through the test returns, so forecasts are genuinely out of sample.
- **Date-based comparison.** `evaluate.py` used to truncate LSTM, GARCH and XGBoost forecasts by position, which compared forecasts with the wrong days (the LSTM starts 60 days into the test set, XGBoost a few days in). Models are now aligned on common dates.
- **TFT scaler** is fit on the training split only.
- **TFT evaluation fixed (2026-11-11).** Validation and test sets were built with `predict=True`, which keeps one window per series: early stopping watched a single sample and the test produced a single forecast. The model was also fitted on the validation years it was then selected on. It now fits on training rows only, validates and tests on one rolling window per day, and returns forecasts indexed by date. Because a TFT encoder only sees rows before the decoder step, its target is lagged one row so it uses the same information as the LSTM (features through day t-1 for the variance of day t).
- **LSTM output scale fixed (2026-11-11).** The head ended in Softplus, which starts near 0.7 while daily variance is near 1e-4. Training was unstable: in a five-seed check, four seeds never improved on epoch 1, and three of them scored worse than the random walk. The head now predicts log variance, with its bias starting at the log of the mean training variance. The earlier single-run result (QLIKE 0.3061) came from the one seed in five that escaped; the numbers above replace it.
- **Results regenerated on 2026-11-11** after these fixes and two more: VIX is now read correctly from newer yfinance (it returns a one-column DataFrame, which previously raised KeyError), and downloads are cached per run so evaluation makes two Yahoo requests instead of four.
