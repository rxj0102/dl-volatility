"""How stable is the LSTM result across training seeds?

Trains the LSTM once per seed, scores each run on the same test days as the
baselines, and writes outputs/lstm_seed_study.csv. Baseline rows are scored on
those same days so every QLIKE in the file is comparable.

Run: python -m evaluation.seed_study [n_seeds]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data.features import download_and_engineer
from evaluation.align import align_on_dates
from evaluation.evaluate import OUTPUT_DIR, get_baseline_preds, get_lstm_preds, get_naive_preds
from evaluation.metrics import diebold_mariano, qlike
from training.train_lstm import train_lstm


def summarise_seeds(target: pd.Series, lstm_runs: dict[int, pd.Series],
                    others: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per LSTM seed and per comparison model, on the common dates."""
    preds = {f"LSTM seed {s}": p for s, p in lstm_runs.items()}
    preds.update(others)
    aligned = align_on_dates(target, preds)
    y = aligned["target"].values
    best_other = min(others, key=lambda k: qlike(y, aligned[k].values))

    rows = []
    for name in preds:
        p = aligned[name].values
        row = {"model": name, "QLIKE": qlike(y, p)}
        if name.startswith("LSTM"):
            dm = diebold_mariano(y, p, aligned[best_other].values)
            row.update({"vs": best_other, "DM_stat": dm["dm_stat"], "DM_p_value": dm["p_value"]})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(n_days=len(aligned), start=aligned.index[0], end=aligned.index[-1])
    return out


def run_seed_study(seeds=range(5)) -> pd.DataFrame:
    df = download_and_engineer()
    baseline = get_baseline_preds(df)
    others = {"GARCH": baseline["garch"], "XGB": baseline["xgb"], **get_naive_preds(df)}

    runs = {}
    for s in seeds:
        name = f"lstm_seed{s}.pt"
        train_lstm(seed=s, ckpt_name=name)
        runs[s], _ = get_lstm_preds(ckpt_name=name)

    out = summarise_seeds(baseline["targets"], runs, others)
    out.round(6).to_csv(OUTPUT_DIR / "lstm_seed_study.csv", index=False)
    lstm = out[out["model"].str.startswith("LSTM")]["QLIKE"]
    print(f"Common test days: {out.attrs['n_days']} "
          f"({out.attrs['start'].date()} to {out.attrs['end'].date()})")
    print(out.round(4).to_string(index=False))
    print(f"LSTM QLIKE across {len(lstm)} seeds: mean {lstm.mean():.4f}, "
          f"min {lstm.min():.4f}, max {lstm.max():.4f}, sd {lstm.std(ddof=1):.4f}")
    return out


if __name__ == "__main__":
    run_seed_study(range(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
