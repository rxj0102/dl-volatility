import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from evaluation.seed_study import summarise_seeds  # noqa: E402


def test_summarise_seeds_scores_on_common_dates_and_tests_vs_best_baseline():
    idx = pd.bdate_range("2022-01-03", periods=300)
    rng = np.random.default_rng(1)
    y = pd.Series(np.exp(rng.normal(-9, 0.5, len(idx))), index=idx)
    good = y * np.exp(rng.normal(0, 0.1, len(idx)))
    bad = y * np.exp(rng.normal(0, 0.8, len(idx)))
    meh = y * np.exp(rng.normal(0, 0.4, len(idx)))
    runs = {0: good.iloc[10:], 1: good.iloc[10:] * 1.01}   # LSTM starts later
    out = summarise_seeds(y, runs, {"Bad": bad, "Meh": meh})

    assert out.attrs["n_days"] == len(idx) - 10           # inner join on dates
    lstm = out[out["model"].str.startswith("LSTM")]
    assert set(lstm["vs"]) == {"Meh"}                      # best non-LSTM model
    assert (lstm["DM_p_value"] < 0.05).all()
    assert out.loc[out["model"] == "Bad", "QLIKE"].item() > out.loc[out["model"] == "Meh", "QLIKE"].item()
