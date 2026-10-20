"""Causal, train-only missing-value policy for model *inputs*.

Rule (docs/PLAN.md section 5): a missing hour t is filled with the already-filled value
at t - 168 h; if t - 168 h is before the start of the series it gets the train-only
median of its local hour-of-week. Only the past and train-fitted constants are used, so
the fill can run at any issue time without peeking. Filled values are flagged and never
used as targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEK = 168


def hour_of_week(index_utc: pd.DatetimeIndex, local_tz: str) -> np.ndarray:
    loc = index_utc.tz_convert(local_tz)
    return (loc.dayofweek.to_numpy() * 24 + loc.hour.to_numpy()).astype(int)


def fit_profile(values: np.ndarray, index_utc: pd.DatetimeIndex, train_end_utc: pd.Timestamp,
                local_tz: str) -> np.ndarray:
    """Median observed energy per local hour-of-week, from train hours only."""
    how = hour_of_week(index_utc, local_tz)
    train = (index_utc < train_end_utc) & ~np.isnan(values)
    if not train.any():
        raise ValueError("no observed training hours to fit the fallback profile")
    fallback = float(np.median(values[train]))
    prof = np.full(WEEK, fallback)
    for k in range(WEEK):
        sel = train & (how == k)
        if sel.any():
            prof[k] = float(np.median(values[sel]))
    return prof


@dataclass
class FillResult:
    filled: np.ndarray
    imputed: np.ndarray       # bool: True where the value was not observed
    n_lag168: int
    n_profile: int


def causal_fill(values: np.ndarray, index_utc: pd.DatetimeIndex, profile: np.ndarray,
                local_tz: str) -> FillResult:
    values = np.asarray(values, dtype=float)
    filled = values.copy()
    imputed = np.isnan(values)
    how = hour_of_week(index_utc, local_tz)
    n_lag = n_prof = 0
    for i in np.flatnonzero(imputed):
        if i >= WEEK:
            filled[i] = filled[i - WEEK]  # already filled: strictly causal recursion
            n_lag += 1
        else:
            filled[i] = profile[how[i]]
            n_prof += 1
    return FillResult(filled, imputed, n_lag, n_prof)


def fill_log(fill: FillResult, labels: np.ndarray) -> dict:
    out = {"policy": "t-168h (already filled) else train-only hour-of-week median; inputs only",
           "total_imputed": int(fill.imputed.sum()),
           "imputed_by_lag168": fill.n_lag168,
           "imputed_by_train_profile": fill.n_profile,
           "per_split": {}}
    for name in np.unique(labels):
        m = labels == name
        out["per_split"][str(name)] = {"hours": int(m.sum()), "imputed": int(fill.imputed[m].sum())}
    return out
