"""Evaluation math, written out explicitly and unit-tested against hand calculations.

All functions take (n_origins, 24) arrays. A cell counts only where the actual is
observed (not NaN) *and* the origin is eligible; every model and baseline is scored on
exactly that same mask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = _mask(y, yhat, mask)
    return float(np.mean(np.abs(y[m] - yhat[m])))


def rmse(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = _mask(y, yhat, mask)
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


def _mask(y, yhat, mask):
    m = ~np.isnan(y) & ~np.isnan(yhat)
    if mask is not None:
        m &= mask
    if not m.any():
        raise ValueError("empty evaluation mask")
    return m


def improvement(base_mae: float, model_mae: float) -> float:
    """Relative MAE reduction vs a baseline (0.10 = 10 % better)."""
    return (base_mae - model_mae) / base_mae


def coverage(y, lo, hi, mask=None) -> float:
    m = _mask(y, lo, mask)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))


def median_width(lo, hi, mask=None) -> float:
    m = ~np.isnan(lo) & ~np.isnan(hi)
    if mask is not None:
        m &= mask
    return float(np.median(hi[m] - lo[m]))


def per_horizon(y, preds: dict[str, np.ndarray], mask) -> pd.DataFrame:
    rows = []
    for h in range(y.shape[1]):
        row = {"horizon": h + 1, "n": int((mask[:, h] & ~np.isnan(y[:, h])).sum())}
        for name, p in preds.items():
            row[f"{name}_mae"] = mae(y[:, h], p[:, h], mask[:, h])
        rows.append(row)
    return pd.DataFrame(rows)


def per_group(y, preds: dict[str, np.ndarray], mask, groups: np.ndarray, key: str) -> pd.DataFrame:
    rows = []
    for g in pd.unique(groups.ravel()):
        gm = mask & (groups == g)
        if not (gm & ~np.isnan(y)).any():
            continue
        row = {key: g, "n": int((gm & ~np.isnan(y)).sum())}
        for name, p in preds.items():
            row[f"{name}_mae"] = mae(y, p, gm)
            row[f"{name}_rmse"] = rmse(y, p, gm)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key).reset_index(drop=True)


def per_origin_mae(y, yhat, mask) -> np.ndarray:
    out = np.full(y.shape[0], np.nan)
    for i in range(y.shape[0]):
        m = mask[i] & ~np.isnan(y[i])
        if m.any():
            out[i] = float(np.mean(np.abs(y[i, m] - yhat[i, m])))
    return out


def flag_scores(actual_high: np.ndarray, flagged: np.ndarray, mask) -> dict:
    """Precision / recall / F1 of hour-level HIGH_DEMAND flags."""
    m = mask & ~np.isnan(actual_high.astype(float))
    a, f = actual_high[m].astype(bool), flagged[m].astype(bool)
    tp, fp, fn = int((a & f).sum()), int((~a & f).sum()), int((a & ~f).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": int((~a & ~f).sum()),
            "precision": prec, "recall": rec, "f1": f1, "base_rate": float(a.mean())}


def peak_hit_rate(y, yhat, mask, k: int = 3) -> float:
    """Share of days whose actual peak hour is among the model's top-k predicted hours."""
    hits, n = 0, 0
    for i in range(y.shape[0]):
        m = mask[i] & ~np.isnan(y[i])
        if m.sum() < y.shape[1]:
            continue   # need the full day of actuals to know the true peak
        n += 1
        hits += int(np.argmax(y[i]) in np.argsort(-yhat[i])[:k])
    return hits / n if n else float("nan")
