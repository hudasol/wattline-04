"""Naive baselines, computed from the same truncated history as the model.

They are deliberately written against the raw history array (not the feature matrix)
so a test can check that the feature columns and the baselines agree independently.
"""

from __future__ import annotations

import numpy as np

from .features import H

NAIVE_BASELINES = ["last_value", "seasonal_naive_day", "seasonal_naive_week"]
REFERENCE_RULES = ["mean_7d_same_hour"]
ALL_BASELINES = NAIVE_BASELINES + REFERENCE_RULES


def baseline_forecast(name: str, history: np.ndarray) -> np.ndarray:
    """24 predictions for horizons 1..24 from history ending at the issue time."""
    hs = np.arange(1, H + 1)
    if name == "last_value":
        return np.full(H, history[-1], dtype=float)
    if name == "seasonal_naive_day":      # y(target - 24 h)
        return history[len(history) - 25 + hs].astype(float)
    if name == "seasonal_naive_week":     # y(target - 168 h)
        return history[len(history) - 169 + hs].astype(float)
    if name == "mean_7d_same_hour":
        return np.mean([history[len(history) - (24 * k + 1) + hs] for k in range(1, 8)], axis=0)
    raise KeyError(name)


def baseline_matrix(name: str, filled: np.ndarray, origin_pos: np.ndarray) -> np.ndarray:
    return np.stack([baseline_forecast(name, filled[:p]) for p in origin_pos])
