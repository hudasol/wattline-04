"""Issue-time feature builder: the information boundary in code.

``origin_features`` receives ``history`` - an array that *physically ends* at the data
cutoff (``history[-1]`` is the hour ``[issue - 1h, issue)``). It cannot read a future
value because it is never given one. The same function is used for training,
validation, the locked test, backtests and live serving, so there is no train/serve skew.

Every data feature is defined by the history positions it reads (``positions``); the
same definition drives the computation and the provenance table shown to the operator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_fr import CALENDAR_FEATURES, calendar_matrix

H = 24  # forecast horizons
MIN_HISTORY = 168

LAG_FEATURES = ["lag_24", "lag_48", "lag_168", "same_hour_mean_7d", "last_value"]
ROLLING_FEATURES = ["mean_24", "std_24", "max_24", "mean_168", "imputed_frac_24", "imputed_frac_168"]
ALL_FEATURES = LAG_FEATURES + ROLLING_FEATURES + CALENDAR_FEATURES

FEATURE_SETS: dict[str, list[str]] = {
    "lags_only": LAG_FEATURES,
    "lags_calendar": LAG_FEATURES + CALENDAR_FEATURES,
    "full": ALL_FEATURES,
}


class InsufficientHistory(ValueError):
    pass


def positions(name: str, h: int) -> np.ndarray:
    """History positions (negative indices) read by data feature ``name`` at horizon h.

    ``history[-k]`` is the hour starting at ``issue - k hours``. Horizon h targets the
    hour starting at ``issue + (h - 1) hours``, so a lag of L hours reads ``-(L + 1 - h)``.
    """
    if not 1 <= h <= H:
        raise ValueError(f"horizon {h} outside 1..{H}")
    if name == "lag_24":
        return np.array([-(25 - h)])
    if name == "lag_48":
        return np.array([-(49 - h)])
    if name == "lag_168":
        return np.array([-(169 - h)])
    if name == "same_hour_mean_7d":
        return np.array([-(24 * k + 1 - h) for k in range(1, 8)])
    if name == "last_value":
        return np.array([-1])
    if name in ("mean_24", "std_24", "max_24", "imputed_frac_24"):
        return np.arange(-24, 0)
    if name in ("mean_168", "imputed_frac_168"):
        return np.arange(-168, 0)
    raise KeyError(name)


def target_times(origin_utc: pd.Timestamp) -> pd.DatetimeIndex:
    """Interval starts of the 24 target hours: issue + (h-1) h, absolute time."""
    return pd.date_range(origin_utc, periods=H, freq="h").as_unit("ns")


def origin_features(history: np.ndarray, history_imputed: np.ndarray, origin_utc: pd.Timestamp,
                    local_tz: str) -> np.ndarray:
    """(24, len(ALL_FEATURES)) feature matrix for one issue time."""
    history = np.asarray(history, dtype=float)
    if len(history) < MIN_HISTORY:
        raise InsufficientHistory(f"need {MIN_HISTORY} h of history, got {len(history)}")
    if np.isnan(history[-MIN_HISTORY:]).any():
        raise ValueError("history must be causally filled before feature building")
    imp = np.asarray(history_imputed, dtype=float)
    hs = np.arange(1, H + 1)
    out = np.empty((H, len(ALL_FEATURES)))
    last24, last168 = history[-24:], history[-168:]
    cols = {
        "lag_24": history[-(25 - hs)],
        "lag_48": history[-(49 - hs)],
        "lag_168": history[-(169 - hs)],
        "same_hour_mean_7d": np.mean([history[-(24 * k + 1 - hs)] for k in range(1, 8)], axis=0),
        "last_value": np.full(H, history[-1]),
        "mean_24": np.full(H, last24.mean()),
        "std_24": np.full(H, last24.std()),
        "max_24": np.full(H, last24.max()),
        "mean_168": np.full(H, last168.mean()),
        "imputed_frac_24": np.full(H, imp[-24:].mean()),
        "imputed_frac_168": np.full(H, imp[-168:].mean()),
    }
    for j, name in enumerate(LAG_FEATURES + ROLLING_FEATURES):
        out[:, j] = cols[name]
    out[:, len(cols):] = calendar_matrix(target_times(origin_utc), local_tz)
    return out


def build_design(filled: np.ndarray, imputed: np.ndarray, index_utc: pd.DatetimeIndex,
                 origin_pos: np.ndarray, local_tz: str) -> np.ndarray:
    """Stack origin_features for many origins; each sees only filled[:p]."""
    X = np.empty((len(origin_pos), H, len(ALL_FEATURES)))
    for i, p in enumerate(origin_pos):
        X[i] = origin_features(filled[:p], imputed[:p], index_utc[p], local_tz)
    return X


def targets(observed: np.ndarray, origin_pos: np.ndarray) -> np.ndarray:
    """(n, 24) actual energy of the target hours; NaN where the hour was not observed."""
    return np.stack([observed[p : p + H] for p in origin_pos])


def provenance(origin_utc: pd.Timestamp, local_tz: str) -> list[dict]:
    """Latest/earliest input hour each feature reads, for the operator report and tests."""
    rows = []
    for name in LAG_FEATURES + ROLLING_FEATURES:
        pos = np.concatenate([positions(name, h) for h in range(1, H + 1)])
        earliest = origin_utc + pd.Timedelta(hours=int(pos.min()))
        latest_start = origin_utc + pd.Timedelta(hours=int(pos.max()))
        latest_end = latest_start + pd.Timedelta(hours=1)
        rows.append({
            "feature": name,
            "earliest_input_hour_start": earliest.tz_convert(local_tz).isoformat(),
            "latest_input_hour_end": latest_end.tz_convert(local_tz).isoformat(),
            "within_cutoff": bool(latest_end <= origin_utc),
        })
    rows.append({
        "feature": "calendar (" + ", ".join(CALENDAR_FEATURES[:2]) + ", ...)",
        "earliest_input_hour_start": "deterministic calendar",
        "latest_input_hour_end": "no measured data",
        "within_cutoff": True,
    })
    return rows
