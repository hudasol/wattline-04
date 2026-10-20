"""Adaptive drift / data-quality monitor.

At issue time the frozen model's forecasts for the previous ``window_days`` issue times
can be scored, because their actuals are now in the past. If those recent errors are
much larger than on validation, the forecast is not as trustworthy as its validation
interval claims. The monitor then (a) stretches the interval - it never narrows it -
and (b) raises an alarm that downgrades the report to UNCERTAIN, so the operator is
not told NORMAL/LOW on the strength of an interval that recently stopped working.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intervals import log_residuals


@dataclass
class MonitorRef:
    val_mae: float
    val_halfwidth: float       # 80th pct of |log residual - median| on validation
    val_median_resid: float

    @classmethod
    def fit(cls, y_val: np.ndarray, yhat_val: np.ndarray) -> "MonitorRef":
        r = log_residuals(np.ravel(y_val), np.ravel(yhat_val))
        med = float(np.median(r))
        ok = ~np.isnan(np.ravel(y_val))
        mae = float(np.mean(np.abs(np.ravel(y_val)[ok] - np.ravel(yhat_val)[ok])))
        return cls(mae, float(np.quantile(np.abs(r - med), 0.8)), med)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MonitorState:
    active: bool
    n_scored: int
    recent_mae: float
    mae_ratio: float
    scale: float
    alarm: bool
    note: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def monitor_state(past_y: np.ndarray, past_pred: np.ndarray, ref: MonitorRef,
                  min_scored: int, alarm_ratio: float, max_widen: float) -> MonitorState:
    """past_y / past_pred: (k, 24) actuals and forecasts of the last k issue times."""
    y, p = np.ravel(past_y), np.ravel(past_pred)
    ok = ~np.isnan(y) & ~np.isnan(p)
    n = int(ok.sum())
    if n < min_scored:
        return MonitorState(False, n, float("nan"), float("nan"), 1.0, False,
                            f"monitor inactive: only {n} recent scored hours (< {min_scored})")
    recent_mae = float(np.mean(np.abs(y[ok] - p[ok])))
    r = log_residuals(y[ok], p[ok])
    hw = float(np.quantile(np.abs(r - ref.val_median_resid), 0.8))
    scale = float(np.clip(hw / ref.val_halfwidth, 1.0, max_widen))
    ratio = recent_mae / ref.val_mae
    alarm = ratio > alarm_ratio
    note = (f"recent {n} h MAE {recent_mae:.3f} kWh = {ratio:.2f}x validation; "
            f"interval stretch x{scale:.2f}")
    if alarm:
        note = "DRIFT ALARM: " + note
    return MonitorState(True, n, recent_mae, ratio, scale, alarm, note)


HOUR_NS = 3_600_000_000_000


def batch_monitor(Y: np.ndarray, P: np.ndarray, target_ns: np.ndarray, origin_ns: np.ndarray,
                  ref: MonitorRef, window: int, min_scored: int, alarm_ratio: float,
                  max_widen: float, rows: np.ndarray | None = None) -> list[MonitorState]:
    """Monitor state for each requested row (consecutive daily origins), using the
    previous ``window`` rows - and only target hours that have *ended* by that row's
    issue time. (On the spring DST day the 24th hour of yesterday's forecast ends after
    today's issue time; using it would be a one-hour leak.)"""
    out = []
    for i in (range(len(Y)) if rows is None else np.flatnonzero(rows)):
        lo = max(0, i - window)
        y = Y[lo:i].copy()
        y[target_ns[lo:i] + HOUR_NS > origin_ns[i]] = np.nan
        out.append(monitor_state(y, P[lo:i], ref, min_scored, alarm_ratio, max_widen))
    return out
