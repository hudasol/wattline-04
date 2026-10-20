"""HIGH_DEMAND / NORMAL / LOW / UNCERTAIN labels for each forecast hour.

Thresholds are quantiles of *observed* train + validation hourly energy - never test.
The flag rule ("which predictive quantile must reach the HIGH threshold") is picked on
validation by F1 (peak-event study). When the report status is UNCERTAIN the interface
refuses to call any hour NORMAL or LOW; it may still warn HIGH_DEMAND.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIGH, NORMAL, LOW, UNCERTAIN = "HIGH_DEMAND", "NORMAL", "LOW", "UNCERTAIN"


@dataclass
class AdvisoryRule:
    high_kwh: float
    low_kwh: float
    flag_level: float       # predictive quantile compared with high_kwh
    high_quantile: float
    low_quantile: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def thresholds(values: np.ndarray, high_q: float, low_q: float) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    return float(np.quantile(v, high_q)), float(np.quantile(v, low_q))


def label_matrix(flag_quantile: np.ndarray, upper: np.ndarray, rule: AdvisoryRule,
                 uncertain_rows: np.ndarray) -> np.ndarray:
    """Vectorised hour_labels for (n, 24) arrays; uncertain_rows marks degraded reports."""
    return np.where(flag_quantile >= rule.high_kwh, HIGH,
                    np.where(np.asarray(uncertain_rows)[:, None], UNCERTAIN,
                             np.where(upper < rule.low_kwh, LOW, NORMAL)))


def false_confidence(y: np.ndarray, labels: np.ndarray, upper: np.ndarray, high_kwh: float,
                     mask: np.ndarray) -> dict:
    """Hours the tool called NORMAL/LOW although the actual was above the interval or HIGH."""
    m = mask & ~np.isnan(y)
    confident = np.isin(labels, [NORMAL, LOW]) & m
    wrong = confident & ((y > upper) | (y >= high_kwh))
    n_conf = int(confident.sum())
    return {"scored_hours": int(m.sum()), "confident_hours": n_conf,
            "confident_but_wrong": int(wrong.sum()),
            "rate_per_scored_hour": float(wrong.sum() / m.sum()) if m.any() else float("nan"),
            "share_of_confident_labels": float(wrong.sum() / n_conf) if n_conf else float("nan")}


def hour_labels(flag_quantile: np.ndarray, upper: np.ndarray, rule: AdvisoryRule,
                report_status: str) -> list[str]:
    labels = []
    for fq, up in zip(flag_quantile, upper):
        if fq >= rule.high_kwh:
            labels.append(HIGH)
        elif report_status != "OK":
            labels.append(UNCERTAIN)     # never claim NORMAL/LOW on degraded input
        elif up < rule.low_kwh:
            labels.append(LOW)
        else:
            labels.append(NORMAL)
    return labels
