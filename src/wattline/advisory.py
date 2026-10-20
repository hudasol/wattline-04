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
