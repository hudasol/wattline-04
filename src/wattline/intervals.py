"""80 % prediction intervals from validation residuals (split-conformal, log space).

Residual r = log1p(y) - log1p(yhat) on validation hours only. The interval is
``expm1(log1p(yhat) + [q_lo, q_hi])``: multiplicative in kWh, never negative, wider at
high load. Order statistics use the (n+1) finite-sample correction. Caveat recorded in
the model card: the 24 residuals of one day are correlated, so the conformal guarantee
is approximate; test coverage is the real check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def log_residuals(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(y) & ~np.isnan(yhat)
    return np.log1p(y[ok]) - np.log1p(np.clip(yhat[ok], 0, None))


def order_stat_quantile(r: np.ndarray, tau: float) -> float:
    """Conformal order statistic: ceil((n+1)tau)-th smallest (upper), floor for lower."""
    s = np.sort(r)
    n = len(s)
    if n == 0:
        raise ValueError("no residuals")
    if tau >= 0.5:
        k = min(n, math.ceil((n + 1) * tau))
    else:
        k = max(1, math.floor((n + 1) * tau))
    return float(s[k - 1])


@dataclass
class LogConformal:
    levels: dict[float, float]   # predictive quantile level -> residual quantile (log space)
    median: float
    n: int

    @classmethod
    def fit(cls, y: np.ndarray, yhat: np.ndarray, extra_levels: list[float] = ()) -> "LogConformal":
        r = log_residuals(np.ravel(y), np.ravel(yhat))
        taus = sorted({0.1, 0.5, 0.9, *extra_levels})
        return cls({t: order_stat_quantile(r, t) for t in taus}, float(np.median(r)), len(r))

    def quantile(self, yhat: np.ndarray, tau: float, scale: np.ndarray | float = 1.0) -> np.ndarray:
        """Predictive tau-quantile; ``scale`` >= 1 stretches spread around the median."""
        q = self.median + (self.levels[tau] - self.median) * np.asarray(scale)
        return np.clip(np.expm1(np.log1p(np.clip(yhat, 0, None)) + q), 0.0, None)

    def bounds(self, yhat: np.ndarray, level: float = 0.8, scale: np.ndarray | float = 1.0):
        lo_t, hi_t = round((1 - level) / 2, 10), round(1 - (1 - level) / 2, 10)
        return self.quantile(yhat, lo_t, scale), self.quantile(yhat, hi_t, scale)

    def to_dict(self) -> dict:
        return {"levels": {str(k): v for k, v in self.levels.items()},
                "median": self.median, "n": self.n}

    @classmethod
    def from_dict(cls, d: dict) -> "LogConformal":
        return cls({float(k): float(v) for k, v in d["levels"].items()}, float(d["median"]), int(d["n"]))
