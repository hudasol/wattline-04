"""Direct multi-horizon models: one regressor per horizon h = 1..24.

The ridge model is stored as plain JSON (scaler means/scales, coefficients, intercepts),
so the frozen artifact is human-readable, hashable and loads without pickle. The
optional boosted-tree contender is only serialised (pickle) if it is ever adopted.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .features import ALL_FEATURES, FEATURE_SETS, H


@dataclass(frozen=True)
class ModelSpec:
    family: str            # "ridge" | "hgb"
    feature_set: str
    target: str = "raw"    # "raw" | "log1p"
    alpha: float = 1.0

    @property
    def name(self) -> str:
        if self.family == "ridge":
            return f"ridge[{self.feature_set},{self.target},alpha={self.alpha:g}]"
        return f"hgb[{self.feature_set},{self.target}]"

    def to_dict(self) -> dict:
        return {"family": self.family, "feature_set": self.feature_set,
                "target": self.target, "alpha": self.alpha}


def _fwd(y: np.ndarray, target: str) -> np.ndarray:
    return np.log1p(y) if target == "log1p" else y


def _inv(z: np.ndarray, target: str) -> np.ndarray:
    return np.expm1(z) if target == "log1p" else z


def _cols(feature_set: str) -> np.ndarray:
    names = FEATURE_SETS[feature_set]
    return np.array([ALL_FEATURES.index(n) for n in names])


@dataclass
class RidgeDirect:
    spec: ModelSpec
    feature_names: list[str]
    mean: np.ndarray        # (24, F)
    scale: np.ndarray       # (24, F)
    coef: np.ndarray        # (24, F)
    intercept: np.ndarray   # (24,)
    n_train: list[int] = field(default_factory=list)
    fit_seconds: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        cols = _cols(self.spec.feature_set)
        Z = (X[:, :, cols] - self.mean[None]) / self.scale[None]
        out = np.einsum("nhf,hf->nh", Z, self.coef) + self.intercept[None]
        return np.clip(_inv(out, self.spec.target), 0.0, None)

    # ---- JSON artifact ----------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "kind": "RidgeDirect",
            "spec": self.spec.to_dict(),
            "feature_names": self.feature_names,
            "mean": self.mean.round(12).tolist(),
            "scale": self.scale.round(12).tolist(),
            "coef": self.coef.round(12).tolist(),
            "intercept": self.intercept.round(12).tolist(),
            "n_train_per_horizon": self.n_train,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RidgeDirect":
        spec = ModelSpec(**d["spec"])
        if d["feature_names"] != FEATURE_SETS[spec.feature_set]:
            raise ValueError("model feature list does not match this code's feature set")
        return cls(spec, d["feature_names"], np.array(d["mean"]), np.array(d["scale"]),
                   np.array(d["coef"]), np.array(d["intercept"]), d.get("n_train_per_horizon", []))


@dataclass
class HGBDirect:
    spec: ModelSpec
    models: list
    fit_seconds: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        cols = _cols(self.spec.feature_set)
        out = np.stack([m.predict(X[:, h, cols]) for h, m in enumerate(self.models)], axis=1)
        return np.clip(_inv(out, self.spec.target), 0.0, None)


def fit_model(spec: ModelSpec, X: np.ndarray, Y: np.ndarray, rows: np.ndarray,
              seed: int = 0, hgb_params: dict | None = None):
    """Fit 24 per-horizon models on origins ``rows`` where the target is observed."""
    cols = _cols(spec.feature_set)
    t0 = time.perf_counter()
    if spec.family == "ridge":
        F = len(cols)
        mean, scale = np.zeros((H, F)), np.ones((H, F))
        coef, icpt, n_train = np.zeros((H, F)), np.zeros(H), []
        for h in range(H):
            ok = rows & ~np.isnan(Y[:, h])
            Xh, yh = X[ok][:, h, cols], _fwd(Y[ok, h], spec.target)
            mu, sd = Xh.mean(axis=0), Xh.std(axis=0)
            sd[sd < 1e-12] = 1.0
            r = Ridge(alpha=spec.alpha).fit((Xh - mu) / sd, yh)
            mean[h], scale[h], coef[h], icpt[h] = mu, sd, r.coef_, r.intercept_
            n_train.append(int(ok.sum()))
        m = RidgeDirect(spec, FEATURE_SETS[spec.feature_set], mean, scale, coef, icpt, n_train)
    elif spec.family == "hgb":
        p = hgb_params or {}
        models = []
        for h in range(H):
            ok = rows & ~np.isnan(Y[:, h])
            reg = HistGradientBoostingRegressor(
                loss="absolute_error", random_state=seed,
                max_iter=p.get("max_iter", 300), learning_rate=p.get("learning_rate", 0.05),
                max_leaf_nodes=p.get("max_leaf_nodes", 15),
            )
            models.append(reg.fit(X[ok][:, h, cols], _fwd(Y[ok, h], spec.target)))
        m = HGBDirect(spec, models)
    else:
        raise ValueError(spec.family)
    m.fit_seconds = time.perf_counter() - t0
    return m


def save_model(model, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, RidgeDirect):
        path = path.with_suffix(".json")
        path.write_text(json.dumps(model.to_json(), indent=1), encoding="utf-8", newline="\n")
    else:
        path = path.with_suffix(".pkl")
        with open(path, "wb") as fh:
            pickle.dump(model, fh)
    return path


def load_model(path: Path):
    if path.suffix == ".json":
        return RidgeDirect.from_json(json.loads(path.read_text(encoding="utf-8")))
    with open(path, "rb") as fh:  # only ever our own artifact, hash-checked by the caller
        return pickle.load(fh)
