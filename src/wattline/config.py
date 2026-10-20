"""Typed access to config/wattline.toml.

All modelling decisions live in the TOML file so that the frozen manifest can hash them.
Paths in the file are resolved relative to the project root (the directory that holds
the ``config`` folder), never relative to the current working directory.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataCfg:
    url: str
    zip_name: str
    member_name: str
    sha256: str
    local_tz: str
    min_minutes_per_hour: int
    max_physical_kw: float
    outlier_iqr_k: float


@dataclass(frozen=True)
class SplitCfg:
    train_end: str
    val_end: str
    test_end: str
    issue_hour_local: int
    history_hours: int


@dataclass(frozen=True)
class ModelCfg:
    alphas: list[float]
    targets: list[str]
    feature_sets: list[str]
    hgb_min_relative_gain: float
    hgb_max_iter: int
    hgb_learning_rate: float
    hgb_max_leaf_nodes: int
    random_seed: int


@dataclass(frozen=True)
class IntervalCfg:
    level: float
    flag_levels: list[float]


@dataclass(frozen=True)
class AdvisoryCfg:
    high_quantile: float
    low_quantile: float
    top_k_peaks: int


@dataclass(frozen=True)
class ServingCfg:
    stale_uncertain_hours: int
    stale_unavailable_hours: int
    missing_168_uncertain: float
    missing_168_unavailable: float
    flatline_hours: int
    unit_ratio_limit: float


@dataclass(frozen=True)
class MonitorCfg:
    window_days: int
    min_scored_hours: int
    alarm_mae_ratio: float
    max_widen: float
    adopt_max_width_increase: float


@dataclass(frozen=True)
class BacktestCfg:
    first_fold: str
    last_fold_end: str
    calibration_days: int


@dataclass(frozen=True)
class Config:
    root: Path
    raw_dir: Path
    interim_dir: Path
    results_dir: Path
    reports_dir: Path
    data: DataCfg
    split: SplitCfg
    max_missing_frac_168: float
    model: ModelCfg
    interval: IntervalCfg
    advisory: AdvisoryCfg
    serving: ServingCfg
    monitor: MonitorCfg
    backtest: BacktestCfg
    source_path: Path = field(default=Path("config/wattline.toml"))

    # ---- derived helpers -------------------------------------------------------------
    @property
    def raw_zip(self) -> Path:
        return self.raw_dir / self.data.zip_name

    @property
    def hourly_csv(self) -> Path:
        return self.interim_dir / "hourly.csv"

    def local_ts(self, text: str) -> pd.Timestamp:
        """Parse a wall-clock string from the config as Europe/Paris and return UTC."""
        ts = pd.Timestamp(text)
        if ts.tzinfo is not None:
            raise ValueError(f"config timestamps are wall-clock strings, got offset in {text!r}")
        return ts.tz_localize(self.data.local_tz).tz_convert("UTC")

    @property
    def train_end_utc(self) -> pd.Timestamp:
        return self.local_ts(self.split.train_end)

    @property
    def val_end_utc(self) -> pd.Timestamp:
        return self.local_ts(self.split.val_end)

    @property
    def test_end_utc(self) -> pd.Timestamp:
        return self.local_ts(self.split.test_end)

    def decision_dict(self) -> dict:
        """Everything that affects modelling results (paths excluded)."""
        return {
            "data": asdict(self.data),
            "split": asdict(self.split),
            "max_missing_frac_168": self.max_missing_frac_168,
            "model": asdict(self.model),
            "interval": asdict(self.interval),
            "advisory": asdict(self.advisory),
            "serving": asdict(self.serving),
            "monitor": asdict(self.monitor),
            "backtest": asdict(self.backtest),
        }

    def decision_hash(self) -> str:
        blob = json.dumps(self.decision_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def find_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "config" / "wattline.toml").exists():
            return cand
    raise FileNotFoundError("could not find config/wattline.toml above " + str(here))


def load_config(path: Path | str | None = None) -> Config:
    if path is None:
        root = find_root()
        path = root / "config" / "wattline.toml"
    path = Path(path).resolve()
    root = path.parent.parent
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    p = raw["paths"]
    return Config(
        root=root,
        raw_dir=root / p["raw_dir"],
        interim_dir=root / p["interim_dir"],
        results_dir=root / p["results_dir"],
        reports_dir=root / p["reports_dir"],
        data=DataCfg(**raw["data"]),
        split=SplitCfg(**raw["split"]),
        max_missing_frac_168=float(raw["eligibility"]["max_missing_frac_168"]),
        model=ModelCfg(**raw["model"]),
        interval=IntervalCfg(**raw["interval"]),
        advisory=AdvisoryCfg(**raw["advisory"]),
        serving=ServingCfg(**raw["serving"]),
        monitor=MonitorCfg(**raw["monitor"]),
        backtest=BacktestCfg(**raw["backtest"]),
        source_path=path,
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
