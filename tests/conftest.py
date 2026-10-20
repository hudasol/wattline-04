from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import hourly_series  # noqa: E402

from wattline.advisory import AdvisoryRule  # noqa: E402
from wattline.config import load_config  # noqa: E402
from wattline.intervals import LogConformal  # noqa: E402
from wattline.model import ModelSpec, fit_model  # noqa: E402
from wattline.monitor import MonitorRef  # noqa: E402
from wattline.serving import Artifacts  # noqa: E402

REAL_CONFIG = Path(__file__).resolve().parents[1] / "config" / "wattline.toml"

# A small synthetic project: ~4 months of minutes spanning the 2010-03-28 DST change.
SYNTH_OVERRIDES = {
    "sha256": None,  # filled by the e2e fixture
    "train_end": "2010-04-10T00:00",
    "val_end": "2010-04-25T00:00",
    "test_end": "2010-05-10T00:00",
    "first_fold": "2010-03-01",
    "last_fold_end": "2010-04-01",
    "calibration_days": "14",
}


def write_config(root: Path, overrides: dict[str, str]) -> Path:
    text = REAL_CONFIG.read_text(encoding="utf-8")
    for key, val in overrides.items():
        if val is None:
            continue
        quoted = val if re.fullmatch(r"[0-9.]+", val) else f'"{val}"'
        text, n = re.subn(rf"(?m)^({key}\s*=\s*)(\"[^\"]*\"|[0-9.]+)", rf"\g<1>{quoted}", text)
        assert n == 1, f"override {key} matched {n} times"
    cfg_path = root / "config" / "wattline.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(text, encoding="utf-8")
    return cfg_path


@pytest.fixture
def real_cfg():
    return load_config(REAL_CONFIG)


@pytest.fixture
def serving_setup(tmp_path):
    """Config + small fitted Artifacts for serving tests (no files needed)."""
    cfg = load_config(write_config(tmp_path, {}))
    from wattline.clean import causal_fill
    from wattline.features import build_design, targets
    s = hourly_series("2010-01-01T23:00Z", 24 * 90)
    vals, idx = s.to_numpy(), s.index
    prof = np.full(168, float(np.median(vals)))
    fill = causal_fill(vals, idx, prof, "Europe/Paris")
    loc = idx.tz_convert("Europe/Paris")
    pos = np.flatnonzero((loc.hour == 0) & (np.arange(len(idx)) >= 168) & (np.arange(len(idx)) + 24 <= len(idx)))
    X = build_design(fill.filled, fill.imputed, idx, pos, "Europe/Paris")
    Y = targets(vals, pos)
    rows = np.ones(len(pos), bool)
    model = fit_model(ModelSpec("ridge", "full", "log1p", 10.0), X, Y, rows)
    P = model.predict(X)
    conf = LogConformal.fit(Y, P, extra_levels=[0.5, 0.6, 0.7, 0.8, 0.9])
    hi_thr, lo_thr = np.quantile(vals, [0.9, 0.25])
    art = Artifacts(model=model, conformal=conf,
                    rule=AdvisoryRule(float(hi_thr), float(lo_thr), 0.7, 0.9, 0.25),
                    profile=prof, train_median=float(np.median(vals)),
                    monitor_ref=MonitorRef.fit(Y, P), monitor_adopted=True,
                    primary_baseline="seasonal_naive_week", manifest_sha256="0" * 64)
    return cfg, art, s


def issue_after(series: pd.Series, days_before_end: int = 1) -> pd.Timestamp:
    """A local-midnight issue time inside the series."""
    wall = series.index[-1].tz_convert("Europe/Paris").tz_localize(None).normalize()
    return (wall - pd.Timedelta(days=days_before_end)).tz_localize("Europe/Paris").tz_convert("UTC")
