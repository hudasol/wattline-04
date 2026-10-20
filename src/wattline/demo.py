"""Clean and deliberately broken meter exports for the acceptance demo.

Each file is a plausible real-world failure: a meter that stopped reporting, an export in
Wh, a duplicated reading, a too-short history, a stuck register and a timezone-naive dump.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .ingest import load_hourly


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", float_format="%.6f")
    return path


def write_demo_inputs(cfg: Config, issue: pd.Timestamp, out: Path) -> list[Path]:
    s = load_hourly(cfg)
    hist = s[(s.index < issue) & (s.index >= issue - pd.Timedelta(days=21))]
    base = pd.DataFrame({"timestamp_utc": [t.isoformat() for t in hist.index], "energy_kwh": hist.to_numpy()})
    paths = [_write(base, out / "clean.csv")]
    paths.append(_write(base.iloc[:-10], out / "stale_meter_10h.csv"))
    paths.append(_write(base.iloc[:-3], out / "late_meter_3h.csv"))
    paths.append(_write(base.rename(columns={"energy_kwh": "energy_wh"}).assign(energy_wh=base.energy_kwh * 1000),
                        out / "wrong_unit_wh.csv"))
    paths.append(_write(base.assign(energy_kwh=base.energy_kwh * 1000), out / "wh_mislabelled_as_kwh.csv"))
    paths.append(_write(pd.concat([base, base.iloc[[-5]].assign(energy_kwh=9.9)]), out / "duplicate_timestamp.csv"))
    paths.append(_write(base.iloc[-100:], out / "short_history_100h.csv"))
    stuck = base.copy()
    stuck.loc[stuck.index[-8:], "energy_kwh"] = 0.312
    paths.append(_write(stuck, out / "stuck_meter.csv"))
    naive = base.assign(timestamp_utc=[t.strftime("%Y-%m-%d %H:%M:%S") for t in hist.index])
    paths.append(_write(naive, out / "timezone_naive.csv"))
    return paths
