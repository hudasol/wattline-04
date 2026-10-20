"""Synthetic data in the exact UCI raw format, with the publisher's DST re-gridding.

The generator logs a household on the Europe/Paris wall clock and then re-grids to
1440 rows per day the way the real file was: the non-existent spring 02:mm rows are
copies of 03:mm, and the autumn repeated hour appears once. Grid voltage follows the
wall clock (aggregate load), so ``detect_clock`` should find a 0-minute shift.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = ("Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;"
          "Sub_metering_1;Sub_metering_2;Sub_metering_3")


def daily_kw(minute_of_day: np.ndarray, dow: np.ndarray) -> np.ndarray:
    h = minute_of_day / 60.0
    base = 0.35 + 0.9 * np.exp(-((h - 7.5) ** 2) / 1.5) + 1.6 * np.exp(-((h - 20.0) ** 2) / 3.0)
    return base * np.where(dow >= 5, 1.25, 1.0)


def raw_frame(start: str, end: str, seed: int = 0, fixed_clock: bool = False,
              gaps: list[tuple[str, str]] = ()) -> pd.DataFrame:
    """Minute rows in logger time (naive) between start and end (end exclusive)."""
    rng = np.random.default_rng(seed)
    logger = pd.date_range(start, end, freq="min", inclusive="left")
    if fixed_clock:
        # logger runs on CET all year; behaviour follows the wall clock
        wall = (logger - pd.Timedelta(hours=1)).tz_localize("UTC").tz_convert("Europe/Paris").tz_localize(None)
    else:
        wall = logger
    mod = (wall.hour * 60 + wall.minute).to_numpy()
    kw = daily_kw(mod, wall.dayofweek.to_numpy()) * rng.lognormal(0, 0.25, len(logger))
    volt = 240 + 2.5 * np.cos(2 * np.pi * (mod - 4 * 60) / 1440) + rng.normal(0, 0.3, len(logger))
    df = pd.DataFrame({"logger_time": logger, "kw": kw, "volt": volt})
    if not fixed_clock:
        # publisher re-grid: spring non-existent 02:mm rows copy 03:mm
        lt = df["logger_time"]
        for day in pd.date_range(start, end, freq="D"):
            if day.month == 3 and day.dayofweek == 6 and (day + pd.Timedelta(days=7)).month == 4:
                a = (lt >= day + pd.Timedelta(hours=2)) & (lt < day + pd.Timedelta(hours=3))
                b = (lt >= day + pd.Timedelta(hours=3)) & (lt < day + pd.Timedelta(hours=4))
                df.loc[a, ["kw", "volt"]] = df.loc[b, ["kw", "volt"]].to_numpy()
    df["missing"] = False
    for g0, g1 in gaps:
        df.loc[(df.logger_time >= g0) & (df.logger_time < g1), "missing"] = True
    return df


def to_raw_text(df: pd.DataFrame) -> str:
    lines = [HEADER]
    d = df["logger_time"].dt.strftime("%d/%m/%Y;%H:%M:%S").to_numpy()
    for t, kw, v, miss in zip(d, df["kw"].to_numpy(), df["volt"].to_numpy(), df["missing"].to_numpy()):
        if miss:
            lines.append(f"{t};?;?;?;?;?;?;")
        else:
            amps = kw * 1000 / v / 0.98
            sub3 = min(kw * 1000 / 60 * 0.3, 20.0)
            lines.append(f"{t};{kw:.3f};0.100;{v:.2f};{amps:.1f};0.000;1.000;{sub3:.3f}")
    return "\n".join(lines) + "\n"


def write_raw_zip(df: pd.DataFrame, path: Path, member: str = "household_power_consumption.txt") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, to_raw_text(df))
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def hourly_series(start_utc: str, hours: int, seed: int = 1) -> pd.Series:
    """Synthetic hourly kWh on a UTC grid with a wall-clock daily pattern."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_utc, periods=hours, freq="h", tz="UTC")
    loc = idx.tz_convert("Europe/Paris")
    kw = daily_kw((loc.hour * 60 + 30).to_numpy(), loc.dayofweek.to_numpy())
    return pd.Series(kw * rng.lognormal(0, 0.2, hours), index=idx, name="energy_kwh")


def write_meter_csv(series: pd.Series, path: Path, value_col: str = "energy_kwh",
                    ts_format: str = "iso", extra: dict | None = None) -> Path:
    if ts_format == "iso":
        ts = [t.isoformat() for t in series.index]
    else:  # naive
        ts = [t.strftime("%Y-%m-%d %H:%M:%S") for t in series.index]
    df = pd.DataFrame({"timestamp_utc": ts, value_col: series.to_numpy()})
    for k, v in (extra or {}).items():
        df[k] = v
    df.to_csv(path, index=False, lineterminator="\n")
    return path
