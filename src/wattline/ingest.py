"""Raw UCI minute file -> canonical hourly energy table, with an audit trail.

Nothing in this module is silent: every row that is dropped, merged, set missing or
flagged is counted in the ingest report. The raw zip is never modified.

Canonical hourly table (``data/interim/hourly.csv``):

    timestamp_utc      interval start, ISO 8601 with +00:00
    timestamp_local    same instant in Europe/Paris, ISO 8601 with offset (display only)
    energy_kwh         energy in the hour [t, t+1h) in kWh, blank when missing
    n_minutes          minutes with a valid reading inside the hour (0..60)
    outlier_flag       1 if above the train-only Q3 + k*IQR fence (kept, not removed)
"""

from __future__ import annotations

import io
import json
import os
import stat
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, sha256_file

RAW_COLUMNS = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


# --------------------------------------------------------------------------- download
def download(cfg: Config, force: bool = False) -> Path:
    """The only function in wattline that touches the network."""
    target = cfg.raw_zip
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(f"raw snapshot already present: {target}")
        verify_snapshot(cfg)
        return target
    part = target.with_suffix(".part")
    print(f"downloading {cfg.data.url}")
    with urllib.request.urlopen(cfg.data.url, timeout=120) as resp, open(part, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    with zipfile.ZipFile(part) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad} in downloaded zip")
    if target.exists():
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
        target.unlink()
    part.rename(target)
    os.chmod(target, stat.S_IREAD)  # read-only raw snapshot
    verify_snapshot(cfg)
    return target


def verify_snapshot(cfg: Config) -> str:
    digest = sha256_file(cfg.raw_zip)
    expected = cfg.data.sha256
    if expected == "PENDING":
        print(f"sha256 = {digest}  (no expected hash configured yet)")
    elif digest != expected:
        raise RuntimeError(
            f"raw snapshot hash mismatch: got {digest}, config expects {expected}. "
            "The upstream file changed or the download is corrupt; refusing to continue."
        )
    else:
        print(f"raw snapshot verified: sha256 {digest}")
    return digest


# --------------------------------------------------------------------------- read raw
def read_raw_minutes(zip_path: Path, member: str) -> pd.DataFrame:
    """Read the semicolon file exactly as published; '?' becomes NaN."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="ascii")
        df = pd.read_csv(
            text,
            sep=";",
            na_values=["?", ""],  # missing rows look like "?;?;...;?;" (last field empty)
            keep_default_na=False,
            dtype={c: "float64" for c in RAW_COLUMNS[2:]} | {"Date": str, "Time": str},
        )
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(f"unexpected raw columns: {list(df.columns)}")
    df.insert(
        0,
        "logger_time",
        pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"),
    )
    return df.drop(columns=["Date", "Time"])


# --------------------------------------------------------------------------- clock
@dataclass
class ClockDecision:
    kind: str                       # "wall_clock_regridded" | "wall_clock_dst" | "fixed_offset"
    utc_offset_hours: float         # only for fixed_offset
    dst_day_row_counts: dict        # transition date -> logger rows on that date
    voltage_shift_minutes: dict     # transition date -> best voltage-profile shift
    spring_hour_duplicate_frac: dict  # spring date -> share of 02:mm rows identical to 03:mm
    rationale: str


def dst_transition_dates(years: range, tz: str) -> list[pd.Timestamp]:
    """Local dates on which the zone changes offset (found from the tz database)."""
    out = []
    for year in years:
        hours = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", tz="UTC")
        off = hours.tz_convert(tz).map(lambda t: t.utcoffset())
        change = np.flatnonzero(np.asarray(off[1:]) != np.asarray(off[:-1])) + 1
        out.extend(pd.Timestamp(hours[i].tz_convert(tz).date()) for i in change)
    return out


def _voltage_profile(minutes: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Mean intraday voltage deviation in 5-min slots (logger time).

    Grid voltage is driven by the aggregate load of the whole local network, i.e. by
    thousands of households living on the legal wall clock. If the logger clock ignored
    DST, this profile would move by ~60 min across each transition in logger time."""
    sel = minutes[(minutes["logger_time"] >= start) & (minutes["logger_time"] < end)]
    v = sel["Voltage"] - sel.groupby(sel["logger_time"].dt.normalize())["Voltage"].transform("mean")
    slot = (sel["logger_time"].dt.hour * 60 + sel["logger_time"].dt.minute) // 5
    prof = v.groupby(slot).mean().reindex(range(288))
    return prof.interpolate(limit_direction="both").to_numpy()


def _spring_duplicate_frac(minutes: pd.DataFrame, day: pd.Timestamp) -> float:
    """Share of the non-existent 02:mm rows that exactly equal the 03:mm rows."""
    t = minutes.set_index("logger_time")
    a = t.loc[day + pd.Timedelta(hours=2): day + pd.Timedelta(hours=2, minutes=59)]
    b = t.loc[day + pd.Timedelta(hours=3): day + pd.Timedelta(hours=3, minutes=59)]
    if len(a) != 60 or len(b) != 60:
        return float("nan")
    same = (a.to_numpy() == b.to_numpy()) | (np.isnan(a.to_numpy()) & np.isnan(b.to_numpy()))
    return float(same.all(axis=1).mean())


def detect_clock(minutes: pd.DataFrame, local_tz: str) -> ClockDecision:
    years = range(minutes["logger_time"].min().year, minutes["logger_time"].max().year + 1)
    counts: dict[str, int] = {}
    shifts: dict[str, int] = {}
    dups: dict[str, float] = {}
    per_date = minutes["logger_time"].dt.normalize().value_counts()
    first_day = minutes["logger_time"].min().normalize() + pd.Timedelta(days=22)
    last_day = minutes["logger_time"].max().normalize() - pd.Timedelta(days=22)
    for d in dst_transition_dates(years, local_tz):
        if not (first_day <= d <= last_day):
            continue
        key = str(d.date())
        counts[key] = int(per_date.get(d, 0))
        before = _voltage_profile(minutes, d - pd.Timedelta(days=21), d)
        after = _voltage_profile(minutes, d + pd.Timedelta(days=1), d + pd.Timedelta(days=22))
        errs = {s: float(np.mean((after - np.roll(before, -s)) ** 2)) for s in range(-24, 25)}
        shifts[key] = 5 * min(errs, key=errs.get)   # after[t] ~ before[t + shift]
        if d.month == 3:
            dups[key] = _spring_duplicate_frac(minutes, d)
    vals = np.array(list(counts.values()))
    med_shift = float(np.median(np.abs(list(shifts.values())))) if shifts else float("nan")
    follows_wall_clock = med_shift <= 15
    if len(vals) and np.all(vals == 1440) and follows_wall_clock:
        dup_txt = ", ".join(f"{k}: {v:.0%}" for k, v in dups.items())
        rationale = (
            "Every DST-transition date has exactly 1440 logger rows, yet the intraday grid-voltage "
            f"profile does not move across any transition (median |shift| {med_shift:.0f} min; a "
            "logger ignoring DST would show ~60 min). The logger therefore follows the Europe/Paris "
            "wall clock and the published file was re-gridded to 1440 rows/day. On spring-forward "
            "days the non-existent 02:00-02:59 rows are exact copies of 03:00-03:59 "
            f"({dup_txt} identical), so they are dropped. On autumn days the repeated 02:00-02:59 "
            "hour appears only once and cannot be attributed to either occurrence, so those rows "
            "are dropped too and both UTC hours become missing. No hour is shifted or guessed."
        )
        return ClockDecision("wall_clock_regridded", float("nan"), counts, shifts, dups, rationale)
    if len(vals) and set(vals.tolist()) <= {1380, 1500}:
        return ClockDecision("wall_clock_dst", float("nan"), counts, shifts, dups,
                             "DST dates have 1380/1500 rows: logger follows Europe/Paris wall clock.")
    if len(vals) and np.all(vals == 1440) and 45 <= med_shift <= 75:
        return ClockDecision("fixed_offset", 1.0, counts, shifts, dups,
                             "1440 rows on DST dates and a ~60 min voltage shift: fixed CET clock.")
    raise ValueError(f"cannot decide logger clock (rows {counts}, voltage shifts {shifts}); resolve manually")


def logger_to_utc(logger_time: pd.Series, decision: ClockDecision, local_tz: str) -> pd.Series:
    """Map logger time to UTC. Non-existent / ambiguous wall-clock minutes become NaT."""
    if decision.kind == "fixed_offset":
        return (logger_time - pd.Timedelta(hours=decision.utc_offset_hours)).dt.tz_localize("UTC")
    if decision.kind == "wall_clock_dst":
        loc = logger_time.dt.tz_localize(local_tz, ambiguous="infer", nonexistent="raise")
    else:
        loc = logger_time.dt.tz_localize(local_tz, ambiguous="NaT", nonexistent="NaT")
    return loc.dt.tz_convert("UTC")


# --------------------------------------------------------------------------- minutes -> hours
@dataclass
class MinuteAudit:
    raw_rows: int
    rows_missing_marker: int          # '?' in Global_active_power
    duplicate_timestamps: int
    conflicting_duplicates: int
    physical_range_violations: int
    unit_check_active_le_apparent_frac: float
    unit_check_submeter_le_total_frac: float
    unit_check_median_power_factor: float
    dst_rows_dropped: int = 0         # non-existent / ambiguous wall-clock minutes


def audit_and_clean_minutes(minutes: pd.DataFrame, max_kw: float) -> tuple[pd.DataFrame, MinuteAudit]:
    df = minutes.copy()
    raw_rows = len(df)
    n_missing = int(df["Global_active_power"].isna().sum())

    # unit verification at source: kW active <= V*A/1000 apparent; sub-meters (Wh per
    # minute) cannot exceed total active energy per minute (kW * 1000 / 60 Wh).
    ok = df["Global_active_power"].notna() & df["Voltage"].notna() & df["Global_intensity"].notna()
    apparent_kva = df.loc[ok, "Voltage"] * df.loc[ok, "Global_intensity"] / 1000.0
    active = df.loc[ok, "Global_active_power"]
    le_apparent = float(np.mean(active <= apparent_kva * 1.02 + 0.01))
    pf = float(np.median(active / apparent_kva.where(apparent_kva > 0)))
    sub_wh = df.loc[ok, ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1)
    sub_le_total = float(np.mean(sub_wh <= active * 1000.0 / 60.0 + 1.0))

    dup_mask = df["logger_time"].duplicated(keep=False)
    n_dup = int(df["logger_time"].duplicated().sum())
    n_conflict = 0
    if n_dup:
        grp = df[dup_mask].groupby("logger_time")["Global_active_power"]
        conflicting = grp.nunique(dropna=False) > 1
        n_conflict = int(conflicting.sum())
        bad_times = conflicting[conflicting].index
        df = df.drop_duplicates("logger_time", keep="first")
        df.loc[df["logger_time"].isin(bad_times), "Global_active_power"] = np.nan

    phys = (df["Global_active_power"] < 0) | (df["Global_active_power"] > max_kw)
    n_phys = int(phys.sum())
    df.loc[phys, "Global_active_power"] = np.nan

    audit = MinuteAudit(
        raw_rows=raw_rows,
        rows_missing_marker=n_missing,
        duplicate_timestamps=n_dup,
        conflicting_duplicates=n_conflict,
        physical_range_violations=n_phys,
        unit_check_active_le_apparent_frac=round(le_apparent, 6),
        unit_check_submeter_le_total_frac=round(sub_le_total, 6),
        unit_check_median_power_factor=round(pf, 4),
    )
    return df, audit


def minutes_to_hourly(utc: pd.Series, kw: pd.Series, min_minutes: int) -> pd.DataFrame:
    """Mean kW over the present minutes of each UTC hour -> kWh for that hour."""
    hour = utc.dt.floor("h")
    frame = pd.DataFrame({"hour": hour, "kw": kw.to_numpy()})
    g = frame.groupby("hour")["kw"]
    out = pd.DataFrame({"n_minutes": g.count(), "mean_kw": g.mean()})
    full = pd.date_range(out.index.min(), out.index.max(), freq="h", tz="UTC")
    out = out.reindex(full)
    out["n_minutes"] = out["n_minutes"].fillna(0).astype(int)
    out["energy_kwh"] = np.where(out["n_minutes"] >= min_minutes, out["mean_kw"] * 1.0, np.nan)
    out.index.name = "timestamp_utc"
    # trim leading/trailing hours that are not valid (partial first/last hour of the file)
    valid = np.flatnonzero(out["energy_kwh"].notna().to_numpy())
    out = out.iloc[valid[0] : valid[-1] + 1]
    return out[["energy_kwh", "n_minutes"]]


def missing_runs(series: pd.Series, local_tz: str) -> pd.DataFrame:
    miss = series.isna().to_numpy()
    if not miss.any():
        return pd.DataFrame(columns=["start_utc", "start_local", "end_utc_exclusive", "hours"])
    edges = np.diff(np.concatenate([[0], miss.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    idx = series.index
    rows = []
    for s, e in zip(starts, ends):
        end_ts = idx[e] if e < len(idx) else idx[-1] + pd.Timedelta(hours=1)
        rows.append(
            {
                "start_utc": idx[s].isoformat(),
                "start_local": idx[s].tz_convert(local_tz).isoformat(),
                "end_utc_exclusive": end_ts.isoformat(),
                "hours": int(e - s),
            }
        )
    return pd.DataFrame(rows)


def split_label(ts: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    lab = np.full(len(ts), "after_test", dtype=object)
    lab[ts < cfg.test_end_utc] = "test"
    lab[ts < cfg.val_end_utc] = "validation"
    lab[ts < cfg.train_end_utc] = "train"
    return lab


# --------------------------------------------------------------------------- orchestration
def ingest(cfg: Config) -> dict:
    if not cfg.raw_zip.exists():
        raise FileNotFoundError(f"{cfg.raw_zip} missing - run `wattline download` first")
    raw_sha = verify_snapshot(cfg)
    minutes = read_raw_minutes(cfg.raw_zip, cfg.data.member_name)
    decision = detect_clock(minutes, cfg.data.local_tz)
    cleaned, audit = audit_and_clean_minutes(minutes, cfg.data.max_physical_kw)
    utc = logger_to_utc(cleaned["logger_time"], decision, cfg.data.local_tz)
    nat = utc.isna().to_numpy()
    audit.dst_rows_dropped = int(nat.sum())
    hourly = minutes_to_hourly(utc[~nat], cleaned.loc[~nat, "Global_active_power"],
                               cfg.data.min_minutes_per_hour)
    return write_hourly(cfg, hourly, decision, audit, raw_sha, minutes["logger_time"])


def write_hourly(
    cfg: Config,
    hourly: pd.DataFrame,
    decision: ClockDecision,
    audit: MinuteAudit,
    raw_sha: str,
    logger_time: pd.Series,
) -> dict:
    idx = hourly.index
    labels = split_label(idx, cfg)
    e = hourly["energy_kwh"]
    train_obs = e[(labels == "train") & e.notna().to_numpy()]
    q1, q3 = np.quantile(train_obs, [0.25, 0.75])
    fence = float(q3 + cfg.data.outlier_iqr_k * (q3 - q1))
    outlier = (e > fence).to_numpy()

    local = idx.tz_convert(cfg.data.local_tz)
    out = pd.DataFrame(
        {
            "timestamp_utc": [t.isoformat() for t in idx],
            "timestamp_local": [t.isoformat() for t in local],
            "energy_kwh": e.round(6).to_numpy(),
            "n_minutes": hourly["n_minutes"].to_numpy(),
            "outlier_flag": outlier.astype(int),
        }
    )
    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.hourly_csv, index=False, lineterminator="\n")

    partial = (hourly["n_minutes"] >= cfg.data.min_minutes_per_hour) & (hourly["n_minutes"] < 60)
    per_split = {}
    for name in ["train", "validation", "test", "after_test"]:
        m = labels == name
        if not m.any():
            continue
        per_split[name] = {
            "hours": int(m.sum()),
            "missing_hours": int(e[m].isna().sum()),
            "partial_hours_used": int(partial[m].sum()),
            "outlier_flags": int(outlier[m].sum()),
            "first_utc": idx[m][0].isoformat(),
            "last_utc": idx[m][-1].isoformat(),
        }
    runs = missing_runs(e, cfg.data.local_tz)
    data_dir = cfg.results_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(data_dir / "missing_hour_runs.csv", index=False, lineterminator="\n")

    report = {
        "source": {
            "provider": "UCI Machine Learning Repository, dataset 235",
            "url": cfg.data.url,
            "licence": "CC BY 4.0",
            "raw_sha256": raw_sha,
            "raw_first_logger_time": str(logger_time.min()),
            "raw_last_logger_time": str(logger_time.max()),
        },
        "units": {
            "raw": "Global_active_power, minute-averaged active power, kW",
            "canonical": "energy_kwh = mean kW over present minutes x 1 h = kWh in hour [t, t+1h)",
            "verification": {
                "active_le_apparent_power_frac": audit.unit_check_active_le_apparent_frac,
                "submeters_le_total_energy_frac": audit.unit_check_submeter_le_total_frac,
                "median_power_factor": audit.unit_check_median_power_factor,
            },
        },
        "timezone": {
            "raw": "naive dd/mm/yyyy hh:mm:ss logger time, no zone declared",
            "decision": asdict(decision),
            "canonical": "UTC interval start; Europe/Paris derived for calendar/display only",
        },
        "minute_audit": asdict(audit),
        "hourly": {
            "first_utc": idx[0].isoformat(),
            "last_utc": idx[-1].isoformat(),
            "hours": len(idx),
            "observed_hours": int(e.notna().sum()),
            "missing_hours": int(e.isna().sum()),
            "duplicate_hours": int(idx.duplicated().sum()),
            "partial_hours_used": int(partial.sum()),
            "min_minutes_rule": cfg.data.min_minutes_per_hour,
            "outlier_fence_kwh_train_only": round(fence, 4),
            "outlier_flags_total": int(outlier.sum()),
            "missing_runs": int(len(runs)),
            "longest_missing_run_hours": int(runs["hours"].max()) if len(runs) else 0,
            "per_split": per_split,
        },
    }
    with open(data_dir / "ingest_report.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
    return report


def load_hourly(cfg: Config) -> pd.Series:
    """Load the canonical hourly table as a complete, tz-aware UTC series."""
    if not cfg.hourly_csv.exists():
        raise FileNotFoundError(f"{cfg.hourly_csv} missing - run `wattline ingest` first")
    df = pd.read_csv(cfg.hourly_csv)
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp_utc"], utc=True), name="timestamp_utc").as_unit("ns")
    s = pd.Series(df["energy_kwh"].to_numpy(dtype=float), index=idx, name="energy_kwh")
    expected = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    if not idx.equals(expected):
        raise ValueError("canonical hourly table is not a complete hourly UTC grid")
    return s
