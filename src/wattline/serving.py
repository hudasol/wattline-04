"""Operator-facing forecast: input contract, failure states and the 24 h advisory.

Report status:
    OK           forecast and labels can be used as an advisory
    UNCERTAIN    forecast shown, but no hour may be called NORMAL or LOW
    UNAVAILABLE  nothing is forecast; the reasons say what to fix

Input contract (CSV): a timestamp column (``timestamp_utc`` or ``timestamp``) whose
values carry an explicit UTC offset, and an ``energy_kwh`` column with the energy of
the hour starting at that timestamp. One row per hour, no duplicates. An optional
``unit`` column must say ``kWh`` on every row. Anything else is refused, not guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .advisory import AdvisoryRule, hour_labels
from .baselines import baseline_forecast
from .clean import causal_fill
from .config import Config, sha256_file
from .features import H, MIN_HISTORY, origin_features, provenance, target_times
from .intervals import LogConformal
from .model import load_model
from .monitor import MonitorRef, MonitorState, monitor_state

OK, UNCERTAIN, UNAVAILABLE = "OK", "UNCERTAIN", "UNAVAILABLE"
EXIT_CODES = {OK: 0, UNCERTAIN: 3, UNAVAILABLE: 2}


class InputContractError(ValueError):
    """The input cannot be interpreted safely -> UNAVAILABLE."""


@dataclass
class Assessment:
    status: str = OK
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def downgrade(self, status: str, reason: str) -> None:
        order = {OK: 0, UNCERTAIN: 1, UNAVAILABLE: 2}
        if order[status] > order[self.status]:
            self.status = status
        self.reasons.append(f"[{status}] {reason}")


# --------------------------------------------------------------------------- artifacts
@dataclass
class Artifacts:
    model: object
    conformal: LogConformal
    rule: AdvisoryRule
    profile: np.ndarray
    train_median: float
    monitor_ref: MonitorRef
    monitor_adopted: bool
    primary_baseline: str
    manifest_sha256: str
    level: float = 0.8


def load_artifacts(cfg: Config) -> Artifacts:
    frozen = cfg.results_dir / "frozen"
    manifest_path = frozen / "frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("no frozen model - run `wattline reproduce` (or freeze) first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("model_file", "artifacts_file"):
        p = frozen / manifest[key]["name"]
        if sha256_file(p) != manifest[key]["sha256"]:
            raise RuntimeError(f"{p.name} does not match the frozen manifest hash; refusing to serve")
    art = json.loads((frozen / manifest["artifacts_file"]["name"]).read_text(encoding="utf-8"))
    return Artifacts(
        model=load_model(frozen / manifest["model_file"]["name"]),
        conformal=LogConformal.from_dict(art["conformal"]),
        rule=AdvisoryRule(**art["advisory_rule"]),
        profile=np.array(art["fill_profile"]),
        train_median=float(art["train_median_kwh"]),
        monitor_ref=MonitorRef(**art["monitor_ref"]),
        monitor_adopted=bool(art["monitor_adopted"]),
        primary_baseline=art["primary_baseline"],
        manifest_sha256=sha256_file(manifest_path),
        level=float(art.get("interval_level", 0.8)),
    )


# --------------------------------------------------------------------------- input contract
UNIT_HINT = re.compile(r"(wh|kw|watt|power|energy|load|consumption|demand|mwh)", re.I)


def read_meter_csv(path: Path | str) -> pd.Series:
    path = Path(path)
    if not path.exists():
        raise InputContractError(f"input file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    ts_col = next((c for c in ("timestamp_utc", "timestamp") if c in df.columns), None)
    if ts_col is None:
        raise InputContractError(f"no timestamp column (need 'timestamp_utc' or 'timestamp'); got {list(df.columns)}")
    if "energy_kwh" not in df.columns:
        hints = [c for c in df.columns if UNIT_HINT.search(c)]
        raise InputContractError(
            "unexpected unit/column: expected 'energy_kwh' (kWh per hour)"
            + (f", found {hints}" if hints else f", found {list(df.columns)}")
            + ". Convert explicitly; wattline will not guess units."
        )
    if "unit" in df.columns:
        bad = sorted(set(df["unit"].str.strip()) - {"kWh"})
        if bad:
            raise InputContractError(f"unit column contains {bad}; only 'kWh' is accepted")
    raw_ts = df[ts_col].str.strip()
    if (~raw_ts.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)).any():
        example = raw_ts[~raw_ts.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)].iloc[0]
        raise InputContractError(f"timestamp without explicit UTC offset (e.g. {example!r}); "
                                 "timezone-naive data is refused")
    try:
        ts = pd.to_datetime(raw_ts, utc=True, format="ISO8601")
    except (ValueError, TypeError) as exc:
        raise InputContractError(f"unparsable timestamps: {exc}") from exc
    vals = pd.to_numeric(df["energy_kwh"].replace("", np.nan), errors="coerce")
    nonnum = vals.isna() & df["energy_kwh"].ne("") & df["energy_kwh"].str.lower().ne("nan")
    if nonnum.any():
        raise InputContractError(f"non-numeric energy_kwh values, e.g. {df['energy_kwh'][nonnum].iloc[0]!r}")
    idx = pd.DatetimeIndex(ts).as_unit("ns")
    dup = idx[idx.duplicated()]
    if len(dup):
        raise InputContractError(
            f"{len(dup)} duplicate timestamp(s), first {dup[0].isoformat()}; "
            "refusing to choose between conflicting readings"
        )
    if ((idx.minute != 0) | (idx.second != 0)).any():
        bad = idx[(idx.minute != 0) | (idx.second != 0)][0]
        raise InputContractError(f"timestamp not on the hour: {bad.isoformat()} (hourly data required)")
    if (vals < 0).any():
        raise InputContractError(f"{int((vals < 0).sum())} negative energy values")
    return pd.Series(vals.to_numpy(dtype=float), index=idx, name="energy_kwh").sort_index()


def _longest_flat_run(x: np.ndarray) -> int:
    best = run = 1 if len(x) else 0
    for a, b in zip(x[:-1], x[1:]):
        run = run + 1 if (not np.isnan(a) and a == b) else 1
        best = max(best, run)
    return best


def assess_input(series: pd.Series, issue_utc: pd.Timestamp, cfg: Config,
                 train_median: float) -> tuple[Assessment, pd.Series | None]:
    """Check the input against the contract at a given issue time.

    Returns the assessment and the hourly history on a complete grid ending at the
    issue time (or None when UNAVAILABLE)."""
    a = Assessment()
    sv = cfg.serving
    tz = cfg.data.local_tz
    if issue_utc.tzinfo is None:
        a.downgrade(UNAVAILABLE, "issue time has no timezone")
        return a, None
    issue_utc = issue_utc.tz_convert("UTC")
    local = issue_utc.tz_convert(tz)
    a.checks["issue_time_local"] = local.isoformat()
    if issue_utc.minute or issue_utc.second or local.hour != cfg.split.issue_hour_local:
        a.downgrade(UNAVAILABLE, f"issue time {local.isoformat()} is outside the model contract "
                                 f"(forecasts are issued at {cfg.split.issue_hour_local:02d}:00 {tz})")
        return a, None

    future = series.index >= issue_utc
    a.checks["rows_after_issue_time_ignored"] = int(future.sum())
    if future.any():
        a.reasons.append(f"[info] {int(future.sum())} rows at/after the issue time were ignored "
                         "(not known at issue time)")
    hist = series[~future]
    observed = hist.dropna()
    if observed.empty:
        a.downgrade(UNAVAILABLE, "no observed history before the issue time")
        return a, None

    last_end = observed.index[-1] + pd.Timedelta(hours=1)
    stale_h = (issue_utc - last_end) / pd.Timedelta(hours=1)
    a.checks["last_observation_end"] = last_end.tz_convert(tz).isoformat()
    a.checks["staleness_hours"] = float(stale_h)
    if stale_h > sv.stale_unavailable_hours:
        a.downgrade(UNAVAILABLE, f"stale data: last reading ended {stale_h:.0f} h before the issue time "
                                 f"(limit {sv.stale_unavailable_hours} h). Stale values are not presented as a forecast")
        return a, None
    if stale_h >= sv.stale_uncertain_hours:
        a.downgrade(UNCERTAIN, f"last reading ended {stale_h:.0f} h before the issue time; recent hours imputed")

    start = issue_utc - pd.Timedelta(hours=MIN_HISTORY)
    if hist.index[0] > start:
        have = (issue_utc - hist.index[0]) / pd.Timedelta(hours=1)
        a.downgrade(UNAVAILABLE, f"insufficient history: {have:.0f} h before issue, need {MIN_HISTORY} h")
        return a, None
    grid = pd.date_range(hist.index[0], issue_utc - pd.Timedelta(hours=1), freq="h", tz="UTC")
    hist = hist.reindex(grid)
    last168, last24 = hist.iloc[-168:], hist.iloc[-24:]
    miss168, miss24 = float(last168.isna().mean()), int(last24.isna().sum())
    a.checks["missing_frac_last_168h"] = round(miss168, 4)
    a.checks["missing_hours_last_24h"] = miss24
    if miss168 > sv.missing_168_unavailable:
        a.downgrade(UNAVAILABLE, f"{miss168:.0%} of the last 168 h are missing (limit {sv.missing_168_unavailable:.0%})")
        return a, None
    if miss168 > sv.missing_168_uncertain:
        a.downgrade(UNCERTAIN, f"{miss168:.0%} of the last 168 h are missing and were imputed")
    if miss24 > 0:
        a.downgrade(UNCERTAIN, f"{miss24} of the last 24 h are missing and were imputed")

    med = float(np.nanmedian(last168.to_numpy()))
    ratio = med / train_median if train_median > 0 else float("inf")
    a.checks["median_vs_training_ratio"] = round(ratio, 3)
    if ratio > sv.unit_ratio_limit or ratio < 1 / sv.unit_ratio_limit:
        a.downgrade(UNAVAILABLE, f"recent median {med:.3g} is {ratio:.3g}x the training median "
                                 f"{train_median:.3g} kWh - suspected unit error (Wh vs kWh?)")
        return a, None

    flat = _longest_flat_run(last24.to_numpy())
    a.checks["longest_flat_run_last_24h"] = flat
    if flat >= sv.flatline_hours:
        a.downgrade(UNCERTAIN, f"{flat} identical consecutive readings in the last 24 h - possible stuck meter")
    return a, hist


# --------------------------------------------------------------------------- forecast
def recent_monitor(art: Artifacts, cfg: Config, filled: np.ndarray, imputed: np.ndarray,
                   observed: np.ndarray, grid: pd.DatetimeIndex, issue_utc: pd.Timestamp) -> MonitorState:
    """Re-score the frozen model on the previous issue times (actuals now known)."""
    tz = cfg.data.local_tz
    base = issue_utc.tz_convert(tz).tz_localize(None).normalize()   # wall-clock date
    ys, ps, bs = [], [], []
    for k in range(1, cfg.monitor.window_days + 1):
        wall = base - pd.Timedelta(days=k) + pd.Timedelta(hours=cfg.split.issue_hour_local)
        o = wall.tz_localize(tz).tz_convert("UTC")
        p = grid.get_indexer([o])[0]
        if p < MIN_HISTORY:
            continue
        x = origin_features(filled[:p], imputed[:p], o, tz)[None]
        pred = art.model.predict(x)[0]
        act = np.full(H, np.nan)
        seg = observed[p : p + H]
        act[: len(seg)] = seg           # history ends at the issue time: no future actuals
        ys.append(act)
        ps.append(pred)
        bs.append(baseline_forecast(art.primary_baseline, filled[:p]))
    if not ys:
        return MonitorState(False, 0, float("nan"), float("nan"), float("nan"), 1.0, False,
                            "monitor inactive: no history")
    m = cfg.monitor
    return monitor_state(np.array(ys), np.array(ps), art.monitor_ref, m.min_scored_hours,
                         m.alarm_mae_ratio, m.max_widen, np.array(bs), m.skill_alarm_ratio)


def make_forecast(cfg: Config, art: Artifacts, series: pd.Series, issue_utc: pd.Timestamp) -> dict:
    tz = cfg.data.local_tz
    issue_utc = pd.Timestamp(issue_utc)
    assessment, hist = assess_input(series, issue_utc, cfg, art.train_median)
    report: dict = {
        "issue_time_local": issue_utc.tz_convert(tz).isoformat() if issue_utc.tzinfo else str(issue_utc),
        "issue_time_utc": issue_utc.tz_convert("UTC").isoformat() if issue_utc.tzinfo else str(issue_utc),
        "data_cutoff": "all inputs end at the issue time (last hour used: [issue-1h, issue))",
        "status": assessment.status,
        "reasons": assessment.reasons,
        "checks": assessment.checks,
        "model": art.model.spec.name,
        "frozen_manifest_sha256": art.manifest_sha256,
        "hours": [],
    }
    if assessment.status == UNAVAILABLE:
        return report
    issue_utc = issue_utc.tz_convert("UTC")
    observed = hist.to_numpy(dtype=float)
    fill = causal_fill(observed, hist.index, art.profile, tz)
    x = origin_features(fill.filled, fill.imputed, issue_utc, tz)[None]
    pred = art.model.predict(x)[0]

    mon = recent_monitor(art, cfg, fill.filled, fill.imputed, observed, hist.index, issue_utc)
    report["monitor"] = mon.to_dict() | {"adopted": art.monitor_adopted}
    scale = 1.0
    if art.monitor_adopted:
        scale = mon.scale
        if mon.alarm:
            assessment.downgrade(UNCERTAIN, mon.note)
    elif mon.active:
        assessment.reasons.append(f"[info] monitor (diagnostic only): {mon.note}")
    report["status"] = assessment.status

    lo, hi = art.conformal.bounds(pred, art.level, scale)
    flag_q = art.conformal.quantile(pred, art.rule.flag_level, scale)
    labels = hour_labels(flag_q, hi, art.rule, assessment.status)
    k = cfg.advisory.top_k_peaks
    peak_idx = set(np.argsort(-pred)[:k].tolist())
    tt = target_times(issue_utc)
    for h in range(H):
        report["hours"].append({
            "horizon": h + 1,
            "hour_start_local": tt[h].tz_convert(tz).isoformat(),
            "hour_start_utc": tt[h].isoformat(),
            "forecast_kwh": round(float(pred[h]), 4),
            "lower80_kwh": round(float(lo[h]), 4),
            "upper80_kwh": round(float(hi[h]), 4),
            "advisory": labels[h],
            "likely_peak": h in peak_idx,
        })
    report["peak_hours_local"] = [tt[i].tz_convert(tz).strftime("%H:%M") for i in sorted(peak_idx, key=lambda i: -pred[i])]
    report["thresholds"] = {
        "high_demand_kwh": round(art.rule.high_kwh, 4),
        "high_demand_definition": f"{art.rule.high_quantile:.0%} quantile of observed train+validation hourly energy",
        "flag_rule": f"HIGH_DEMAND when the predicted {art.rule.flag_level:.0%} quantile >= threshold",
        "low_kwh": round(art.rule.low_kwh, 4),
        "low_definition": f"LOW when the 80% upper bound < {art.rule.low_quantile:.0%} quantile",
        "interval_stretch": round(float(scale), 3),
    }
    report["feature_provenance"] = provenance(issue_utc, tz)
    report["imputed_input_hours_last_168h"] = int(fill.imputed[-168:].sum())
    return report
