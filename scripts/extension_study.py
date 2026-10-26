"""Pre test extension studies on the walk forward backtest folds (no test values loaded).

Three candidate improvements are evaluated here on the 21 pre validation folds only, using
the same models, calibration windows and monitor as `wattline backtest`:

* A  per hour block interval calibration (fixes uneven coverage across the day)
* D1 hysteresis for the skill and drift alarm (fixes flicker)
* D2 operator impact simulation (peak staggering of a flexible load, advisory only)

Nothing here changes a frozen decision or the served tool. The block layout, the
hysteresis levels and the simulation assumptions are fixed in docs/EXTENSIONS.md before the
results are read. Outputs: results/extensions/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wattline.advisory import HIGH, AdvisoryRule, false_confidence, label_matrix, thresholds
from wattline.config import load_config
from wattline.evaluate import coverage, median_width
from wattline.features import H
from wattline.intervals import LogConformal
from wattline.model import ModelSpec, fit_model
from wattline.monitor import MonitorRef, batch_monitor
from wattline.pipeline import prepare
from wattline.util import read_json, write_csv, write_json

# Pre registered choices (docs/EXTENSIONS.md)
BLOCKS = [(0, 5), (6, 9), (10, 16), (17, 23)]          # local target hour, inclusive
ALARM_ON_RATIO, ALARM_OFF_RATIO = 1.5, 1.3             # recent MAE / validation MAE
SKILL_ON, SKILL_OFF = 1.0, 0.85                        # recent MAE / recent naive MAE
FLEX_KW, FLEX_HOURS, DEFAULT_START = 1.0, 2, 19        # flexible load assumption
START_MIN, START_MAX = 10, 21                          # allowed start hours (local)
ALPHA = 0.2


def winkler(y, lo, hi) -> np.ndarray:
    return (hi - lo) + (2 / ALPHA) * np.clip(lo - y, 0, None) + (2 / ALPHA) * np.clip(y - hi, 0, None)


def hysteresis(ratio: np.ndarray, skill: np.ndarray) -> np.ndarray:
    """Alarm switches on above the original limits and off only below lower limits."""
    on, out = False, np.zeros(len(ratio), bool)
    for i, (r, s) in enumerate(zip(ratio, skill)):
        r_hi, s_hi = (r > ALARM_ON_RATIO), (s > SKILL_ON)
        r_lo, s_lo = (not r > ALARM_OFF_RATIO), (not s > SKILL_OFF)
        if not on and (r_hi or s_hi):
            on = True
        elif on and r_lo and s_lo:
            on = False
        out[i] = on
    return out


def stagger_day(actual, forecast, start_default=DEFAULT_START):
    """Daily peak (kWh) with the flexible load at its default start, at the start chosen
    from ``forecast``, and at the start chosen with hindsight (oracle)."""
    def peak(start):
        add = np.zeros(24)
        add[start:start + FLEX_HOURS] = FLEX_KW
        return float(np.max(actual + add))

    def choose(f):
        best, best_val = start_default, np.inf
        for s in range(START_MIN, START_MAX + 1):
            add = np.zeros(24)
            add[s:s + FLEX_HOURS] = FLEX_KW
            v = float(np.max(f + add))
            if v < best_val - 1e-12:
                best, best_val = s, v
        return best

    return peak(start_default), peak(choose(forecast)), peak(choose(actual))


def stagger_summary(days: list[tuple[float, float, float]], high_kwh: float) -> dict:
    d = np.array(days)
    default, policy, oracle = d[:, 0], d[:, 1], d[:, 2]
    gain, best = default - policy, default - oracle
    return {
        "days": int(len(d)),
        "mean_daily_peak_default": float(default.mean()),
        "mean_daily_peak_policy": float(policy.mean()),
        "mean_daily_peak_oracle": float(oracle.mean()),
        "mean_reduction_kwh": float(gain.mean()),
        "share_of_oracle_gain_captured": float(gain.sum() / best.sum()) if best.sum() > 0 else float("nan"),
        "days_improved": int((gain > 1e-9).sum()),
        "days_worse": int((gain < -1e-9).sum()),
        "days_above_high_default": int((default >= high_kwh).sum()),
        "days_above_high_policy": int((policy >= high_kwh).sum()),
    }


def main() -> None:
    cfg = load_config()
    sel = read_json(cfg.results_dir / "validation" / "selection.json")
    chosen = ModelSpec(**sel["chosen_model"])
    primary = sel["primary_baseline"]
    flag_level = float(sel["advisory_rule"]["flag_level"])
    prep = prepare(cfg, upto=cfg.val_end_utc)           # structural test blindness
    tz, level, mc = cfg.data.local_tz, cfg.interval.level, cfg.monitor
    hgb = {"max_iter": cfg.model.hgb_max_iter, "learning_rate": cfg.model.hgb_learning_rate,
           "max_leaf_nodes": cfg.model.hgb_max_leaf_nodes}
    walls = pd.date_range(cfg.backtest.first_fold, cfg.backtest.last_fold_end, freq="MS")
    end = prep.origins + pd.Timedelta(hours=H)
    variants = ["static_pooled", "static_block", "adaptive_pooled", "adaptive_block"]
    pool = {v: {"y": [], "lo": [], "hi": [], "hour": [], "fold": []} for v in variants}
    hyst = {"orig": {"toggles": 0, "alarm_days": 0, "fc": [], "n": 0},
            "hyst": {"toggles": 0, "alarm_days": 0, "fc": [], "n": 0}}
    stag_model, stag_base = [], []
    fold_rows = []

    for w0, w1 in zip(walls[:-1], walls[1:]):
        F0 = w0.tz_localize(tz).tz_convert("UTC")
        F1 = w1.tz_localize(tz).tz_convert("UTC")
        C0 = (w0 - pd.Timedelta(days=cfg.backtest.calibration_days)).tz_localize(tz).tz_convert("UTC")
        train_rows = prep.eligible & (end <= C0) & (prep.split == "train")
        cal_rows = prep.eligible & (prep.origins >= C0) & (end <= F0)
        ev_rows = (prep.origins >= F0) & (end <= F1)
        M = prep.mask(ev_rows)
        if M.sum() < 24 or cal_rows.sum() < 14:
            continue
        model = fit_model(chosen, prep.X, prep.Y, train_rows, cfg.model.random_seed, hgb)
        P = model.predict(prep.X)
        conf = LogConformal.fit(prep.Y[cal_rows], P[cal_rows], extra_levels=[flag_level])
        ref = MonitorRef.fit(prep.Y[cal_rows], P[cal_rows])
        states = batch_monitor(prep.Y, P, prep.target_ns, prep.origin_ns, ref, mc.window_days,
                               mc.min_scored_hours, mc.alarm_mae_ratio, mc.max_widen, rows=ev_rows,
                               B=prep.base[primary], skill_ratio_limit=mc.skill_alarm_ratio)
        scale = np.ones(len(P))
        scale[ev_rows] = [s.scale for s in states]
        sc = scale[:, None]

        lo_sp, hi_sp = conf.bounds(P, level)
        lo_ap, hi_ap = conf.bounds(P, level, sc)
        lo_sb, hi_sb, lo_ab, hi_ab = (np.zeros_like(P) for _ in range(4))
        hl = prep.hour_local
        for a, b in BLOCKS:
            cal_mask = ((hl >= a) & (hl <= b)) & cal_rows[:, None] & ~np.isnan(prep.Y)
            cb = LogConformal.fit(prep.Y[cal_mask], P[cal_mask])
            m_all = (hl >= a) & (hl <= b)
            l1, h1 = cb.bounds(P, level)
            l2, h2 = cb.bounds(P, level, sc)
            lo_sb[m_all], hi_sb[m_all] = l1[m_all], h1[m_all]
            lo_ab[m_all], hi_ab[m_all] = l2[m_all], h2[m_all]
        got = {"static_pooled": (lo_sp, hi_sp), "static_block": (lo_sb, hi_sb),
               "adaptive_pooled": (lo_ap, hi_ap), "adaptive_block": (lo_ab, hi_ab)}
        row = {"fold": w0.strftime("%Y-%m"), "scored_hours": int(M.sum())}
        for v, (lo, hi) in got.items():
            pool[v]["y"].append(prep.Y[M]); pool[v]["lo"].append(lo[M]); pool[v]["hi"].append(hi[M])
            pool[v]["hour"].append(hl[M]); pool[v]["fold"].append(np.full(M.sum(), len(fold_rows)))
            row[f"cov_{v}"] = coverage(prep.Y, lo, hi, M)
            row[f"width_{v}"] = median_width(lo, hi, M)
        fold_rows.append(row)

        # D1 hysteresis on the alarm sequence of this fold
        ratio = np.array([s.mae_ratio for s in states])
        skill = np.array([s.skill_ratio for s in states])
        orig = np.array([s.alarm for s in states])
        hys = hysteresis(np.nan_to_num(ratio, nan=0.0), np.nan_to_num(skill, nan=0.0))
        high, low = thresholds(prep.observed[prep.index < F0], cfg.advisory.high_quantile,
                               cfg.advisory.low_quantile)
        rule = AdvisoryRule(high, low, flag_level, cfg.advisory.high_quantile, cfg.advisory.low_quantile)
        idx = np.flatnonzero(ev_rows)
        for name, al in (("orig", orig), ("hyst", hys)):
            full = np.zeros(len(P), bool)
            full[idx] = al
            lab = label_matrix(conf.quantile(P, flag_level, sc), hi_ap, rule, full)
            fc = false_confidence(prep.Y, lab, hi_ap, high, M)
            hyst[name]["toggles"] += int(np.abs(np.diff(al.astype(int))).sum())
            hyst[name]["alarm_days"] += int(al.sum())
            hyst[name]["fc"].append((fc["rate_per_scored_hour"], int(M.sum())))
            hyst[name]["n"] += len(al)

        # D2 staggering simulation: full days with 24 observed local hours 0..23
        for i in idx:
            if not prep.eligible[i] or np.isnan(prep.Y[i]).any():
                continue
            if not (hl[i] == np.arange(24)).all():
                continue
            stag_model.append(stagger_day(prep.Y[i], P[i]))
            stag_base.append(stagger_day(prep.Y[i], prep.base[primary][i]))

    out = cfg.results_dir / "extensions"
    out.mkdir(parents=True, exist_ok=True)
    fdf = pd.DataFrame(fold_rows)
    write_csv(fdf, out / "interval_blocks_folds.csv")

    summary = {"folds": len(fdf), "variants": {}}
    for v in variants:
        y, lo, hi = (np.concatenate(pool[v][k]) for k in ("y", "lo", "hi"))
        hour = np.concatenate(pool[v]["hour"])
        inside = (y >= lo) & (y <= hi)
        per_hour = np.array([inside[hour == h].mean() for h in range(24)])
        per_block = {f"{a:02d}-{b:02d}": float(inside[(hour >= a) & (hour <= b)].mean()) for a, b in BLOCKS}
        fold_cov = fdf[f"cov_{v}"].to_numpy()
        summary["variants"][v] = {
            "coverage": float(inside.mean()), "median_width": float(np.median(hi - lo)),
            "mean_interval_score": float(winkler(y, lo, hi).mean()),
            "coverage_by_block": per_block,
            "worst_block_gap": float(max(abs(c - level) for c in per_block.values())),
            "mean_abs_hour_gap": float(np.mean(np.abs(per_hour - level))),
            "min_hour_coverage": float(per_hour.min()), "max_hour_coverage": float(per_hour.max()),
            "folds_below_70": int((fold_cov < 0.70).sum()), "min_fold_coverage": float(fold_cov.min()),
        }

    def fc_rate(h):
        r = np.array([a for a, _ in h["fc"]]); w = np.array([b for _, b in h["fc"]])
        return float(np.average(r, weights=w))
    summary["alarm"] = {k: {"alarm_days": v["alarm_days"], "origins": v["n"], "toggles": v["toggles"],
                            "false_confidence": fc_rate(v)} for k, v in hyst.items()}
    high_all, _ = thresholds(prep.observed, cfg.advisory.high_quantile, cfg.advisory.low_quantile)
    summary["staggering_backtest"] = {
        "model_policy": stagger_summary(stag_model, high_all),
        "naive_policy": stagger_summary(stag_base, high_all)}
    write_json(summary, out / "extension_backtest.json")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
