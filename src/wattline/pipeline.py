"""Pipeline stages: prepare -> select -> backtest -> freeze -> final-eval.

Test blindness is structural: ``select``, ``backtest`` and ``freeze`` call
``prepare(cfg, upto=val_end)``, which truncates the hourly series *before* anything else
runs, so no test-period value exists in memory during model, interval, threshold or
monitor decisions. Only ``final_eval`` loads the full series, and it refuses to run if
the frozen decisions differ from those recorded the first time the test was scored.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import __version__
from .advisory import (HIGH, LOW, NORMAL, UNCERTAIN, AdvisoryRule, false_confidence,
                       label_matrix, thresholds)
from .baselines import ALL_BASELINES, NAIVE_BASELINES, baseline_matrix
from .calendar_fr import day_type
from .clean import FillResult, causal_fill, fill_log, fit_profile
from .config import Config, sha256_file
from .evaluate import (coverage, flag_scores, improvement, mae, median_width, peak_hit_rate,
                       per_group, per_horizon, per_origin_mae, rmse)
from .features import FEATURE_SETS, H, MIN_HISTORY, build_design, provenance, targets
from .ingest import load_hourly
from .intervals import LogConformal
from .model import ModelSpec, fit_model, save_model
from .monitor import HOUR_NS, MonitorRef, batch_monitor
from .util import md_table, read_json, write_csv, write_json


# =========================================================================== prepare
@dataclass
class Prepared:
    index: pd.DatetimeIndex
    observed: np.ndarray
    fill: FillResult
    profile: np.ndarray
    origins: pd.DatetimeIndex
    pos: np.ndarray
    split: np.ndarray
    eligible: np.ndarray
    miss168: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    target_ns: np.ndarray
    hour_local: np.ndarray
    day_type: np.ndarray
    base: dict

    @property
    def origin_ns(self) -> np.ndarray:
        return self.origins.as_unit("ns").asi8

    def mask(self, rows: np.ndarray) -> np.ndarray:
        """Scoring mask: chosen rows x eligible origins x observed target hours."""
        return (rows & self.eligible)[:, None] & ~np.isnan(self.Y)


def split_of_origins(origins: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    end = origins + pd.Timedelta(hours=H)
    split = np.full(len(origins), "none", dtype=object)
    split[end <= cfg.train_end_utc] = "train"
    split[(origins >= cfg.train_end_utc) & (end <= cfg.val_end_utc)] = "validation"
    split[(origins >= cfg.val_end_utc) & (end <= cfg.test_end_utc)] = "test"
    return split


def daily_origins(index: pd.DatetimeIndex, cfg: Config) -> pd.DatetimeIndex:
    tz = cfg.data.local_tz
    first = index[0].tz_convert(tz).tz_localize(None).normalize()
    last = index[-1].tz_convert(tz).tz_localize(None).normalize()
    walls = pd.date_range(first, last, freq="D") + pd.Timedelta(hours=cfg.split.issue_hour_local)
    return walls.tz_localize(tz).tz_convert("UTC").as_unit("ns")


def prepare(cfg: Config, upto: pd.Timestamp | None = None) -> Prepared:
    series = load_hourly(cfg)
    if upto is not None:
        series = series[series.index < upto]       # structural test blindness
    index, observed = series.index, series.to_numpy(dtype=float)
    tz = cfg.data.local_tz
    profile = fit_profile(observed, index, cfg.train_end_utc, tz)
    fill = causal_fill(observed, index, profile, tz)

    origins = daily_origins(index, cfg)
    pos = index.get_indexer(origins)
    keep = (pos >= MIN_HISTORY) & (pos + H <= len(index))
    origins, pos = origins[keep], pos[keep]
    split = split_of_origins(origins, cfg)
    miss168 = np.array([np.isnan(observed[p - 168 : p]).mean() for p in pos])
    all24 = np.array([np.isnan(observed[p - 24 : p]).all() for p in pos])
    eligible = (miss168 <= cfg.max_missing_frac_168) & ~all24

    X = build_design(fill.filled, fill.imputed, index, pos, tz)
    Y = targets(observed, pos)
    target_ns = origins.asi8[:, None] + np.arange(H)[None, :] * HOUR_NS
    tt = pd.to_datetime(target_ns.ravel(), utc=True).tz_convert(tz)
    hour_local = tt.hour.to_numpy().reshape(-1, H)
    dtype = np.array([day_type(t) for t in tt], dtype=object).reshape(-1, H)
    base = {b: baseline_matrix(b, fill.filled, pos) for b in ALL_BASELINES}
    return Prepared(index, observed, fill, profile, origins, pos, split, eligible, miss168,
                    X, Y, target_ns, hour_local, dtype, base)


def origin_log(prep: Prepared) -> dict:
    out = {}
    for name in ["train", "validation", "test", "none"]:
        rows = prep.split == name
        if not rows.any():
            continue
        out[name] = {
            "origins": int(rows.sum()),
            "eligible_origins": int((rows & prep.eligible).sum()),
            "ineligible_origins": [str(o.tz_convert("Europe/Paris").date())
                                   for o in prep.origins[rows & ~prep.eligible]],
            "scored_hours": int(prep.mask(rows).sum()),
            "missing_target_hours_in_eligible_origins": int(
                np.isnan(prep.Y[rows & prep.eligible]).sum()),
            "first_origin_local": prep.origins[rows][0].tz_convert("Europe/Paris").isoformat(),
            "last_origin_local": prep.origins[rows][-1].tz_convert("Europe/Paris").isoformat(),
        }
    return out


# =========================================================================== select
def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_select(cfg: Config) -> dict:
    prep = prepare(cfg, upto=cfg.val_end_utc)
    out = cfg.results_dir / "validation"
    tr = (prep.split == "train") & prep.eligible
    va = prep.split == "validation"
    Mva = prep.mask(va)
    Y = prep.Y

    # ---- baselines: primary chosen on validation only --------------------------------
    brow = []
    for b in ALL_BASELINES:
        brow.append({"baseline": b, "eligible_as_primary": b in NAIVE_BASELINES,
                     "val_mae": mae(Y, prep.base[b], Mva), "val_rmse": rmse(Y, prep.base[b], Mva)})
    bdf = pd.DataFrame(brow)
    primary = bdf[bdf.eligible_as_primary].sort_values("val_mae").iloc[0]["baseline"]

    # ---- ridge grid + feature ablation -----------------------------------------------
    grid, timings, preds = [], {}, {}
    for fs in cfg.model.feature_sets:
        for tgt in cfg.model.targets:
            for a in cfg.model.alphas:
                spec = ModelSpec("ridge", fs, tgt, float(a))
                m = fit_model(spec, prep.X, Y, tr)
                P = m.predict(prep.X)
                preds[spec] = P
                timings[spec.name] = round(m.fit_seconds, 4)
                grid.append({"model": spec.name, "feature_set": fs, "target": tgt, "alpha": a,
                             "val_mae": mae(Y, P, Mva), "val_rmse": rmse(Y, P, Mva)})
    gdf = pd.DataFrame(grid)
    best_row = gdf.sort_values(["val_mae", "alpha"], ascending=[True, False]).iloc[0]
    best_spec = ModelSpec("ridge", best_row.feature_set, best_row.target, float(best_row.alpha))

    ablation = []
    for fs in cfg.model.feature_sets:
        r = gdf[gdf.feature_set == fs].sort_values("val_mae").iloc[0]
        ablation.append({"feature_set": fs, "n_features": len(FEATURE_SETS[fs]),
                         "spec": ModelSpec("ridge", fs, r.target, float(r.alpha)).to_dict(),
                         "val_mae": r.val_mae, "val_rmse": r.val_rmse})

    # ---- optional boosted trees: must earn >= 3 % --------------------------------------
    hspec = ModelSpec("hgb", best_spec.feature_set, "raw", 0.0)
    hm = fit_model(hspec, prep.X, Y, tr, seed=cfg.model.random_seed,
                   hgb_params={"max_iter": cfg.model.hgb_max_iter,
                               "learning_rate": cfg.model.hgb_learning_rate,
                               "max_leaf_nodes": cfg.model.hgb_max_leaf_nodes})
    Ph = hm.predict(prep.X)
    timings[hspec.name] = round(hm.fit_seconds, 4)
    hgb_mae = mae(Y, Ph, Mva)
    gain = improvement(float(best_row.val_mae), hgb_mae)
    adopt_hgb = gain >= cfg.model.hgb_min_relative_gain
    chosen = hspec if adopt_hgb else best_spec
    P = Ph if adopt_hgb else preds[best_spec]
    chosen_val_mae = mae(Y, P, Mva)

    # ---- intervals from validation residuals -----------------------------------------
    level = cfg.interval.level
    rows_va = va & prep.eligible
    conf = LogConformal.fit(Y[rows_va], P[rows_va], extra_levels=cfg.interval.flag_levels)
    idx = np.flatnonzero(rows_va)
    halves = [idx[: len(idx) // 2], idx[len(idx) // 2 :]]
    crossfit = []
    for cal, ev in [(halves[0], halves[1]), (halves[1], halves[0])]:
        c = LogConformal.fit(Y[cal], P[cal])
        lo, hi = c.bounds(P[ev], level)
        crossfit.append({"coverage": coverage(Y[ev], lo, hi), "median_width": median_width(lo, hi)})
    lo, hi = conf.bounds(P, level)

    # ---- advisory thresholds (train + validation observed hours only) -----------------
    high, low = thresholds(prep.observed, cfg.advisory.high_quantile, cfg.advisory.low_quantile)
    actual_high = np.where(np.isnan(Y), False, Y >= high)
    flag_rows = []
    for lvl in cfg.interval.flag_levels:
        s = flag_scores(actual_high, conf.quantile(P, lvl) >= high, Mva)
        flag_rows.append({"flag_level": lvl, **s})
    fdf = pd.DataFrame(flag_rows)
    best_flag = fdf.sort_values(["f1", "recall"], ascending=False).iloc[0]
    flag_level = float(best_flag.flag_level)
    base_flag = flag_scores(actual_high, prep.base[primary] >= high, Mva)
    rule = AdvisoryRule(high, low, flag_level, cfg.advisory.high_quantile, cfg.advisory.low_quantile)
    mref = MonitorRef.fit(Y[rows_va], P[rows_va])

    selection = {
        "stage": "select (validation only; test values not loaded)",
        "origins": origin_log(prep),
        "missing_policy": fill_log(prep.fill, _hour_labels(prep, cfg)),
        "baselines": brow,
        "primary_baseline": primary,
        "ridge_grid_best": best_spec.to_dict(),
        "ridge_best_val_mae": float(best_row.val_mae),
        "hgb": {"spec": hspec.to_dict(), "val_mae": hgb_mae, "relative_gain_vs_best_ridge": gain,
                "required_gain": cfg.model.hgb_min_relative_gain, "adopted": bool(adopt_hgb)},
        "ablation": ablation,
        "chosen_model": chosen.to_dict(),
        "chosen_model_name": chosen.name,
        "chosen_val_mae": chosen_val_mae,
        "chosen_val_rmse": rmse(Y, P, Mva),
        "val_improvement_vs_primary": improvement(float(bdf.set_index("baseline").loc[primary, "val_mae"]),
                                                  chosen_val_mae),
        "conformal": conf.to_dict(),
        "interval_validation": {
            "in_sample_coverage": coverage(Y, lo, hi, Mva),
            "in_sample_median_width": median_width(lo, hi, Mva),
            "crossfit_halves": crossfit,
            "note": "in-sample coverage is ~80 % by construction; the cross-fit halves are the honest pre-test estimate",
        },
        "advisory_rule": rule.to_dict(),
        "flag_study_validation": flag_rows,
        "primary_baseline_flag_validation": base_flag,
        "monitor_ref": mref.to_dict(),
    }
    write_json(selection, out / "selection.json")
    write_json(timings, out / "timings_seconds.json")
    write_csv(bdf, out / "baselines.csv")
    write_csv(gdf, out / "ridge_grid.csv")
    write_csv(fdf, out / "flag_study.csv")
    _write_selection_md(selection, bdf, gdf, fdf, out)
    return selection


def _hour_labels(prep: Prepared, cfg: Config) -> np.ndarray:
    lab = np.full(len(prep.index), "after_test", dtype=object)
    lab[prep.index < cfg.test_end_utc] = "test"
    lab[prep.index < cfg.val_end_utc] = "validation"
    lab[prep.index < cfg.train_end_utc] = "train"
    return lab


def _write_selection_md(sel, bdf, gdf, fdf, out) -> None:
    abl = pd.DataFrame([{k: v for k, v in r.items() if k != "spec"} | {"best_variant": ModelSpec(**r["spec"]).name}
                        for r in sel["ablation"]])
    lines = [
        "# Validation-stage selection (generated by `wattline select`)", "",
        "Test-period values were not loaded when this file was produced.", "",
        "## Baselines (validation)", md_table(bdf, ".4f"), "",
        f"**Primary baseline (locked): `{sel['primary_baseline']}`**", "",
        "## Feature-set ablation (best variant per set, validation)", md_table(abl, ".4f"), "",
        "## Boosted-tree contender",
        f"HGB val MAE {sel['hgb']['val_mae']:.4f} vs best ridge {sel['ridge_best_val_mae']:.4f} "
        f"-> relative gain {sel['hgb']['relative_gain_vs_best_ridge']:+.2%} "
        f"(needs >= {sel['hgb']['required_gain']:.0%}); adopted: {sel['hgb']['adopted']}", "",
        f"**Chosen model: `{sel['chosen_model_name']}`**, val MAE {sel['chosen_val_mae']:.4f} kWh, "
        f"improvement vs primary {sel['val_improvement_vs_primary']:+.2%}", "",
        "## Interval (validation)",
        f"in-sample coverage {sel['interval_validation']['in_sample_coverage']:.3f}; cross-fit halves: "
        + ", ".join(f"{c['coverage']:.3f} (width {c['median_width']:.3f})" for c in sel["interval_validation"]["crossfit_halves"]),
        "", "## HIGH_DEMAND flag study (validation)", md_table(fdf, ".3f"), "",
        f"Chosen flag level: {sel['advisory_rule']['flag_level']}; HIGH threshold "
        f"{sel['advisory_rule']['high_kwh']:.3f} kWh; LOW threshold {sel['advisory_rule']['low_kwh']:.3f} kWh", "",
        "## Ridge grid (all variants)", md_table(gdf, ".4f"), "",
    ]
    (out / "selection.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


# =========================================================================== backtest
def _ks(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a[~np.isnan(a)]), np.sort(b[~np.isnan(b)])
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a)
                               - np.searchsorted(b, grid, "right") / len(b))))


def _psi(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    ref, new = ref[~np.isnan(ref)], new[~np.isnan(new)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    # additive (0.5-count) smoothing: an empty bin is real shift but must not dominate via log(1e-6)
    p = (np.histogram(ref, edges)[0] + 0.5) / (len(ref) + 0.5 * (len(edges) - 1))
    q = (np.histogram(new, edges)[0] + 0.5) / (len(new) + 0.5 * (len(edges) - 1))
    return float(np.sum((q - p) * np.log(q / p)))


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def run_backtest(cfg: Config) -> dict:
    sel = read_json(cfg.results_dir / "validation" / "selection.json")
    chosen = ModelSpec(**sel["chosen_model"])
    primary = sel["primary_baseline"]
    flag_level = float(sel["advisory_rule"]["flag_level"])
    abl_specs = [ModelSpec(**r["spec"]) for r in sel["ablation"]]
    prep = prepare(cfg, upto=cfg.val_end_utc)
    tz, level, mc = cfg.data.local_tz, cfg.interval.level, cfg.monitor
    hgb_params = {"max_iter": cfg.model.hgb_max_iter, "learning_rate": cfg.model.hgb_learning_rate,
                  "max_leaf_nodes": cfg.model.hgb_max_leaf_nodes}
    walls = pd.date_range(cfg.backtest.first_fold, cfg.backtest.last_fold_end, freq="MS")
    end = prep.origins + pd.Timedelta(hours=H)
    folds, pooled = [], {k: [] for k in ["y", "p", "base", "mean7", "lo", "hi", "lo_a", "hi_a",
                                          "lab_s", "lab_a", "high", "flag"]}
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
        model = fit_model(chosen, prep.X, prep.Y, train_rows, cfg.model.random_seed, hgb_params)
        P = model.predict(prep.X)
        conf = LogConformal.fit(prep.Y[cal_rows], P[cal_rows], extra_levels=[flag_level])
        ref = MonitorRef.fit(prep.Y[cal_rows], P[cal_rows])
        high, low = thresholds(prep.observed[prep.index < F0], cfg.advisory.high_quantile,
                               cfg.advisory.low_quantile)
        rule = AdvisoryRule(high, low, flag_level, cfg.advisory.high_quantile, cfg.advisory.low_quantile)
        states = batch_monitor(prep.Y, P, prep.target_ns, prep.origin_ns, ref, mc.window_days,
                               mc.min_scored_hours, mc.alarm_mae_ratio, mc.max_widen, rows=ev_rows,
                               B=prep.base[primary], skill_ratio_limit=mc.skill_alarm_ratio)
        scale, alarm = np.ones(len(P)), np.zeros(len(P), bool)
        scale[ev_rows] = [s.scale for s in states]
        alarm[ev_rows] = [s.alarm for s in states]
        lo, hi = conf.bounds(P, level)
        lo_a, hi_a = conf.bounds(P, level, scale[:, None])
        lab_s = label_matrix(conf.quantile(P, flag_level), hi, rule, np.zeros(len(P), bool))
        lab_a = label_matrix(conf.quantile(P, flag_level, scale[:, None]), hi_a, rule, alarm)
        fc_s = false_confidence(prep.Y, lab_s, hi, high, M)
        fc_a = false_confidence(prep.Y, lab_a, hi_a, high, M)
        actual_high = np.where(np.isnan(prep.Y), False, prep.Y >= high)
        fl = flag_scores(actual_high, lab_s == HIGH, M)

        abl = {}
        for spec in abl_specs:
            mm = fit_model(spec, prep.X, prep.Y, train_rows)
            abl[spec.feature_set] = mae(prep.Y, mm.predict(prep.X), M)

        ref_vals = prep.observed[prep.index < C0]
        new_vals = prep.observed[(prep.index >= F0) & (prep.index < F1)]
        base_maes = {b: mae(prep.Y, prep.base[b], M) for b in ALL_BASELINES}
        row = {
            "fold": w0.strftime("%Y-%m"), "season": SEASON[w0.month],
            "train_origins": int(train_rows.sum()), "eval_origins": int((ev_rows & prep.eligible).sum()),
            "scored_hours": int(M.sum()),
            "model_mae": mae(prep.Y, P, M), "model_rmse": rmse(prep.Y, P, M),
            **{f"{b}_mae": v for b, v in base_maes.items()},
            "improvement_vs_primary": improvement(base_maes[primary], mae(prep.Y, P, M)),
            "improvement_vs_mean7": improvement(base_maes["mean_7d_same_hour"], mae(prep.Y, P, M)),
            **{f"ablation_{k}_mae": v for k, v in abl.items()},
            "coverage_static": coverage(prep.Y, lo, hi, M), "width_static": median_width(lo, hi, M),
            "coverage_adaptive": coverage(prep.Y, lo_a, hi_a, M), "width_adaptive": median_width(lo_a, hi_a, M),
            "alarm_origins": int(alarm[ev_rows].sum()),
            "skill_alarm_origins": int(sum(s.skill_ratio > mc.skill_alarm_ratio for s in states)),
            "false_conf_static": fc_s["rate_per_scored_hour"], "false_conf_adaptive": fc_a["rate_per_scored_hour"],
            "flag_precision": fl["precision"], "flag_recall": fl["recall"], "flag_f1": fl["f1"],
            "peak_top3_hit": peak_hit_rate(prep.Y, P, M, cfg.advisory.top_k_peaks),
            "mean_kwh_train_window": float(np.nanmean(ref_vals)), "mean_kwh_fold": float(np.nanmean(new_vals)),
            "mean_ratio": float(np.nanmean(new_vals) / np.nanmean(ref_vals)),
            "ks_vs_train": _ks(ref_vals, new_vals), "psi_vs_train": _psi(ref_vals, new_vals),
        }
        folds.append(row)
        for k, v in [("y", prep.Y), ("p", P), ("base", prep.base[primary]),
                     ("mean7", prep.base["mean_7d_same_hour"]), ("lo", lo), ("hi", hi),
                     ("lo_a", lo_a), ("hi_a", hi_a), ("lab_s", lab_s), ("lab_a", lab_a),
                     ("flag", lab_s == HIGH)]:
            pooled[k].append(v[M])
        pooled["high"].append(actual_high[M])

    fdf = pd.DataFrame(folds)
    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    yv = cat["y"]
    one = np.ones_like(yv, dtype=bool)
    width_s = float(np.median(cat["hi"] - cat["lo"]))
    width_a = float(np.median(cat["hi_a"] - cat["lo_a"]))
    fc_static = float(np.mean(np.isin(cat["lab_s"], [NORMAL, LOW]) & ((yv > cat["hi"]) | cat["high"])))
    fc_adapt = float(np.mean(np.isin(cat["lab_a"], [NORMAL, LOW]) & ((yv > cat["hi_a"]) | cat["high"])))
    adopt = bool(fc_adapt < fc_static and width_a / width_s - 1 <= mc.adopt_max_width_increase)
    pooled_summary = {
        "folds": len(fdf), "scored_hours": int(len(yv)),
        "model_mae": float(np.mean(np.abs(yv - cat["p"]))),
        "primary_baseline_mae": float(np.mean(np.abs(yv - cat["base"]))),
        "mean7_rule_mae": float(np.mean(np.abs(yv - cat["mean7"]))),
        "coverage_static": float(np.mean((yv >= cat["lo"]) & (yv <= cat["hi"]))),
        "coverage_adaptive": float(np.mean((yv >= cat["lo_a"]) & (yv <= cat["hi_a"]))),
        "median_width_static": width_s, "median_width_adaptive": width_a,
        "false_confidence_static": fc_static, "false_confidence_adaptive": fc_adapt,
        "flag": flag_scores(cat["high"], cat["flag"], one),
        "folds_model_beats_primary": int((fdf.improvement_vs_primary > 0).sum()),
        "folds_model_beats_primary_by_5pct": int((fdf.improvement_vs_primary >= 0.05).sum()),
        "folds_model_beats_mean7": int((fdf.improvement_vs_mean7 > 0).sum()),
    }
    pooled_summary["improvement_vs_primary"] = improvement(pooled_summary["primary_baseline_mae"], pooled_summary["model_mae"])
    pooled_summary["improvement_vs_mean7"] = improvement(pooled_summary["mean7_rule_mae"], pooled_summary["model_mae"])
    abl_cols = [c for c in fdf.columns if c.startswith("ablation_")]
    ablation_pooled = {c.replace("ablation_", "").replace("_mae", ""): float(np.average(fdf[c], weights=fdf.scored_hours))
                       for c in abl_cols}
    ablation_pooled |= {b: float(np.average(fdf[f"{b}_mae"], weights=fdf.scored_hours)) for b in ALL_BASELINES}
    seasons = (fdf.groupby("season")
               .apply(lambda g: pd.Series({
                   "folds": len(g), "model_mae": np.average(g.model_mae, weights=g.scored_hours),
                   "primary_mae": np.average(g[f"{primary}_mae"], weights=g.scored_hours),
                   "improvement_vs_primary": np.average(g.improvement_vs_primary, weights=g.scored_hours),
                   "coverage_static": np.average(g.coverage_static, weights=g.scored_hours),
                   "coverage_adaptive": np.average(g.coverage_adaptive, weights=g.scored_hours),
                   "mean_ks_vs_train": g.ks_vs_train.mean()}), include_groups=False)
               .reset_index())
    shift = {
        "spearman_ks_vs_coverage_gap_static": _spearman(fdf.ks_vs_train, (fdf.coverage_static - level).abs()),
        "spearman_ks_vs_model_mae": _spearman(fdf.ks_vs_train, fdf.model_mae),
        "spearman_ks_vs_improvement": _spearman(fdf.ks_vs_train, fdf.improvement_vs_primary),
        "spearman_mean_ratio_vs_coverage_static": _spearman(fdf.mean_ratio, fdf.coverage_static),
    }
    result = {
        "stage": "walk-forward backtest (pre-validation folds; test values not loaded)",
        "protocol": (f"monthly folds {cfg.backtest.first_fold}..{cfg.backtest.last_fold_end}; model refit on "
                     f"origins ending before (fold start - {cfg.backtest.calibration_days} d); interval and "
                     "monitor reference calibrated on the next 44 d; thresholds from hours before the fold; "
                     "frozen spec and flag level from validation"),
        "chosen_model": chosen.name, "primary_baseline": primary,
        "pooled": pooled_summary, "seasons": seasons.to_dict(orient="records"),
        "ablation_pooled_mae": ablation_pooled, "shift_correlations": shift,
        "monitor_adoption": {"adopted": adopt,
                             "rule": f"adopt if pooled false-confidence falls and median width grows <= "
                                     f"{mc.adopt_max_width_increase:.0%}",
                             "false_confidence_static": fc_static, "false_confidence_adaptive": fc_adapt,
                             "width_increase": width_a / width_s - 1},
    }
    out = cfg.results_dir / "backtest"
    write_json(result, out / "backtest.json")
    write_csv(fdf, out / "folds.csv")
    from .report import backtest_figure
    backtest_figure(fdf, primary, out / "backtest_folds.png")
    lines = ["# Walk-forward backtest (generated by `wattline backtest`)", "", result["protocol"], "",
             "## Pooled", "```json", json.dumps(pooled_summary, indent=1, default=float), "```", "",
             "## By season", md_table(seasons, ".3f"), "",
             "## Learned vs rules ablation (pooled MAE, kWh)",
             md_table(pd.DataFrame([ablation_pooled]), ".4f"), "",
             "## Shift correlations (Spearman across folds)", "```json", json.dumps(shift, indent=1), "```", "",
             "## Monitor adoption", "```json", json.dumps(result["monitor_adoption"], indent=1), "```", "",
             "## Folds", md_table(fdf[["fold", "season", "scored_hours", "model_mae", f"{primary}_mae",
                                       "mean_7d_same_hour_mae", "improvement_vs_primary", "coverage_static",
                                       "coverage_adaptive", "alarm_origins", "mean_ratio", "ks_vs_train",
                                       "psi_vs_train"]], ".3f")]
    (out / "backtest.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return result


# =========================================================================== freeze
def run_freeze(cfg: Config) -> dict:
    sel = read_json(cfg.results_dir / "validation" / "selection.json")
    bt = read_json(cfg.results_dir / "backtest" / "backtest.json")
    prep = prepare(cfg, upto=cfg.val_end_utc)
    spec = ModelSpec(**sel["chosen_model"])
    tr = (prep.split == "train") & prep.eligible
    model = fit_model(spec, prep.X, prep.Y, tr, cfg.model.random_seed,
                      {"max_iter": cfg.model.hgb_max_iter, "learning_rate": cfg.model.hgb_learning_rate,
                       "max_leaf_nodes": cfg.model.hgb_max_leaf_nodes})
    val_mae = mae(prep.Y, model.predict(prep.X), prep.mask(prep.split == "validation"))
    if abs(val_mae - sel["chosen_val_mae"]) > 1e-9:
        raise RuntimeError(f"refit is not deterministic: val MAE {val_mae} vs {sel['chosen_val_mae']}")
    frozen = cfg.results_dir / "frozen"
    model_path = save_model(model, frozen / "model")
    train_obs = prep.observed[prep.index < cfg.train_end_utc]
    artifacts = {
        "conformal": sel["conformal"],
        "advisory_rule": sel["advisory_rule"],
        "fill_profile": prep.profile.tolist(),
        "train_median_kwh": float(np.nanmedian(train_obs)),
        "monitor_ref": sel["monitor_ref"],
        "monitor_adopted": bt["monitor_adoption"]["adopted"],
        "primary_baseline": sel["primary_baseline"],
        "interval_level": cfg.interval.level,
        "feature_names": FEATURE_SETS[spec.feature_set],
    }
    art_path = write_json(artifacts, frozen / "artifacts.json")
    decisions = {
        "plan": "docs/PLAN.md (incl. amendment A1)",
        "config_decision_sha256": cfg.decision_hash(),
        "raw_data_sha256": cfg.data.sha256,
        "issue_time": f"{cfg.split.issue_hour_local:02d}:00 {cfg.data.local_tz}, data cutoff = issue time",
        "split_local_end_exclusive": {"train": cfg.split.train_end, "validation": cfg.split.val_end,
                                      "test": cfg.split.test_end},
        "features": FEATURE_SETS[spec.feature_set],
        "model": spec.to_dict(),
        "primary_baseline": sel["primary_baseline"],
        "interval_method": "split-conformal on log1p residuals of validation, 80 %",
        "advisory_flag_level": sel["advisory_rule"]["flag_level"],
        "monitor_adopted": bt["monitor_adoption"]["adopted"],
    }
    numeric = {
        "high_kwh": sel["advisory_rule"]["high_kwh"], "low_kwh": sel["advisory_rule"]["low_kwh"],
        "conformal_levels": sel["conformal"]["levels"], "chosen_val_mae": sel["chosen_val_mae"],
        "train_median_kwh": artifacts["train_median_kwh"],
    }
    manifest = {
        "wattline_version": __version__,
        "decisions": decisions,
        "decisions_sha256": _sha(decisions),
        "numeric_fingerprint": numeric,
        "model_file": {"name": model_path.name, "sha256": sha256_file(model_path)},
        "artifacts_file": {"name": art_path.name, "sha256": sha256_file(art_path)},
        "origins": origin_log(prep),
        "note": "test-period values were not loaded when this manifest was written",
    }
    write_json(manifest, frozen / "frozen_manifest.json")
    return manifest


# =========================================================================== final evaluation
class FrozenMismatch(RuntimeError):
    pass


def _check_lock(manifest: dict, lock_path) -> dict | None:
    if not lock_path.exists():
        return None
    lock = read_json(lock_path)
    if lock["decisions_sha256"] != manifest["decisions_sha256"]:
        raise FrozenMismatch(
            "The locked test was already scored with different frozen decisions "
            f"({lock['decisions_sha256'][:12]} != {manifest['decisions_sha256'][:12]}). "
            "Re-scoring the test after changing the model, split, features or thresholds is refused. "
            "Restore the original decisions (git) to regenerate the original results.")
    a, b = lock["numeric_fingerprint"], manifest["numeric_fingerprint"]
    flat_a, flat_b = json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True)
    if flat_a != flat_b:
        va = np.array([float(x) for x in _flatten(a)])
        vb = np.array([float(x) for x in _flatten(b)])
        if va.shape != vb.shape or not np.allclose(va, vb, rtol=1e-6, atol=1e-9):
            raise FrozenMismatch("frozen numeric artifacts (thresholds/interval quantiles) changed since "
                                 "the locked test was first scored; refusing to re-score")
    return lock


def _flatten(d):
    if isinstance(d, dict):
        for k in sorted(d):
            yield from _flatten(d[k])
    else:
        yield d


def _git_commit(cfg: Config) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cfg.root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def run_final_eval(cfg: Config) -> dict:
    from .report import final_figures
    from .serving import load_artifacts, make_forecast

    frozen = cfg.results_dir / "frozen"
    manifest = read_json(frozen / "frozen_manifest.json")
    if manifest["decisions"]["config_decision_sha256"] != cfg.decision_hash():
        raise FrozenMismatch("config/wattline.toml changed after freeze; run freeze again before evaluating")
    out = cfg.results_dir / "final"
    lock_path = out / "final_eval_lock.json"
    lock = _check_lock(manifest, lock_path)
    art = load_artifacts(cfg)
    prep = prepare(cfg, upto=None)
    tz, level = cfg.data.local_tz, art.level
    te = prep.split == "test"
    rows = np.flatnonzero(te)
    M = prep.mask(te)
    Y = prep.Y

    # ---- leakage / boundary audit ------------------------------------------------------
    first_test_target = prep.origins[te][0]
    last_val_target_end = (prep.origins[prep.split == "validation"][-1] + pd.Timedelta(hours=H))
    last_train_target_end = (prep.origins[prep.split == "train"][-1] + pd.Timedelta(hours=H))
    audit = {
        "train_targets_end_before": last_train_target_end.tz_convert(tz).isoformat(),
        "validation_targets_end_before": last_val_target_end.tz_convert(tz).isoformat(),
        "first_test_target_hour": first_test_target.tz_convert(tz).isoformat(),
        "test_starts_after_train_and_validation": bool(first_test_target >= last_val_target_end >= last_train_target_end),
        "all_feature_inputs_end_at_or_before_issue_time": all(
            r["within_cutoff"] for o in prep.origins[te] for r in provenance(o, tz)),
        "frozen_manifest_written_before_test_loaded": True,
    }
    if not (audit["test_starts_after_train_and_validation"] and audit["all_feature_inputs_end_at_or_before_issue_time"]):
        raise RuntimeError(f"boundary audit failed: {audit}")

    # ---- predictions --------------------------------------------------------------------
    P = art.model.predict(prep.X)
    mc = cfg.monitor
    states = batch_monitor(Y, P, prep.target_ns, prep.origin_ns, art.monitor_ref, mc.window_days,
                           mc.min_scored_hours, mc.alarm_mae_ratio, mc.max_widen, rows=te,
                           B=prep.base[art.primary_baseline], skill_ratio_limit=mc.skill_alarm_ratio)
    scale, alarm = np.ones(len(P)), np.zeros(len(P), bool)
    scale[te] = [s.scale for s in states]
    alarm[te] = [s.alarm for s in states]
    lo_s, hi_s = art.conformal.bounds(P, level)
    lo_a, hi_a = art.conformal.bounds(P, level, scale[:, None])
    lo, hi = (lo_a, hi_a) if art.monitor_adopted else (lo_s, hi_s)

    # ---- the operator tool, run exactly as served, for every test issue time ------------
    series = load_hourly(cfg)
    status = np.full(len(P), "", dtype=object)
    labels = np.full(P.shape, "", dtype=object)
    serve_max_diff = 0.0
    for i in rows:
        rep = make_forecast(cfg, art, series, prep.origins[i])
        status[i] = rep["status"]
        if rep["status"] == "UNAVAILABLE":
            labels[i] = "UNAVAILABLE"
            continue
        f = np.array([hh["forecast_kwh"] for hh in rep["hours"]])
        u = np.array([hh["upper80_kwh"] for hh in rep["hours"]])
        serve_max_diff = max(serve_max_diff, float(np.max(np.abs(f - P[i]))), float(np.max(np.abs(u - hi[i]))))
        labels[i] = [hh["advisory"] for hh in rep["hours"]]
    if serve_max_diff > 1e-3:
        raise RuntimeError(f"served forecast differs from evaluated forecast by {serve_max_diff}")

    # ---- metrics ------------------------------------------------------------------------
    preds = {"model": P, **prep.base}
    primary = art.primary_baseline
    model_mae, base_mae = mae(Y, P, M), {b: mae(Y, prep.base[b], M) for b in ALL_BASELINES}
    best_naive = min(NAIVE_BASELINES, key=lambda b: base_mae[b])
    headline = {
        "test_origins": int(te.sum()), "eligible_test_origins": int((te & prep.eligible).sum()),
        "scored_hours": int(M.sum()), "missing_target_hours": int(np.isnan(Y[te]).sum()),
        "model": art.model.spec.name,
        "model_mae_kwh": model_mae, "model_rmse_kwh": rmse(Y, P, M),
        "baselines": {b: {"mae_kwh": base_mae[b], "rmse_kwh": rmse(Y, prep.base[b], M)} for b in ALL_BASELINES},
        "primary_baseline": primary,
        "improvement_vs_primary": improvement(base_mae[primary], model_mae),
        "bar_5pct_mae_improvement_met": bool(improvement(base_mae[primary], model_mae) >= 0.05),
        "best_naive_on_test (descriptive only)": best_naive,
        "improvement_vs_best_naive_on_test": improvement(base_mae[best_naive], model_mae),
        "improvement_vs_mean7_rule": improvement(base_mae["mean_7d_same_hour"], model_mae),
    }
    pom = per_origin_mae(Y, P, M)
    pob = per_origin_mae(Y, prep.base[primary], M)
    po7 = per_origin_mae(Y, prep.base["mean_7d_same_hour"], M)
    cov_o = [coverage(Y[i : i + 1], lo[i : i + 1], hi[i : i + 1], M[i : i + 1]) if M[i].any() else np.nan
             for i in range(len(P))]
    odf = pd.DataFrame({
        "origin_local": [o.tz_convert(tz).isoformat() for o in prep.origins[te]],
        "day_type": prep.day_type[te][:, 0], "status": status[te],
        "scored_hours": M[te].sum(axis=1), "model_mae": pom[te], f"{primary}_mae": pob[te],
        "mean_7d_same_hour_mae": po7[te], "coverage": np.array(cov_o)[te],
        "max_actual_kwh": np.nanmax(Y[te], axis=1), "monitor_scale": scale[te], "monitor_alarm": alarm[te],
    })
    worst_i = rows[int(np.nanargmax(pom[te]))]
    best_i = rows[int(np.nanargmin(pom[te]))]
    hdf = per_horizon(Y, preds, M)
    hdf["coverage"] = [coverage(Y[:, h], lo[:, h], hi[:, h], M[:, h]) for h in range(H)]
    hdf["median_width"] = [median_width(lo[:, h], hi[:, h], M[:, h]) for h in range(H)]
    hour_df = per_group(Y, {"model": P, primary: prep.base[primary]}, M, prep.hour_local, "hour_local")
    hour_df["coverage"] = [coverage(Y, lo, hi, M & (prep.hour_local == h)) for h in hour_df.hour_local]
    dt_df = per_group(Y, {"model": P, primary: prep.base[primary], "mean_7d_same_hour": prep.base["mean_7d_same_hour"]},
                      M, prep.day_type, "day_type")

    # ---- uncertainty ----------------------------------------------------------------------
    pre = prep.index < cfg.val_end_utc
    q10, q90 = np.nanquantile(prep.observed[pre], [0.1, 0.9])
    clim_lo, clim_hi = np.full_like(P, q10), np.full_like(P, q90)
    above = M & (Y > hi)
    below = M & (Y < lo)
    unc = {
        "policy": "adaptive (monitor adopted)" if art.monitor_adopted else "static validation interval",
        "nominal": level, "coverage": coverage(Y, lo, hi, M), "median_width_kwh": median_width(lo, hi, M),
        "bar_coverage_ge_70pct_met": bool(coverage(Y, lo, hi, M) >= 0.70),
        "static": {"coverage": coverage(Y, lo_s, hi_s, M), "median_width_kwh": median_width(lo_s, hi_s, M)},
        "adaptive": {"coverage": coverage(Y, lo_a, hi_a, M), "median_width_kwh": median_width(lo_a, hi_a, M),
                     "alarm_origins": int(alarm[te].sum()), "mean_scale": float(scale[te].mean())},
        "misses_above_upper_share": float(above.sum() / M.sum()),
        "misses_below_lower_share": float(below.sum() / M.sum()),
        "mean_excess_above_upper_kwh": float(np.mean((Y - hi)[above])) if above.any() else 0.0,
        "climatology_band_reference": {"band_kwh": [float(q10), float(q90)],
                                       "coverage": coverage(Y, clim_lo, clim_hi, M),
                                       "width_kwh": float(q90 - q10)},
    }

    # ---- advisory / peak-event study --------------------------------------------------------
    rule = art.rule
    actual_high = np.where(np.isnan(Y), False, Y >= rule.high_kwh)
    served = M & (status != "UNAVAILABLE")[:, None]
    lab_static = label_matrix(art.conformal.quantile(P, rule.flag_level), hi_s, rule, np.zeros(len(P), bool))
    status_counts = pd.Series(status[te]).value_counts().to_dict()
    label_counts = pd.Series(labels[te].ravel()).value_counts().to_dict()
    adv = {
        "rule": rule.to_dict(),
        "flags_as_served": flag_scores(actual_high, labels == HIGH, served),
        "primary_baseline_flags": flag_scores(actual_high, prep.base[primary] >= rule.high_kwh, served),
        "peak_top3_hit_rate_model": peak_hit_rate(Y, P, M, cfg.advisory.top_k_peaks),
        "peak_top3_hit_rate_primary_baseline": peak_hit_rate(Y, prep.base[primary], M, cfg.advisory.top_k_peaks),
        "peak_top3_hit_rate_mean7_rule": peak_hit_rate(Y, prep.base["mean_7d_same_hour"], M, cfg.advisory.top_k_peaks),
        "report_status_counts": status_counts,
        "hour_label_counts": label_counts,
        "false_confidence_as_served": false_confidence(Y, labels, hi, rule.high_kwh, served),
        "false_confidence_static_reference": false_confidence(Y, lab_static, hi_s, rule.high_kwh, served),
    }

    # ---- DST day + distribution shift (descriptive, after freeze) ----------------------------
    dst = [i for i in rows if prep.origins[i].tz_convert(tz).date() == pd.Timestamp("2010-10-31").date()]
    dst_info = {}
    if dst:
        i = dst[0]
        dst_info = {"origin_local": prep.origins[i].tz_convert(tz).isoformat(), "model_mae": pom[i],
                    "scored_hours": int(M[i].sum()), "missing_target_hours": int(np.isnan(Y[i]).sum()),
                    "rank_among_test_days_worst_first": int((pom[te] > pom[i]).sum() + 1),
                    "target_hours_local": [t.isoformat() for t in pd.to_datetime(prep.target_ns[i], utc=True).tz_convert(tz)][:4] + ["..."]}
    val_vals = prep.observed[(prep.index >= cfg.train_end_utc) & (prep.index < cfg.val_end_utc)]
    test_vals = prep.observed[(prep.index >= cfg.val_end_utc) & (prep.index < cfg.test_end_utc)]
    train_vals = prep.observed[prep.index < cfg.train_end_utc]
    shift = {"mean_kwh": {"train": float(np.nanmean(train_vals)), "validation": float(np.nanmean(val_vals)),
                          "test": float(np.nanmean(test_vals))},
             "ks_test_vs_validation": _ks(val_vals, test_vals), "psi_test_vs_validation": _psi(val_vals, test_vals),
             "ks_test_vs_train": _ks(train_vals, test_vals)}

    worst = odf.iloc[int(np.nanargmax(odf.model_mae))].to_dict()
    best = odf.iloc[int(np.nanargmin(odf.model_mae))].to_dict()
    metrics = {
        "stage": "final evaluation on the locked test window",
        "frozen_decisions_sha256": manifest["decisions_sha256"],
        "boundary_audit": audit,
        "headline": headline,
        "worst_origin": worst, "best_origin": best,
        "uncertainty": unc, "advisory": adv, "dst_transition_day": dst_info,
        "distribution_shift": shift,
        "serving_equivalence_max_abs_diff_kwh": serve_max_diff,
    }

    # ---- write --------------------------------------------------------------------------------
    write_json(metrics, out / "metrics.json")
    write_csv(hdf, out / "per_horizon.csv")
    write_csv(hour_df, out / "per_hour_of_day.csv")
    write_csv(dt_df, out / "per_day_type.csv")
    write_csv(odf, out / "per_origin.csv")
    long = []
    for i in rows:
        tt = pd.to_datetime(prep.target_ns[i], utc=True)
        for h in range(H):
            long.append({"origin_local": prep.origins[i].tz_convert(tz).isoformat(), "horizon": h + 1,
                         "target_utc": tt[h].isoformat(), "target_local": tt[h].tz_convert(tz).isoformat(),
                         "actual_kwh": Y[i, h], "scored": bool(M[i, h]), "forecast_kwh": P[i, h],
                         "lower80_kwh": lo[i, h], "upper80_kwh": hi[i, h],
                         **{f"{b}_kwh": prep.base[b][i, h] for b in ALL_BASELINES},
                         "report_status": status[i], "advisory": labels[i, h]})
    write_csv(pd.DataFrame(long), out / "test_predictions.csv")
    final_figures(cfg, prep, P, lo, hi, M, hdf, hour_df, worst_i, best_i, primary, rule, out / "figures")
    _write_final_md(metrics, hdf, hour_df, dt_df, odf, primary, out)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if lock is None:
        write_json({"first_scored_utc": now, "decisions_sha256": manifest["decisions_sha256"],
                    "numeric_fingerprint": manifest["numeric_fingerprint"],
                    "headline": {"model_mae_kwh": model_mae, "improvement_vs_primary": headline["improvement_vs_primary"],
                                 "coverage": unc["coverage"]}}, lock_path)
        kind = "FIRST SCORING of the locked test"
    else:
        same = abs(lock["headline"]["model_mae_kwh"] - model_mae) < 1e-9
        kind = f"REGENERATION (identical frozen decisions; first scored {lock['first_scored_utc']}; headline identical: {same})"
    log = cfg.results_dir / "final_run.log"
    with open(log, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{now} | {kind} | git {_git_commit(cfg)[:12]} | decisions {manifest['decisions_sha256'][:12]} | "
                 f"python {sys.version.split()[0]} {platform.system()} | origins {headline['test_origins']} "
                 f"scored_h {headline['scored_hours']} | MAE {model_mae:.4f} vs {primary} {base_mae[primary]:.4f} "
                 f"({headline['improvement_vs_primary']:+.2%}) | cov80 {unc['coverage']:.3f} width {unc['median_width_kwh']:.3f}\n")
    return metrics


def _write_final_md(m, hdf, hour_df, dt_df, odf, primary, out) -> None:
    h = m["headline"]
    brows = pd.DataFrame([{"model/baseline": k, "MAE kWh": v["mae_kwh"], "RMSE kWh": v["rmse_kwh"]}
                          for k, v in {"MODEL " + h["model"]: {"mae_kwh": h["model_mae_kwh"], "rmse_kwh": h["model_rmse_kwh"]},
                                       **h["baselines"]}.items()])
    u, a = m["uncertainty"], m["advisory"]
    lines = [
        "# Final locked-test results (generated by `wattline final-eval`)", "",
        f"Frozen decisions sha256 `{m['frozen_decisions_sha256'][:16]}`. "
        f"{h['test_origins']} daily issue times, {h['scored_hours']} scored hours, "
        f"{h['missing_target_hours']} missing target hours excluded for every model.", "",
        "## Accuracy (same origins, same hours for every row)", md_table(brows, ".4f"), "",
        f"Improvement vs locked primary baseline `{primary}`: **{h['improvement_vs_primary']:+.2%}** "
        f"(bar >= 5 %: {'MET' if h['bar_5pct_mae_improvement_met'] else 'NOT MET'}); vs 7-day mean rule "
        f"{h['improvement_vs_mean7_rule']:+.2%}.", "",
        f"Worst origin: {m['worst_origin']['origin_local']} - model MAE {m['worst_origin']['model_mae']:.3f} kWh "
        f"(primary {m['worst_origin'][primary + '_mae']:.3f}). Best origin: {m['best_origin']['origin_local']} - "
        f"{m['best_origin']['model_mae']:.3f} kWh.", "",
        "## 80 % interval",
        f"Policy: {u['policy']}. Coverage **{u['coverage']:.3f}** (bar >= 0.70: {'MET' if u['bar_coverage_ge_70pct_met'] else 'NOT MET'}), "
        f"median width {u['median_width_kwh']:.3f} kWh. Static {u['static']['coverage']:.3f} / {u['static']['median_width_kwh']:.3f}; "
        f"adaptive {u['adaptive']['coverage']:.3f} / {u['adaptive']['median_width_kwh']:.3f}. Misses above upper "
        f"{u['misses_above_upper_share']:.1%}, below lower {u['misses_below_lower_share']:.1%}. Climatology band "
        f"{u['climatology_band_reference']['band_kwh'][0]:.2f}-{u['climatology_band_reference']['band_kwh'][1]:.2f} kWh covers "
        f"{u['climatology_band_reference']['coverage']:.3f} at width {u['climatology_band_reference']['width_kwh']:.3f}.", "",
        "## Advisory",
        f"HIGH_DEMAND flags as served: precision {a['flags_as_served']['precision']:.3f}, recall {a['flags_as_served']['recall']:.3f}, "
        f"F1 {a['flags_as_served']['f1']:.3f} (base rate {a['flags_as_served']['base_rate']:.3f}); primary-baseline flags F1 "
        f"{a['primary_baseline_flags']['f1']:.3f}. Actual daily peak in predicted top-3 hours: model "
        f"{a['peak_top3_hit_rate_model']:.3f}, primary baseline {a['peak_top3_hit_rate_primary_baseline']:.3f}, "
        f"7-day mean rule {a['peak_top3_hit_rate_mean7_rule']:.3f}.",
        f"Report statuses: {a['report_status_counts']}. Hour labels: {a['hour_label_counts']}.", "",
        "## Per horizon", md_table(hdf, ".3f"), "",
        "## Per local hour of day", md_table(hour_df, ".3f"), "",
        "## Per day type", md_table(dt_df, ".3f"), "",
        "## Per origin", md_table(odf, ".3f"), "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


# =========================================================================== reproduce
def run_reproduce(cfg: Config) -> dict:
    from .ingest import ingest
    from .util import no_network
    with no_network():
        steps = {}
        steps["ingest"] = ingest(cfg)["hourly"]["hours"]
        steps["select"] = run_select(cfg)["chosen_model_name"]
        steps["backtest"] = run_backtest(cfg)["pooled"]["model_mae"]
        steps["freeze"] = run_freeze(cfg)["decisions_sha256"]
        steps["final"] = run_final_eval(cfg)["headline"]
    return steps
