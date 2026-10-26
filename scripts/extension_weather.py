"""Past only temperature ablation on the pre test folds (study C in docs/EXTENSIONS.md).

Adds five temperature features that are all read from hours strictly before the issue time
(never a target hour, never a forecast) to the chosen ridge model and compares it with the
chosen model on the frozen validation split and on the 21 walk forward folds. Temperature
comes from the free Open-Meteo historical archive, requested only up to the validation end,
so no test period value is loaded. Nothing in the frozen pipeline changes.
Outputs: results/extensions/weather_ablation.json and weather_ablation_folds.csv.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import wattline.model as wm
from wattline.config import load_config
from wattline.evaluate import improvement, mae
from wattline.features import ALL_FEATURES, FEATURE_SETS, H
from wattline.model import ModelSpec, fit_model
from wattline.pipeline import SEASON, prepare
from wattline.util import read_json, write_csv, write_json

LAT, LON = 48.78, 2.29
START = "2006-12-16"
TEMP_FEATURES = ["t_last", "t_mean24", "t_min24", "hdd24", "t_lag24"]
HDD_BASE = 15.0


def fetch(cfg, end_date: str) -> tuple[pd.Series, str]:
    cache = Path(cfg.root) / "data" / "external" / "open_meteo_temperature_2m.json"
    if not cache.exists():
        url = ("https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={LAT}&longitude={LON}&start_date={START}&end_date={end_date}"
               "&hourly=temperature_2m&timezone=UTC")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            cache.write_bytes(r.read())
    raw = cache.read_bytes()
    j = json.loads(raw)["hourly"]
    idx = pd.to_datetime(j["time"]).tz_localize("UTC")
    return pd.Series(j["temperature_2m"], index=idx, dtype=float), hashlib.sha256(raw).hexdigest()


def temp_features(T: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """(n_origins, 24, 5) features read only from positions before each origin."""
    out = np.full((len(pos), H, len(TEMP_FEATURES)), np.nan)
    for i, p in enumerate(pos):
        day = T[p - 24:p]
        out[i, :, 0] = T[p - 1]
        out[i, :, 1] = np.nanmean(day)
        out[i, :, 2] = np.nanmin(day)
        out[i, :, 3] = np.nanmean(np.clip(HDD_BASE - day, 0, None))
        out[i, :, 4] = T[p - 24:p]                    # horizon h reads hour p + h - 24
    return out


def main() -> None:
    cfg = load_config()
    sel = read_json(cfg.results_dir / "validation" / "selection.json")
    chosen = ModelSpec(**sel["chosen_model"])
    prep = prepare(cfg, upto=cfg.val_end_utc)           # structural test blindness
    end_date = cfg.val_end_utc.strftime("%Y-%m-%d")
    temp, sha = fetch(cfg, end_date)
    T = temp.reindex(prep.index).to_numpy(dtype=float)
    missing = int(np.isnan(T).sum())
    T = pd.Series(T).interpolate(limit_direction="both").to_numpy()  # past values only matter
    TF = temp_features(T, prep.pos)
    X2 = np.concatenate([prep.X, TF], axis=2)

    # register the temporary feature set without touching the frozen definitions on disk
    wm.ALL_FEATURES = list(ALL_FEATURES) + TEMP_FEATURES
    wm.FEATURE_SETS = dict(FEATURE_SETS)
    wm.FEATURE_SETS["lags_calendar_temp"] = FEATURE_SETS["lags_calendar"] + TEMP_FEATURES
    spec_t = replace(chosen, feature_set="lags_calendar_temp")

    tz = cfg.data.local_tz
    end = prep.origins + pd.Timedelta(hours=H)
    # validation (frozen split)
    tr = prep.eligible & (prep.split == "train")
    va = prep.eligible & (prep.split == "validation")
    Mv = prep.mask(va)
    m0 = fit_model(chosen, X2, prep.Y, tr, cfg.model.random_seed)
    m1 = fit_model(spec_t, X2, prep.Y, tr, cfg.model.random_seed)
    val = {"base_mae": mae(prep.Y, m0.predict(X2), Mv), "temp_mae": mae(prep.Y, m1.predict(X2), Mv)}
    val["improvement"] = improvement(val["base_mae"], val["temp_mae"])

    walls = pd.date_range(cfg.backtest.first_fold, cfg.backtest.last_fold_end, freq="MS")
    rows = []
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
        a = mae(prep.Y, fit_model(chosen, X2, prep.Y, train_rows, cfg.model.random_seed).predict(X2), M)
        b = mae(prep.Y, fit_model(spec_t, X2, prep.Y, train_rows, cfg.model.random_seed).predict(X2), M)
        rows.append({"fold": w0.strftime("%Y-%m"), "season": SEASON[w0.month],
                     "scored_hours": int(M.sum()), "mae_base": a, "mae_temp": b,
                     "improvement": improvement(a, b)})
    fdf = pd.DataFrame(rows)
    w = fdf["scored_hours"]
    pooled_base = float(np.average(fdf["mae_base"], weights=w))
    pooled_temp = float(np.average(fdf["mae_temp"], weights=w))
    seasons = {s: {"improvement": float(np.average(g["improvement"], weights=g["scored_hours"])),
                   "folds": int(len(g))} for s, g in fdf.groupby("season")}
    pooled_improvement = improvement(pooled_base, pooled_temp)
    wins = int((fdf["improvement"] > 0).sum())
    verdict = bool(pooled_improvement >= 0.02 and wins >= 12 and val["improvement"] >= 0)
    result = {
        "data": {"source": "Open-Meteo historical archive, temperature_2m, "
                           f"lat {LAT} lon {LON}, {temp.index[0]:%Y-%m-%d} to {temp.index[-1]:%Y-%m-%d} (UTC)",
                 "response_sha256": sha, "missing_hours_filled": missing},
        "features": TEMP_FEATURES, "model": chosen.name,
        "validation": val,
        "backtest": {"folds": len(fdf), "pooled_mae_base": pooled_base, "pooled_mae_temp": pooled_temp,
                     "pooled_improvement": pooled_improvement, "folds_improved": wins,
                     "by_season": seasons},
        "criteria": {"pooled_improvement_at_least": 0.02, "folds_improved_at_least": 12,
                     "validation_not_worse": True},
        "worth_its_cost": verdict,
    }
    out = cfg.results_dir / "extensions"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(fdf, out / "weather_ablation_folds.csv")
    write_json(result, out / "weather_ablation.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
