"""Descriptive peak staggering simulation on the saved locked test predictions.

Runs the simulation fixed in docs/EXTENSIONS.md once, unchanged, on
results/final/test_predictions.csv. It fits nothing and tunes nothing: the forecasts are
the ones already served and scored in v1.0.0. Output: results/extensions/staggering_test.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from extension_study import stagger_day, stagger_summary

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "final" / "test_predictions.csv"
METRICS = ROOT / "results" / "final" / "metrics.json"
OUT = ROOT / "results" / "extensions" / "staggering_test.json"


def main() -> None:
    high = json.loads(METRICS.read_text())["advisory"]["rule"]["high_kwh"]
    d = pd.read_csv(PRED)
    d["hour"] = pd.to_datetime(d["target_local"].str[:19]).dt.hour
    model_days, naive_days, skipped = [], [], 0
    for _, g in d.groupby("origin_local", sort=True):
        g = g.sort_values("horizon")
        if len(g) != 24 or not g["scored"].all() or not (g["hour"].to_numpy() == np.arange(24)).all():
            skipped += 1
            continue
        y = g["actual_kwh"].to_numpy()
        model_days.append(stagger_day(y, g["forecast_kwh"].to_numpy()))
        naive_days.append(stagger_day(y, g["seasonal_naive_day_kwh"].to_numpy()))
    result = {
        "source": "results/final/test_predictions.csv (locked test, descriptive, not tuned)",
        "days_used": len(model_days), "days_skipped": skipped,
        "model_policy": stagger_summary(model_days, high),
        "naive_policy": stagger_summary(naive_days, high),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
