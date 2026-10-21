# wattline

A 24-hour-ahead hourly electricity-demand forecaster for one site, built as a
**time-ordered decision** rather than a fit-and-plot exercise. Every forecast is issued at
local midnight from data that ends at that instant, it is scored against naive baselines on
identical origins, it carries an 80 % interval calibrated on validation only, and it refuses
loudly when the input cannot support a forecast. Advisory only: it never controls equipment.

| | |
|---|---|
| Site / data | one household in Sceaux, France — UCI *Individual Household Electric Power Consumption*, CC BY 4.0, 2006-12-16 → 2010-11-26, 1-minute → hourly |
| Target | energy in each clock hour, **kWh**, next 24 hours |
| Issue time | **00:00 Europe/Paris**; data cutoff = issue time |
| Model | 24 direct ridge regressions (lags + calendar), stored as plain JSON |
| Locked test | 2010-10-15 → 2010-11-25, 42 daily issue times, 1006 scored hours, scored **once** |

## Results on the locked test (from `results/final/metrics.json`)

| | MAE kWh | RMSE kWh |
|---|---|---|
| **ridge (lags + calendar)** | **0.494** | **0.695** |
| seasonal-naive-day (primary baseline, locked on validation) | 0.639 | 0.948 |
| seasonal-naive-week | 0.723 | 1.025 |
| last-value | 0.826 | 1.090 |
| 7-day same-hour mean (reference rule) | 0.544 | 0.755 |

- **+22.7 % MAE vs the locked primary baseline** (bar: ≥ 5 %); +9.2 % vs the 7-day-mean rule.
  Better than the primary baseline on 36 of 42 days.
- **80 % interval coverage 0.794**, median width 1.60 kWh (bar: ≥ 0.70). This figure comes from
  the served adaptive interval, which widens during the test period using recent actuals that are
  already available at each issue time. The strict, validation only static interval covered
  **0.734** (median width 1.37 kWh); that is the conservative figure.
- **Worst day:** Mon 2010-10-18, MAE 0.98 kWh, interval coverage 0.54 — an unusual all-day
  high-load Monday the model smoothed over (`results/final/figures/worst_day.png`).
- **HIGH_DEMAND flags:** precision 0.31, recall 0.51 (base rate 0.11) vs F1 0.28 for flags
  from the primary baseline. Actual daily peak inside the predicted top-3 hours on 41 % of days.
- Walk-forward backtest (21 monthly folds, pre-validation): model beats the primary baseline
  in 20/21 folds, **but loses badly in August 2010 when the household was away** — see
  [docs/EVALUATION.md](docs/EVALUATION.md).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/) (dependencies are locked in `uv.lock`).

```bash
uv sync --frozen --extra dev
```

```bash
uv run wattline download
```

`download` is the **only** command that uses the network. It fetches the UCI zip, checks its
SHA-256 against `config/wattline.toml` and makes the raw snapshot read-only under `data/raw/`.

```bash
uv run wattline reproduce
```

`reproduce` regenerates everything offline (sockets are disabled while it runs): ingest →
select → backtest → freeze → final-eval. Because the locked test has already been scored, the
final step is a *regeneration*: it verifies the frozen decisions are unchanged and appends a
`REGENERATION` line to `results/final_run.log`. If a decision changed, it refuses (exit code 4).

### Stage by stage

| command | what it does | writes |
|---|---|---|
| `wattline ingest` | raw minutes → canonical hourly table; clock/DST detection, unit cross-checks, missing/duplicate/outlier audit | `data/interim/hourly.csv`, `results/data/` |
| `wattline select` | **validation only** (test not loaded): baselines, primary baseline, ridge grid, feature ablation, boosted-tree check, interval, thresholds, flag rule | `results/validation/` |
| `wattline backtest` | walk-forward monthly folds before validation; shift stats; monitor adoption | `results/backtest/` |
| `wattline freeze` | refit chosen model, write model + artifacts + manifest with hashes | `results/frozen/` |
| `wattline final-eval` | score the locked test once (later runs only regenerate identical results) | `results/final/`, `results/final_run.log` |

### Forecast for one issue time (the operator tool)

```bash
uv run wattline forecast --issue-time 2010-11-22T00:00+01:00
```

Writes `reports/forecast_<time>/forecast.{json,csv,png,html}` and prints the 24 hours, the 80 %
interval, the likely peak hours, a per-hour advisory and the reasons for the status. No feature
file needs editing. To forecast from your own meter export:

```bash
uv run wattline forecast --issue-time 2010-11-22T00:00+01:00 --input my_meter.csv
```

**Input contract:** CSV with `timestamp_utc` (or `timestamp`) carrying an explicit UTC offset,
and `energy_kwh` = energy of the hour starting at that timestamp. One row per hour. An optional
`unit` column must read `kWh`. Rows at or after the issue time are ignored.

**Status and exit codes**

| status | exit | meaning |
|---|---|---|
| `OK` | 0 | forecast usable as an advisory; hours labelled HIGH_DEMAND or NORMAL (LOW is defined but never produced, see below) |
| `UNCERTAIN` | 3 | forecast shown, but **no hour is called NORMAL or LOW** (late/stuck meter, gaps, drift or skill alarm) |
| `UNAVAILABLE` | 2 | nothing forecast: stale > 6 h, < 168 h history, wrong unit, duplicate or naive timestamps, wrong issue hour, > 25 % missing |

**What the hour labels can and cannot tell you.** The `LOW` label is defined in the status
vocabulary but is never produced by the current system: even a forecast of 0 kWh receives an
80 % upper bound of 0.365 kWh, which is above the 0.340 kWh LOW threshold, and the monitor can
only widen the interval. The `HIGH_DEMAND` flag had precision 0.307 and recall 0.514 on the locked
test, so roughly two flags in three are false alarms and about half of the high hours are missed.
Treat it as an advisory to check, not as an alarm to act on alone.

Try the failure cases (clean file plus eight deliberately broken exports):

```bash
uv run wattline demo-inputs --issue-time 2010-11-22T00:00+01:00
```

```bash
uv run wattline forecast --issue-time 2010-11-22T00:00+01:00 --input reports/demo_inputs/stale_meter_10h.csv
```

Pre-rendered outputs for every case are in [`reports/examples/`](reports/examples/).

### Tests

```bash
uv run pytest
```

57 tests: resampling, timezone and DST handling, no-leakage corruption tests, forecast-window
alignment, evaluation math against hand calculations, invalid/stale input, the final-eval lock,
and an end-to-end pipeline on synthetic UCI-format data with network access disabled.

## Repository map

```
config/wattline.toml      every decision (splits, grids, thresholds, monitor) - hashed into the manifest
src/wattline/             ingest, clean, features, baselines, model, intervals, monitor, serving, pipeline, report, cli
tests/                    unit + end-to-end tests (synthetic data generator in tests/synthetic.py)
results/data/             ingest report, missing-hour runs
results/validation/       baseline + model selection, flag study (test not loaded)
results/backtest/         walk-forward folds, shift stats, monitor adoption
results/frozen/           frozen_manifest.json, model.json, artifacts.json
results/final/            metrics.json, per-horizon/hour/day-type/origin tables, test_predictions.csv, figures
results/final_run.log     append-only log of the first scoring and every regeneration
reports/                  example operator reports and demo inputs
docs/                     PLAN, DATA_CARD, MODEL_CARD, EVALUATION, PROCESS_LOG, RETROSPECTIVE, DEMO_SCRIPT, CHECKIN
```

## Scope and limits

The data comes from one French household in Sceaux (UCI dataset 235), not from a commercial or
industrial facility. The results demonstrate the method and the evaluation discipline: a fixed
issue time with an explicit information boundary, baselines scored on identical origins, a single
frozen test evaluation and explicit failure states. They do not show that this model is a
validated forecaster for any UAE facility or grid. A UAE deployment would need local metered data,
a local calendar (weekend and public holidays) and local weather inputs, followed by a fresh
plan, a fresh chronological split and a fresh frozen evaluation.

## What this forecast must never be used to claim

It does not know the weather, occupancy or appliance schedules; it has never seen another
building; and its typical error in an evening hour is about 0.7 kWh. It must not drive
equipment, dispatch or billing, and an `OK` status is not a guarantee — about one hour in five is
expected to fall outside the 80 % interval. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Licence

Code: MIT (see `LICENSE`). Data: © Georges Hébrail and Alice Bérard, UCI Machine Learning
Repository, CC BY 4.0 — not redistributed here; `wattline download` fetches it from the source.
