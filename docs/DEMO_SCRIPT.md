# Demo script (3–5 minutes)

Record from the repository root after `uv sync --frozen --extra dev` and `uv run wattline download`.
Commands are copy-paste ready. Timings are targets.

## 0:00 – 0:30  What it is and when it issues

- Open `README.md`: one site, kWh per hour, **issued 00:00 Europe/Paris, data cutoff = issue time**.
- Open `docs/PLAN.md` section 2 (information boundary) and show its commit is the first after setup:

```bash
git log --oneline --reverse | head -4
```

## 0:30 – 1:10  Data, licence, units, timezone

- `docs/DATA_CARD.md`: CC BY 4.0, DOI, SHA-256; unit cross-checks; the missing-hour table.
- The DST story: 1440 rows on DST days, voltage shift 0 min, spring hour copied → wall clock,
  480 rows dropped, nothing shifted.

```bash
uv run wattline ingest
```

## 1:10 – 1:50  The split is honest

- `results/final/metrics.json → boundary_audit`: train ends 2010-09-01, validation ends 10-15,
  first test target 10-15 00:00, all feature inputs end at the issue time.
- `src/wattline/pipeline.py` `prepare(cfg, upto=val_end)`: pre-test stages never load test values.
- `results/final_run.log`: one FIRST SCORING line, later lines are REGENERATION.

## 1:50 – 2:40  Baselines vs model, worst day, horizons, interval

- `results/final/report.md` accuracy table: 0.494 vs 0.639 (+22.7 %) on the same 1,006 hours.
- Open `results/final/figures/worst_day.png` (Mon 2010-10-18, MAE 0.98, coverage 0.54) — say why.
- `figures/mae_by_horizon.png` and `figures/coverage_by_horizon.png`: coverage 0.794; nights
  over-cover, afternoons under-cover.

## 2:40 – 3:20  Clean case: the operator view

```bash
uv run wattline forecast --issue-time 2010-11-22T00:00+01:00 --input reports/demo_inputs/clean.csv --out reports/demo_clean
```

Open `reports/demo_clean/forecast.html`: status OK, 24 hours with intervals, peak hours, HIGH_DEMAND
hours shaded, threshold explained (90th percentile of train+validation hours, flag when the 80 %
quantile reaches 2.34 kWh), data-cutoff table for every feature.

## 3:20 – 4:20  Break it on purpose

```bash
uv run wattline forecast --issue-time 2010-11-22T00:00+01:00 --input reports/demo_inputs/stale_meter_10h.csv --out reports/demo_stale
```

→ `UNAVAILABLE`, exit 2: "stale data: last reading ended 10 h before the issue time".

```bash
uv run wattline forecast --issue-time 2010-11-22T00:00+01:00 --input reports/demo_inputs/wrong_unit_wh.csv --out reports/demo_unit
```

→ `UNAVAILABLE`: unexpected unit `energy_wh`.

```bash
uv run wattline forecast --issue-time 2010-11-22T00:00+01:00 --input reports/demo_inputs/duplicate_timestamp.csv --out reports/demo_dup
```

→ `UNAVAILABLE`: duplicate timestamp, refusing to choose.

```bash
uv run wattline forecast --issue-time 2010-11-22T00:00+01:00 --input reports/demo_inputs/stuck_meter.csv --out reports/demo_stuck
```

→ `UNCERTAIN`, exit 3: no hour is called NORMAL or LOW.

## 4:20 – 5:00  Tests and offline regeneration

```bash
uv run pytest
```

```bash
uv run wattline reproduce
```

Point out that `reproduce` runs with sockets disabled (a network call would raise
`NetworkDisabled`) and that the last line of `results/final_run.log` says
`REGENERATION … headline identical: True`.
