# wattline plan (frozen before any modelling code)

Status: **frozen at commit time**, 2026-10-20. Changes after this commit are logged in
the "Amendments" section at the bottom with a reason, and none may use test period
information.

What I knew when writing this: the dataset's published description (span, units,
missing value marker) and its licence page. I had **not** fitted any model or looked at
any validation or test period number. Split dates below are chosen from the
calendar span alone.

---

## 1. Forecast question

| item | decision |
|---|---|
| site | one household in Sceaux (7 km from Paris), France, the single site in the UCI dataset |
| target | **hourly active energy, kWh per hour**, for the clock hour `[t, t+1h)` |
| derivation | `energy_kwh(t) = mean(Global_active_power over the minutes present in the hour) [kW] × 1 h` |
| hour validity | an hour is *observed* only if **≥ 45 of its 60 minutes** are present; otherwise it is *missing* (never a target) |
| unit | kWh. One unit end to end. The serving input contract demands a column literally named `energy_kwh`; anything else is refused |
| why useful | a day ahead load curve and likely peak hours let the occupant / facility manager plan deferrable loads in advance (laundry, water heating, EV charging) away from peaks and judge whether tomorrow looks unusual. Advisory only |

Honest framing: this is a household, not a commercial building. It is the best legally
clean, long, minute resolution single site record I could verify. The noise level of a
household (appliance spikes) makes this a *harder* point forecasting problem than a
large building, which is why interval quality and peak flag precision/recall matter more
than a smooth curve.

## 2. Forecast origin and information boundary

- **Issue time:** every day at **00:00 Europe/Paris wall clock** (local midnight).
- **Data cutoff:** the issue time itself. The last hour any feature may use is the hour
  `[issue − 1h, issue)`. Nothing timestamped at or after the issue time may enter a
  feature. This is enforced structurally (the feature builder receives an array that physically
  ends at the cutoff) and by a corruption test (changing any value at/after the cutoff must
  not change any feature).
- **Horizons:** `h = 1 … 24`. Horizon `h` predicts the hour starting at
  `issue + (h − 1) h` in absolute (UTC) time. On DST days that means the 24 hours run
  to 01:00 (autumn) or 23:00 (spring) local: "next 24 hours", not "calendar day".
- **Allowed inputs at issue time:** past observed load (lags, rolling stats), the
  causal imputation flags, and the deterministic calendar of each target hour
  (local hour, weekday, French public holiday, annual cycle).
- **Forbidden:** observed weather of the target hours, any actual from the horizon,
  any statistic fitted on validation or test data used as a feature.
- **No weather.** There is no archived *forecast* weather source for this site and
  period that I can prove existed at issue time; historical observed weather would be
  leakage. Calendar features are therefore the only exogenous inputs.

## 3. Data and licence

- **Provider:** UCI Machine Learning Repository, *Individual Household Electric Power
  Consumption* (Hebrail & Berard, 2006–2010), dataset id 235.
- **Access URL:** https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
  (zip: `https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip`)
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0) as stated on
  the UCI page. Redistribution and adaptation allowed with attribution. I still will not
  commit the raw file: the repository stores a SHA-256 of the download and a one command
  downloader, and the raw snapshot is made read only on disk.
- **Sample span (published):** 2006-12-16 17:24 → 2010-11-26 21:02, one minute
  sampling, ~2.07 M rows, ~1.25 % of rows missing (marked `?`). That is ≈ 47 months of
  hourly coverage, far above the 120 day minimum.
- **Timezone:** the file has naive `dd/mm/yyyy` + `hh:mm:ss` columns and no declared
  zone. Policy, decided now, applied at ingest:
  1. Count minutes on each DST transition date. 1380/1500 minutes ⇒ wall clock with DST;
     1440 ⇒ a fixed offset clock.
  2. If fixed offset, determine the offset with a *behavioural* check: household
     routines follow the wall clock, so if the logger clock does not move with DST the
     morning ramp will shift by ~1 h across each transition in logger time. Record the
     evidence and the chosen offset in the ingest report.
  3. Canonical timestamps are stored as **UTC interval starts**; local Europe/Paris time
     is derived only for calendar features and display. No hour is silently shifted; any
     nonexistent/duplicated local hour is counted and logged.

## 4. Time boundaries (chronological, nonoverlapping, by target hour, Europe/Paris)

| block | target hours | days | origins |
|---|---|---|---|
| train | first origin with 168 h history → **2010-09-01 00:00** (exclusive) | ≈ 1 350 | daily, 00:00 |
| validation | **2010-09-01 00:00 → 2010-10-15 00:00** | 44 | 44 |
| locked test | **2010-10-15 00:00 → 2010-11-26 00:00** | 42 consecutive | 42 |

- Validation ≥ 14 days ✔, test ≥ 21 consecutive days ✔, test ≥ 21 daily issue times ✔.
- The test window deliberately contains the **2010-10-31 DST change** (25 hour local day).
- Test origin *inputs* come from the validation period (they are the past at issue time);
  test *targets* never overlap validation or train targets.
- No random splits anywhere. No test timestamp is used for tuning, thresholds, interval
  calibration or monitor settings.

## 5. Missing data and cleaning (no hidden cleanup)

- Raw → hourly: minute rows with `?` are missing. Duplicated minute timestamps are counted;
  identical duplicates collapse, conflicting duplicates are counted and the hour is set
  missing. Values outside the physical range (negative, or > 15 kW for a 230 V domestic
  supply) are counted and set missing. Outliers inside the physical range are **flagged
  and counted, not removed** (appliance spikes are real load).
- Hours with < 45 present minutes → missing. Count logged.
- **Causal fill for inputs only:** missing hour `t` is filled with the (already filled)
  value at `t − 168 h`; if that does not exist, with the **train only** median of the
  hour of week profile. Only past values and train fitted constants are used. Every
  filled point is flagged and counted per split in `results/missing_policy_log.json`.
- **Imputed values are never targets.** Missing target hours are dropped from scoring for
  every model identically.
- **Origin eligibility** (same for all models): an origin is scored only if its last
  168 h contain ≤ 25 % missing hours and its last 24 h are not entirely missing.

## 6. Baselines (scored on exactly the same origins and target hours as the model)

| name | prediction for target hour τ |
|---|---|
| last value | `y(issue − 1h)` for all 24 horizons |
| seasonal naive day | `y(τ − 24 h)` |
| seasonal naive week | `y(τ − 168 h)` |
| (reference rule, not eligible as primary) 7 day same hour mean | `mean_k y(τ − 24k h), k = 1..7` |

**Primary baseline** = the lowest **validation** MAE among the three naive baselines.
Locked in the frozen manifest before the test is scored. The 7 day mean rule is reported
alongside as an honesty check: if the learned model barely beats a rule, say so.

## 7. Models

1. **First contender (transparent):** direct multihorizon **ridge regression**: 24
   separate linear models, one per horizon, trained on daily 00:00 origins. Features
   (all ≤ cutoff): `lag_24, lag_48, lag_168` of the target hour, 7 day same hour mean,
   last value, mean/std/max of the last 24 h, mean of the last 168 h, share of imputed
   hours in the window, and target hour calendar (weekday one-hot, weekend, French public
   holiday, local hour sin/cos, day of year sin/cos). Features standardised on train only.
   Variants selected on **validation MAE**: raw vs `log1p` target, alpha ∈ {0.1, 1, 10, 100}.
2. **Optional stronger model:** `HistGradientBoostingRegressor` (absolute error loss), per
   horizon. Adopted **only if** it improves validation MAE by ≥ 3 % over the best ridge,
   and its fit time and loss of transparency are reported. Otherwise ridge is frozen.
3. **Feature ablation (validation + walk forward):** lags only vs lags+calendar vs full.
4. **What can break it:** holidays/vacancy (August absence), regime change (new occupant
   behaviour or appliances), DST days (absolute hour lags misalign by 1 h for one day),
   sustained data gaps, stuck meter, unit changes upstream.

## 8. Uncertainty

- **80 % interval** from **validation residuals only**, split conformal style:
  residuals `r = log1p(y) − log1p(ŷ)`, interval `expm1(log1p(ŷ) + [q₁₀, q₉₀])`, clipped at
  0. Multiplicative in kWh, so it widens at high load where household errors are larger.
- **Reported on test:** empirical coverage (target ≥ 70 %, nominal 80 %), median width
  (kWh), coverage by horizon and hour, where it fails (direction and magnitude of misses).
  A trivially wide interval is not success: width is reported next to coverage, and
  compared with the width of the naive "train 10th–90th percentile of load" band.

## 9. Operator output

For a given issue timestamp the tool writes a JSON + CSV + PNG + HTML report with: the 24
hourly predictions (kWh), the 80 % interval, predicted peak hour(s) (top 3), per hour
advisory, the report level status, the data cutoff of every feature group, and the reasons
for any downgrade.

- `HIGH_DEMAND` threshold: **90th percentile of observed hourly energy over train +
  validation** (never test). The flag rule (which predictive quantile must exceed the
  threshold) is chosen on validation by F1 in a peak event study.
- `LOW`: the interval's upper bound is below the train+validation 25th percentile.
- `NORMAL`: otherwise.
- **The interface refuses to say NORMAL or LOW** (it says `UNCERTAIN`) whenever the
  report status is `UNCERTAIN`; it can still raise `HIGH_DEMAND`. It says nothing at all
  (`UNAVAILABLE`, nonzero exit) when the input cannot support a forecast.

| condition | status |
|---|---|
| missing `energy_kwh` column / other unit / unparsable or timezone naive timestamps / duplicate timestamps / timestamps that are not hourly or are misaligned / negative values | UNAVAILABLE |
| issue time not 00:00 Europe/Paris (outside model contract) | UNAVAILABLE |
| < 168 h of history before issue time | UNAVAILABLE |
| last observation older than 6 h at issue time (stale meter) | UNAVAILABLE |
| > 25 % of last 168 h missing | UNAVAILABLE |
| magnitude 20× off the training scale (suspected confusion of Wh and kWh) | UNAVAILABLE |
| last observation 1–6 h old, any missing hour in last 24 h, > 10 % missing in last 168 h | UNCERTAIN |
| flat line ≥ 6 identical consecutive hours in last 24 h (stuck meter) | UNCERTAIN |
| drift monitor alarm (recent errors far above validation) | UNCERTAIN + widened interval |
| rows timestamped after the issue time | ignored and logged (never used) |

## 10. Evaluation protocol

- Fixed split, frozen feature set, frozen model, frozen interval quantiles, frozen
  thresholds, all written to `results/frozen_manifest.json` with SHA-256 hashes of the
  model and data. **The test is scored once.** `final-eval` writes a lock; rerunning it is
  allowed only to regenerate identical numbers from the identical manifest hash, and it
  refuses if the manifest changed after the lock.
- Metrics in kWh: MAE, RMSE, improvement vs the locked primary baseline (bar: ≥ 5 % MAE),
  worst 24 h origin (reported pass or fail), error by hour of day, day type
  (weekday/weekend/holiday) and each of the 24 horizons separately.

## 11. Exceeds the bar items (all pretest unless stated)

| item | engineering risk it reduces |
|---|---|
| **Walk forward backtest** over monthly folds from 2008-12 to 2010-08 (expanding window retrain, 44 day rolling calibration), with per fold distribution shift stats (mean ratio, KS, PSI) vs the training window | a single 44 day validation window could be a lucky season; shows whether gains and coverage survive seasonal shift |
| **Peak event decision study**: precision/recall/F1 of HIGH_DEMAND flags at the validation chosen rule, plus "actual daily peak inside predicted top 3 hours" | a low MAE says nothing about whether the advisory catches the hours that matter |
| **Learned vs rules ablation** (rules; lags only; +calendar; full) on validation and walk forward | calendar features carry upkeep (holiday tables); prove they earn it, and prove learning beats a rule |
| **Adaptive drift / data quality monitor**: at each issue time, rescore the frozen model on the last 7 days (actuals now known); widen the interval (never narrow) and downgrade to UNCERTAIN when recent error is far above validation | a static interval silently overpromises after a regime change; the monitor cuts false confidence (reported as "confident label but actual outside interval / actually high") |

## 12. Deliverables and reproducibility

- Python 3.12, `uv.lock` locked dependencies, CLI `wattline`.
- `wattline download` is the **only** command that touches the network (SHA-256 checked).
  `wattline reproduce` regenerates everything offline from the raw snapshot.
- pytest: resampling/timezones/DST, no leakage corruption tests, forecast window
  alignment, evaluation math against hand computed values, invalid/stale input, one
  end to end pipeline test on synthetic raw data with network sockets disabled.
- Docs: README, DATA_CARD, MODEL_CARD, EVALUATION, PROCESS_LOG, RETROSPECTIVE.

## Amendments

### A1 (2026-10-20): timezone rule (before any model was fitted)

The decision rule in section 3 ("1440 rows on DST dates ⇒ fixed offset clock") turned out
to be wrong for this file. Evidence from ingest (no model, no validation/test metric involved):

- every DST date has exactly 1440 rows, **but** the intraday grid voltage profile shows a
  0 minute shift across all 8 transitions (a clock ignoring DST would show ~60 min; ±60 min
  alignments fit 2× worse). Voltage reflects the aggregate load of the local network, i.e.
  thousands of households on legal wall clock time, so it is a much cleaner clock witness than
  this one household's routine (which I tried first and found too noisy: shifts of −58…+20 min
  with no consistent sign);
- on all four spring forward dates the nonexistent 02:00–02:59 rows are exact copies of
  03:00–03:59 (59 of 60 rows identical in every column).

New rule: the logger follows **Europe/Paris wall clock** and the publisher regridded the file.
Nonexistent spring rows are dropped (fabricated duplicates); the single autumn 02:00–02:59 hour
cannot be attributed to either occurrence and is dropped, leaving both UTC hours missing.
480 rows dropped in total, all counted in `results/data/ingest_report.json`. Detection lives in
`wattline.ingest.detect_clock` and is unit tested.

### A2 (2026-10-20): monitor skill alarm (pretest evidence only; before freeze)

The walk forward backtest (folds before validation, test never loaded) exposed a failure mode the
plan had listed but not guarded: in **August 2010 the household was away** and the model, which
keeps predicting normal occupancy, was 74 % *worse* than seasonal naive day (MAE 0.351 vs 0.202 kWh).
The planned drift alarm (recent MAE > 1.5× validation MAE) never fired, because absolute errors on a
near empty house are small. Added, before freezing: a **skill alarm**: if over the last 7 issue times
the model's MAE exceeds the locked primary baseline's MAE on the same hours, the report is
downgraded to UNCERTAIN (`monitor.skill_alarm_ratio = 1.0`). It changes labels and status only,
never the point forecast or the interval. Adoption still follows the predeclared rule (pooled
false confidence must fall, width growth ≤ 25 %): false confidence 9.6 % static → 8.1 % adaptive,
width +6 %. I did not add a fallback to naive forecast: it would be a new point forecast policy
fitted to essentially two holiday months, which is too little evidence.
