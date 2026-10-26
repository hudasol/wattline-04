# Evaluation

Every number here is read from generated files: `results/validation/selection.{json,md}`,
`results/backtest/backtest.{json,md}`, `results/final/metrics.json`,
`results/final/report.md` and the CSV tables next to them. Regenerate with `wattline reproduce`.

## 1. Protocol

| item | value |
|---|---|
| issue time | 00:00 Europe/Paris daily; every feature ends at the issue time |
| horizons | h = 1…24, target hour starts at issue + (h−1) h, each horizon scored separately |
| train | target hours before 2010-09-01 00:00 local (1,347 origins, 1,316 eligible) |
| validation | 2010-09-01 → 2010-10-15 (44 origins, 36 eligible; an 87 hour gap removes 8) |
| locked test | 2010-10-15 → 2010-11-26 (42 consecutive origins, all eligible, 1,006 scored hours; 2 DST fold hours missing) |
| eligibility | ≤ 25 % of the previous 168 h missing; applied identically to every model and baseline |
| scoring mask | identical for model and all baselines: eligible origin × observed target hour |
| test blindness | `select`, `backtest`, `freeze` truncate the series at 2010-10-15 before doing anything |
| one scoring | `final-eval` writes `results/final/final_eval_lock.json`; reruns only regenerate identical numbers and log `REGENERATION`; changed decisions → refused |

**Boundary audit** (computed inside `final-eval`, `metrics.json → boundary_audit`): train targets
end before 2010-09-01 00:00, validation targets end before 2010-10-15 00:00, the first test target
is 2010-10-15 00:00, and for all 42 test origins every feature's latest input hour ends at or
before the issue time. The frozen manifest and pretest results were committed (`680cdc8`,
refrozen with identical decisions in `dfe0e8f`) before the first test scoring
(`results/final_run.log`, 2026-10-20 06:50 UTC).

**Serving equivalence:** `final-eval` also runs the operator tool (`wattline forecast` code path)
for every test issue time and checks it reproduces the evaluated forecast and interval
(max difference 5·10⁻⁵ kWh = JSON rounding). The evaluated system *is* the served system.

## 2. Baselines and primary baseline choice (validation only)

| baseline | val MAE | val RMSE |
|---|---|---|
| last value | 0.641 | 0.854 |
| **seasonal naive day** | **0.507** | 0.702 |
| seasonal naive week | 0.533 | 0.746 |
| 7 day same hour mean (reference rule, not eligible as primary) | 0.398 | 0.539 |

Primary baseline locked: **seasonal naive day**. On the test it was also the best naive
baseline, so the choice did not flatter the model.

## 3. Model selection (validation only)

| feature set | features | best val MAE |
|---|---|---|
| lags only | 5 | 0.3876 |
| **lags + calendar** | 18 | **0.3732** |
| full (+ rolling stats, imputation share) | 24 | 0.3765 |
| boosted trees on lags + calendar | 18 | 0.3728 (+0.1 %, rejected: bar 3 %, 200× fit time) |

Chosen: ridge, lags + calendar, raw target, alpha 100: val MAE 0.373 (+26.3 % vs primary).
log1p targets were slightly worse on MAE (0.379) despite the skewed load.

## 4. Locked test: accuracy

| | MAE kWh | RMSE kWh | vs primary |
|---|---|---|---|
| **model** | **0.494** | **0.695** | **+22.7 %** |
| seasonal naive day (primary) | 0.639 | 0.948 | n/a |
| seasonal naive week | 0.723 | 1.025 | −13.1 % |
| last value | 0.826 | 1.090 | −29.1 % |
| 7 day same hour mean rule | 0.544 | 0.755 | +14.9 % (model is +9.2 % better than it) |

**Bar (≥ 5 % MAE vs locked primary): met (+22.7 %).** The model beats the primary baseline on
36 of 42 days and the 7 day rule on 29 of 42.

### Worst and best days

| | issued | status | model MAE | primary | 7 day rule | 80 % cov |
|---|---|---|---|---|---|---|
| **worst** | Mon 2010-10-18 | OK | **0.979** | 1.250 | 0.887 | 0.54 |
| 2nd | Thu 2010-11-04 | UNCERTAIN | 0.918 | 0.946 | 1.135 | 0.75 |
| 3rd | Sun 2010-11-21 | OK | 0.884 | 0.980 | 0.659 | 0.46 |
| 4th | Sun 2010-10-31 (DST) | UNCERTAIN | 0.826 | 1.225 | 0.805 | 0.50 |
| 5th | Sat 2010-10-16 | OK | 0.723 | 0.459 | 0.627 | 0.50 |
| best | Sun 2010-11-07 | UNCERTAIN | 0.252 | 0.410 | 0.456 | 1.00 |

The worst day (`results/final/figures/worst_day.png`) is a Monday with sustained 2.5–3.8 kWh from
07:00 to 13:00 and again 18:00–22:00: a laundry, cooking or heating day unlike the preceding week.
The forecast (a smooth conditional mean) stayed between 0.8 and 2.0 kWh; 11 of 24 hours fell
outside the interval and the report still said OK; the monitor had no way to know in advance.
On 2010-10-16 the model lost to seasonal naive day: a quiet Saturday that repeated Friday.

### By horizon (identical origins for every column)

| h | n | model | snaive_day | snaive_week | last_value | mean7_rule | cov80 | width |
|---|---|---|---|---|---|---|---|---|
| 1 | 42 | 0.248 | 0.344 | 0.489 | 0.333 | 0.388 | 0.929 | 1.166 |
| 2 | 42 | 0.174 | 0.244 | 0.362 | 0.344 | 0.284 | 1.000 | 1.073 |
| 3 | 41 | 0.218 | 0.369 | 0.352 | 0.384 | 0.278 | 0.927 | 1.025 |
| 4 | 41 | 0.163 | 0.237 | 0.202 | 0.447 | 0.213 | 0.927 | 1.009 |
| 5 | 42 | 0.152 | 0.211 | 0.200 | 0.465 | 0.171 | 0.976 | 1.039 |
| 6 | 42 | 0.179 | 0.214 | 0.232 | 0.529 | 0.201 | 0.976 | 1.014 |
| 7 | 42 | 0.313 | 0.376 | 0.453 | 0.587 | 0.341 | 0.929 | 1.318 |
| 8 | 42 | 0.634 | 0.809 | 0.920 | 1.525 | 0.718 | 0.810 | 2.095 |
| 9 | 42 | 0.506 | 0.769 | 0.718 | 1.141 | 0.565 | 0.833 | 1.894 |
| 10 | 42 | 0.313 | 0.515 | 0.579 | 0.853 | 0.391 | 0.857 | 1.718 |
| 11 | 42 | 0.601 | 0.886 | 0.966 | 0.969 | 0.620 | 0.667 | 1.651 |
| 12 | 42 | 0.726 | 0.978 | 1.000 | 1.137 | 0.780 | 0.690 | 1.719 |
| 13 | 42 | 0.654 | 0.824 | 0.905 | 0.946 | 0.642 | 0.714 | 1.682 |
| 14 | 42 | 0.759 | 0.919 | 0.900 | 0.838 | 0.721 | 0.619 | 1.645 |
| 15 | 42 | 0.670 | 0.726 | 0.892 | 0.739 | 0.720 | 0.548 | 1.558 |
| 16 | 42 | 0.646 | 0.722 | 0.831 | 0.753 | 0.716 | 0.667 | 1.506 |
| 17 | 42 | 0.526 | 0.705 | 0.750 | 0.704 | 0.644 | 0.762 | 1.490 |
| 18 | 42 | 0.579 | 0.682 | 0.904 | 0.778 | 0.652 | 0.738 | 1.587 |
| 19 | 42 | 0.712 | 1.014 | 1.074 | 1.204 | 0.766 | 0.667 | 1.908 |
| 20 | 42 | 0.692 | 0.995 | 1.047 | 1.335 | 0.749 | 0.762 | 2.151 |
| 21 | 42 | 0.691 | 0.843 | 1.091 | 1.331 | 0.732 | 0.762 | 2.154 |
| 22 | 42 | 0.769 | 0.802 | 1.210 | 1.239 | 0.801 | 0.690 | 2.035 |
| 23 | 42 | 0.483 | 0.638 | 0.683 | 0.706 | 0.504 | 0.833 | 1.688 |
| 24 | 42 | 0.439 | 0.507 | 0.574 | 0.507 | 0.447 | 0.786 | 1.388 |

Because issue time is midnight, horizon ≈ local hour + 1. Error is not driven by distance from
the issue time but by time of day: quiet nights are easy, late mornings and evenings are hard.
The model beats seasonal naive day at every horizon; it is slightly worse than the 7 day rule at
h = 13–14 (early afternoon).

### By day type

| day type | hours | model MAE | primary MAE | 7 day rule MAE |
|---|---|---|---|---|
| weekday | 672 | 0.449 | 0.616 | 0.498 |
| weekend | 286 | 0.593 | 0.697 | 0.649 |
| holiday (1 & 11 Nov) | 48 | 0.541 | 0.630 | 0.562 |

Per local hour of day: `results/final/per_hour_of_day.csv` and `figures/mae_by_hour.png`.

### DST transition day

2010-10-31 (25 hour local day): 24 absolute hour targets 00:00 CEST → 22:00 CET, with the two
02:00 hours missing (the ingest dropped the ambiguous hour rather than guess). MAE 0.83 kWh on 22
scored hours, fourth worst day. Part of the error is structural: absolute hour lags compare
evening hours with a local hour that is one hour off.

## 5. Locked test: 80 % interval

| | coverage | median width kWh |
|---|---|---|
| **as served (adaptive, monitor adopted pretest)** | **0.794** | **1.605** |
| static validation interval (reference) | 0.734 | 1.374 |
| climatology band 0.27–2.34 kWh (train+val 10–90 %) | 0.889 | 2.071 |

The served coverage of 0.794 comes from the adaptive interval, which widens during the test
period using recent actuals that are available at each issue time (never actuals from the
forecast window itself). The strict, validation only static interval covered 0.734; that is the
conservative figure, and it also clears the bar.

**Bar (coverage ≥ 0.70): met.** The interval is 23 % narrower than a climatology band and
date specific, so it is not a trivially wide success. It is still wide, however: 1.6 kWh around a
median hourly load of 0.8 kWh.

Successes and failures:
- **Conditional coverage is poor:** nights 0.93–1.00 (too wide), late morning and afternoon
  0.55–0.71 (too narrow). Pooling residuals across horizons calibrates the average, not each hour.
- **Misses are asymmetric:** 13.4 % of hours fell *below* the lower bound and 7.2 % above the upper
  bound (mean excess 0.60 kWh). Above interval misses are the dangerous ones (unexpected peaks).
- **Validation was calmer than the test:** September 2010 errors were smaller than Oct–Nov, so the
  static interval covered too little. The monitor saw recent errors at 1.4–1.6× validation from
  2010-10-19 and stretched the interval by up to 1.53×, raising coverage by 6 points.

## 6. Locked test: advisory (peak event decision study)

HIGH threshold 2.344 kWh (90th percentile of train+validation hours); flag when the predicted
80 % quantile reaches it (rule chosen on validation by F1 from {0.5 … 0.9}).

| | precision | recall | F1 |
|---|---|---|---|
| validation (rule chosen here) | 0.345 | 0.513 | 0.412 |
| backtest pooled (21 folds before validation) | 0.352 | 0.512 | 0.417 |
| **test, as served** | **0.307** | **0.514** | **0.385** |
| test, flags from the primary baseline (`y(τ−24h)` ≥ threshold) | 0.270 | 0.280 | 0.275 |

Base rate on test 10.6 %. Recall held at ~0.51 across validation, backtest and test; precision
dropped on test. In plain words: the tool catches about half of the high demand hours and about
one flag in three is a real high hour. Daily peak hour inside the predicted top 3 hours: model
41 %, primary baseline 34 %, 7 day rule 37 %.

Statuses on the 42 test days: 32 OK, 10 UNCERTAIN (all drift alarms). Hour labels: 629 NORMAL,
179 HIGH_DEMAND, 200 UNCERTAIN, 0 LOW. **False confidence** (an hour called NORMAL/LOW that ended
above the interval or above the HIGH threshold): 6.3 % of hours as served vs 9.3 % with the
static interval and no monitor.

## 7. Exceeds the bar items and the risk each reduces

### 7.1 Walk forward backtest with distribution shift comparison
*Risk reduced: a 36 origin validation window could be a lucky or unrepresentative season.*

21 monthly folds (2008-12 → 2010-08), expanding window refit, 44 day rolling calibration,
thresholds from data before each fold (`results/backtest/`, `backtest_folds.png`).

| season | folds | model MAE | primary MAE | improvement | cov static | cov adaptive | mean KS vs train |
|---|---|---|---|---|---|---|---|
| DJF | 6 | 0.537 | 0.753 | +28.4 % | 0.805 | 0.828 | 0.17 |
| MAM | 6 | 0.431 | 0.584 | +26.2 % | 0.812 | 0.830 | 0.20 |
| JJA | 6 | 0.370 | 0.416 | **+4.2 %** | 0.784 | 0.821 | 0.22 |
| SON | 3 | 0.432 | 0.629 | +31.1 % | 0.806 | 0.823 | 0.13 |

Pooled: +25.0 % vs primary, +5.9 % vs the 7 day rule; model better than the primary in 20/21
folds and by ≥ 5 % in 19. **Summer is the weak season and August 2010 the outright failure**
(−74 %: model 0.351 vs naive 0.202 kWh): the household was away and yesterday's flat load beat a
model that keeps predicting occupancy. July 2010 static coverage fell to 0.66 (adaptive: 0.78).
Across folds, larger KS shift went with *lower* MAE (Spearman −0.36: summer is shifted *and*
low load) and smaller improvement (−0.40); KS did not predict coverage error (−0.09), so a shift
statistic alone would not have warned about interval failure, which is why the monitor watches
recent errors instead. Test vs validation shift: KS 0.075, PSI 0.08, mean 1.17 vs 1.05 kWh.

### 7.2 Peak event decision study
*Risk reduced: a good MAE says nothing about catching the hours that matter, and a flag can look acceptable on average while failing in the hours an operator cares about.*

Section 6 gives the headline: the flag rule was chosen on validation, its recall was stable across validation, backtest and test, and it beats flags derived from the primary baseline (F1 0.38 vs 0.28). `scripts/peak_study.py` adds a descriptive breakdown from the saved test predictions (`results/final/peak_study.md`). It fits nothing and changes no decision.

| time of day | real high hours | flagged | caught | precision | recall |
|---|---|---|---|---|---|
| night 23:00 to 05:59 | 2 | 1 | 0 | 0.00 | 0.00 |
| morning 06:00 to 09:59 | 25 | 40 | 18 | 0.45 | 0.72 |
| midday 10:00 to 16:59 | 30 | 13 | 2 | 0.15 | 0.07 |
| evening 17:00 to 22:59 | 50 | 125 | 35 | 0.28 | 0.70 |
| all scored hours | 107 | 179 | 55 | 0.31 | 0.51 |

What the flag would and would not catch:
- **It works as a window warning, not an hour alarm.** In the morning and evening windows it catches about 70 % of real high hours, but only 28 to 45 % of the hours it flags are high.
- **Midday is a blind spot.** It caught 2 of 30 real high hours between 10:00 and 17:00. These hours depend on occupancy that yesterday's pattern does not show, so a missing flag at midday says nothing.
- **The largest peaks are the ones it misses.** It caught 44 of 76 real high hours between 2.34 and 3.00 kWh, but only 11 of the 31 hours above 3.00 kWh.
- **Many false alarms are near misses.** Of 124 false alarms, 42 reached 1.80 kWh or more (77 % of the threshold); the median was 1.57 kWh.
- **A day level flag is useless.** 41 of 42 test days had at least one flagged hour (33 had a real high hour), so only the hour and window carry information.

Operator reading: treat HIGH_DEMAND as "prepare for a possible peak in the morning or evening window", never as the only evidence, and never read its absence as a safe hour.

### 7.3 Learned vs rules ablation
*Risk reduced: paying upkeep for features (holiday tables) or for learning that a rule matches.*

| | validation MAE | backtest pooled MAE |
|---|---|---|
| seasonal naive day (rule) | 0.507 | 0.593 |
| 7 day same hour mean (rule) | 0.398 | 0.472 |
| ridge, lags only | 0.388 | 0.463 |
| **ridge, lags + calendar** | **0.373** | **0.445** |
| ridge, + rolling stats | 0.377 | 0.445 |

Learning on lags alone adds only ~2–3 % over the best rule; the calendar adds another ~4 %. The
calendar costs a hand maintained French holiday table (`calendar_fr.py`), which is worth it here. Rolling
statistics add nothing and were dropped. **Weather was deliberately not used:** no archived
*forecast* exists for this site and period, and observed weather in target hours would be leakage;
the honest upper bound of its value is therefore unknown, not zero.

### 7.4 Adaptive drift / data quality monitor
*Risk reduced: a static interval silently overpromising after a regime change.* Adoption was
decided on the backtest by a rule written before seeing its outcome (false confidence must fall,
width growth ≤ 25 %): 9.6 % → 8.1 % false confidence for +6 % width. The skill alarm (amendment
A2) was added after the backtest showed the August vacancy; it fired on 20 August 2010 issue
times. On the locked test the monitor raised coverage from 0.734 (the conservative, validation
only static interval) to 0.794 (the served adaptive interval, widened using recent actuals known
at each issue time) and cut false
confidence from 9.3 % to 6.3 % of hours. Data quality checks (stale, gaps, stuck meter, unit
scale) make the same "no reassurance on bad evidence" promise at the input side.

## 8. Self-criticism

- **The worst day was labelled OK.** Nothing in the issue time information predicted it; the
  monitor reacts to recent errors, so it is always one day late.
- **Interval conditional coverage is poor** (night vs afternoon). Next step: conformal quantiles per
  block of hours, calibrated on more than 36 days.
- **The drift alarm flickers** around 1.5× (on 10-20, off 10-21…10-29, on 10-30). Hysteresis
  (e.g. on at 1.5, off below 1.3) would be a calmer operator experience.
- **LOW is never issued.** With 1.6-kWh-wide intervals the "confidently low" label cannot fire;
  it should be redesigned or removed rather than shown as a feature.
- **Small validation set:** 36 eligible origins chose the feature set, alpha, the flag level and
  the interval. The backtest is the evidence that these choices generalise, but the alpha choice
  sits on the grid edge.
- **Absolute hour lags on DST days** cost accuracy on 2 days a year; local clock lags would fix it
  at the price of handling the repeated/missing local hour in features.
