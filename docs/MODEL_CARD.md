# Forecast card — wattline v1.0.0

Frozen decisions SHA-256 `7ead74ad2613ff67def00e8bb06a0c6cd1a94b77be7ce9fcca5046e51400a8d0`
(`results/frozen/frozen_manifest.json`). Model file `results/frozen/model.json` (plain JSON,
hash-checked before every forecast).

## What it predicts

The energy (kWh) the household will use in each of the next 24 clock hours, issued at **00:00
Europe/Paris** from data ending at that instant. On DST days "24 hours" means 24 absolute hours
(the window ends at 22:00 local on the 25-hour autumn day, at 00:00 next day in spring).

## Model

- **Direct multi-horizon ridge regression:** 24 independent linear models, one per horizon,
  trained on 1,316 eligible daily origins (2006-12-24 → 2010-08-31; ≈1,310 samples per horizon
  after dropping missing targets). Standardisation statistics are from training only.
- **Features (18), all known at the issue time:** `lag_24`, `lag_48`, `lag_168` of the target
  hour, 7-day same-hour mean, last observed hour; target-hour calendar: local hour (sin/cos),
  weekday one-hot, weekend, French public holiday, day of year (sin/cos). Rolling statistics were
  tested (feature set `full`) and dropped because they did not help on validation.
- **Selection (validation only, test not loaded):** grid over feature set × raw/log1p target ×
  alpha ∈ {0.1, 1, 10, 100}. Chosen: `lags_calendar`, raw target, alpha 100 (val MAE 0.3732 kWh).
  Validation MAE is flat across alphas (third decimal); alpha 100 sits at the grid edge and I did not
  extend the grid to avoid tuning on a 36-origin window.
- **Boosted trees rejected:** `HistGradientBoostingRegressor` (absolute-error loss) reached val MAE
  0.3728 — a 0.1 % gain for **200× the fit time** (24.7 s vs 0.12 s) and no inspectable
  coefficients. The pre-declared bar was ≥ 3 %.
- **What the coefficients say** (standardised, mean |coef| over horizons): day-of-year cosine
  (winter vs summer level) 0.13, 7-day same-hour mean 0.11, last value 0.07 (dominant at h = 1:
  0.40), weekend / Sunday 0.06, `lag_24` 0.05, `lag_168` 0.04. At 07:00 (h = 8) the weekend
  terms are strongly negative — the late weekend morning.

## Uncertainty

- **80 % interval:** split-conformal quantiles of `log1p(y) − log1p(ŷ)` on validation residuals
  (n = 820), applied multiplicatively: `expm1(log1p(ŷ) + [q₁₀, q₉₀])`, clipped at 0. The 24
  residuals of one day are correlated, so the finite-sample guarantee is approximate.
- **Adaptive monitor (adopted on the pre-test backtest by a pre-declared rule):** at each issue time
  the frozen model is re-scored on the last 7 issue times. The interval is stretched by
  `clip(recent spread / validation spread, 1, 3)` — it never narrows. Two alarms downgrade the
  report to UNCERTAIN: **drift** (recent MAE > 1.5× validation MAE) and **skill** (recent MAE worse
  than the primary naive baseline on the same hours).

## Performance (locked test, 42 issue times, scored once)

| metric | value |
|---|---|
| MAE / RMSE | **0.494 / 0.695 kWh** |
| vs primary baseline seasonal-naive-day (0.639) | **+22.7 %** MAE |
| vs 7-day same-hour mean rule (0.544) | +9.2 % |
| worst day | Mon 2010-10-18, MAE 0.98 kWh |
| 80 % coverage / median width | **0.794 / 1.60 kWh** from the served adaptive interval, which widens using recent actuals available at each issue time; the strict, validation only static interval gives **0.734 / 1.37 kWh**, the conservative figure |
| misses | 7.2 % above the upper bound (mean excess 0.60 kWh), 13.4 % below the lower bound |
| HIGH_DEMAND flags | precision 0.31, recall 0.51, F1 0.38 (base rate 0.11) |
| daily peak in predicted top-3 hours | 41 % of days (primary baseline 34 %) |
| report statuses | 32 OK, 10 UNCERTAIN (all drift alarms, 2010-10-20 and 10-30 → 11-07) |

Error grows from 0.15–0.25 kWh overnight to 0.6–0.75 kWh in the late morning and evening. Weekend
MAE 0.59 kWh vs weekday 0.45. Walk-forward backtest over 21 pre-validation months: +25 % vs the
primary baseline pooled, positive in 20/21 folds — **negative (−74 %) in August 2010** (vacancy).

## Advisory thresholds (train + validation hours only)

- `HIGH_DEMAND`: predicted **80 % quantile ≥ 2.344 kWh** (the 90th percentile of observed hourly
  energy). The 80 % quantile was chosen over the median by F1 on validation: the median almost
  never reaches the threshold (recall 0.03).
- `LOW`: 80 % upper bound < 0.340 kWh (25th percentile). With intervals this wide, **LOW was never
  issued on the test** — the label exists but is effectively unreachable; NORMAL covers quiet hours.
- `UNCERTAIN`: replaces NORMAL/LOW whenever the report is UNCERTAIN.

## Failure behaviour

| input problem | result |
|---|---|
| no `energy_kwh` column (e.g. `energy_wh`), `unit` ≠ kWh, timezone-naive, duplicate, off-the-hour or negative timestamps/values | UNAVAILABLE, exit 2, nothing forecast |
| values 20× off the training scale (Wh mislabelled as kWh) | UNAVAILABLE |
| last reading > 6 h before issue / < 168 h history / > 25 % of last week missing | UNAVAILABLE |
| last reading 1–6 h old, any gap in last 24 h, > 10 % of last week missing, ≥ 6 identical readings | UNCERTAIN, exit 3 |
| drift or skill alarm | UNCERTAIN + wider interval |
| issue time not 00:00 Europe/Paris | UNAVAILABLE (outside the trained contract) |

## Limitations

- One household; no weather, occupancy, tariff or appliance information.
- **Vacancy and regime change:** the model assumes normal occupancy and is slow to notice a
  holiday; the skill alarm flags it after a few days but the point forecast does not adapt.
- **DST days:** lags are absolute hours, so the day after a change compares with a shifted local
  hour. The 2010-10-31 test day was the 4th-worst (MAE 0.83 kWh).
- **Peaks are under-predicted:** the forecast is a conditional mean; single-appliance spikes
  above 3 kWh are rarely predicted in time or size.
- **Validation was unrepresentative of autumn:** September 2010 was calmer than Oct–Nov, so the
  static interval under-covered (0.73). The monitor compensated, but the alarm flickers around its
  1.5× threshold (no hysteresis).
- Interval lower bounds are often 0 at night (log1p is nearly additive at low load).

## Environmental cost

Training the frozen model takes ~0.12 s on one laptop CPU core; the full validation grid,
backtest (21 folds × 4 fits) and final evaluation take ~2 minutes in total. No GPU, no cloud. The
rejected boosted-tree contender would have cost ~25 s per fit — small, but 200× for 0.1 %.

## What the forecast must never claim

- that the house will *not* exceed the HIGH threshold in an hour labelled NORMAL — 6.3 % of test
  hours were confidently labelled NORMAL yet ended above the interval or above the threshold;
- anything about other sites, future years with different occupants, or modern loads (EV, PV,
  heat pump);
- causal statements ("using X less will reduce Y");
- that a forecast is valid when the status is UNAVAILABLE or that UNCERTAIN hours are quiet;
- control authority: it is an advisory for a person, never a set-point for equipment.
