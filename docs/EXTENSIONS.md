# Extension studies (pre registered)

Status: written before any result of these studies was computed. The v1.0.0 release stays the
single scored evaluation. Nothing here changes a frozen decision, the served tool or any number
in `docs/EVALUATION.md`; the locked test period is not rescored with any candidate.

All candidates are evaluated with `scripts/extension_study.py` on the 21 walk forward folds
before validation (2008-12 to 2010-08), with the same models, calibration windows and monitor
as `wattline backtest`. The series is truncated at the validation end before anything runs.
Results go to `results/extensions/` and are added to this file as a results section.

## Why these three

The self-criticism in `docs/EVALUATION.md` section 8 names three weaknesses a reviewer can
press. Each study tests a concrete fix and reports honestly whether it works.

## A. Per hour block interval calibration

Problem: one pooled residual quantile calibrates the average, not each hour. On the test,
night coverage was 0.93 to 1.00 and late morning and afternoon 0.55 to 0.71.

Candidate: fit the log residual quantiles separately for four blocks of local target hour,
using the same calibration window as before (the previous 44 days).

| block | local target hours |
|---|---|
| night | 00:00 to 05:59 |
| morning | 06:00 to 09:59 |
| midday | 10:00 to 16:59 |
| evening | 17:00 to 23:59 |

Compared variants: static pooled (current), static block, adaptive pooled (current served),
adaptive block (block quantiles with the same monitor stretch).

Success criteria, fixed in advance:
1. Pooled coverage stays at or above 0.78 and at or below 0.82 for the static block variant.
2. The worst block coverage gap from 0.80 shrinks against static pooled.
3. The mean interval score (80 percent Winkler score) does not get worse, so any gain is not
   bought by simply widening the interval.
4. Median width grows by at most 10 percent.

If these hold it is reported as a recommended change for a later release. If not, it is
reported as a negative result.

## D1. Hysteresis for the drift and skill alarm

Problem: the alarm flickers around its limit (on, off, on within days).

Candidate: keep the same on limits (recent MAE above 1.5 times validation MAE, or above 1.0
times the recent naive MAE) and switch the alarm off only when recent MAE is below 1.3 times
validation MAE and below 0.85 times the naive MAE. The state starts off at each fold.

Success criteria: fewer alarm switches per fold, and false confidence (an hour called NORMAL
or LOW that ended above the upper bound or above the HIGH threshold) does not rise by more
than 0.5 percentage points. The cost is more alarm days, and that is reported.

## D2. Operator impact simulation (peak staggering)

This is the fifth item of the brief's exceeds list. It is advisory and simulated only: no
control action, no dispatch advice.

Assumptions, fixed in advance:
- A site has one flexible load of 1.0 kWh per hour for 2 consecutive hours.
- Default start 19:00 local time (the usual evening habit).
- At issue time the operator may instead choose any start from 10:00 to 21:00.
- The policy picks the start that minimises the forecast daily peak, using the forecast made
  at issue time. A second policy does the same with the primary baseline forecast.
- An oracle that chooses with hindsight gives the upper bound.
- Only full days with 24 observed local hours are used (DST days are skipped).
- Outcome: the realised daily peak (maximum hourly actual plus the flexible load) per day.

Reported: mean daily peak under the default, the model policy, the baseline policy and the
oracle; the share of the oracle gain captured; days improved and days made worse; days at or
above the HIGH threshold.

The same simulation is later run once, unchanged, on the saved locked test predictions as a
descriptive check. It is not tuned on them.

## What is not claimed

- No result here changes the v1.0.0 scores.
- The household data is one French home. The simulation shows a method, not savings for any
  UAE site, and its assumptions are illustrative.
