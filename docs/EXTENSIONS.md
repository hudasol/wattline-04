# Extension studies (pre registered)

Status: each study below was written down before its result was computed. The v1.0.0
release stays the single scored evaluation. Nothing here changes a frozen decision, the
served tool or any number in `docs/EVALUATION.md`; the locked test period is not rescored
with any candidate.

Studies A, D1 and D2 are evaluated with `scripts/extension_study.py` on the 21 walk forward
folds before validation (2008-12 to 2010-08), with the same models, calibration windows and
monitor as `wattline backtest`. The series is truncated at the validation end before anything
runs. Results go to `results/extensions/` and are added to this file as a results section.

## Why these studies

The self-criticism in `docs/EVALUATION.md` section 8 and the exceeds list of the brief name
four things a reviewer can press: uneven interval coverage, a flickering alarm, no operator
impact simulation and an untested weather question. Each study tests a concrete fix and
reports honestly whether it works.

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

## C. Weather from the past only (learned vs rules, weather half)

Question: is temperature worth its leakage risk and maintenance cost? The brief forbids
observed weather in the forecast hours. This study uses only temperature from hours before
the issue time, which a site could really have at issue time. It never uses weather in a
target hour and never a weather forecast.

Data: hourly 2 m temperature for Sceaux (latitude 48.78, longitude 2.29) from the free
Open-Meteo historical archive (ERA5 based reanalysis), requested only up to the validation
end (2010-10-15) so the test period is never touched. Licence and attribution: Open-Meteo data
is CC BY 4.0; the underlying ERA5 data is Copernicus licensed. The response hash is recorded
with the results. A reanalysis value stands in for a station reading, so the result is an
estimate of the value of a real temperature feed, not a guarantee.

Features, all read from hours strictly before the issue time:
- `t_last`: temperature in the hour before the issue time.
- `t_mean24`, `t_min24`: mean and minimum of the previous 24 hours.
- `hdd24`: mean over the previous 24 hours of max(0, 15 minus temperature).
- `t_lag24`: temperature 24 hours before the target hour (one value per horizon).

Compared on the same folds and the same ridge settings as the chosen model:
`lags_calendar` (chosen) versus `lags_calendar` plus the five temperature features, on the
frozen validation split and on the 21 walk forward folds.

Success criteria, fixed in advance: temperature is called worth its cost only if pooled
backtest MAE improves by at least 2 percent, it wins in at least 12 of 21 folds, and
validation MAE is not worse. Otherwise the answer is that it is not worth it for this site.

## Results

All figures come from `results/extensions/`. Studies A, D1 and D2 were run by
`scripts/extension_study.py` on the 21 walk forward folds; study C by
`scripts/extension_weather.py`; the descriptive test check by
`scripts/extension_test_staggering.py`.

### A. Per hour block interval calibration: criteria met

| variant | coverage | median width | mean interval score | worst block gap |
|---|---|---|---|---|
| static pooled (current) | 0.802 | 1.353 | 2.060 | 0.114 |
| static block | 0.807 | 1.446 | 2.041 | 0.017 |
| adaptive pooled (served) | 0.826 | 1.434 | 2.052 | 0.126 |
| adaptive block | 0.832 | 1.517 | 2.044 | 0.044 |

For the static block variant: coverage 0.807 is inside 0.78 to 0.82 (criterion 1 met); the
worst block gap falls from 0.114 to 0.017 (criterion 2 met); the interval score improves from
2.060 to 2.041 (criterion 3 met); median width grows by 6.9 percent (criterion 4 met). The
night block, which was over covered at 0.914, comes down to 0.796, and the evening block,
which was under covered at 0.729, rises to 0.817. The fix is therefore a recommended change
for a later release. It is not applied to v1.0.0.

### D1. Hysteresis for the alarm: modest gain

Alarm switches across the 21 folds fall from 13 to 11 and false confidence moves from 0.0810
to 0.0802, so the second criterion holds. The cost is more alarm days (58 to 69). The gain is
real but small, and it is reported as modest.

### D2. Operator impact simulation: positive in the backtest, mixed on the test

Backtest (601 full days): the mean daily peak falls from 3.184 kWh under the default start to
2.841 kWh under the model policy, a reduction of 0.343 kWh that captures 71.2 percent of the
oracle gain. The baseline policy reduces it by 0.293 kWh (60.8 percent). The model policy
improved 390 days and made 103 worse.

Locked test, descriptive only (41 full days, 1 skipped, nothing tuned): the model policy
reduces the mean daily peak by 0.144 kWh (32.6 percent of the oracle gain; 22 days improved,
12 worse), while the seasonal naive policy reduces it by 0.232 kWh (52.5 percent; 23 days
improved, 8 worse). On the test the simple baseline chose better start times than the model.
This is reported as it came out, on a small sample of days, and it is a reason not to claim
that the model improves the operator decision.

### C. Past only temperature: not worth its cost for this site

On the frozen validation split, temperature improves MAE from 0.3732 to 0.3703 kWh (0.8
percent). On the 21 walk forward folds the pooled MAE worsens from 0.4445 to 0.4461 kWh
(minus 0.4 percent) and temperature wins in 11 folds. The pre registered bar (at least 2
percent pooled, at least 12 folds, validation not worse) is not met. By season the change is
minus 1.75 percent in winter, plus 0.90 percent in summer, and close to zero in spring and
autumn. The answer is that, for this household, a temperature feed is not worth its leakage
risk and maintenance cost. Open-Meteo response hash:
`d4a653ac712ca3311e5570663fc18a9258a625b18f87df5e9893e7aeecd19acc`.

## What is not claimed

- No result here changes the v1.0.0 scores.
- The household data is one French home. The simulation shows a method, not savings for any
  UAE site, and its assumptions are illustrative.
- A temperature feature in production needs a real, timely data feed; its cost and failure
  modes (stale or missing weather data) are not part of this study.
