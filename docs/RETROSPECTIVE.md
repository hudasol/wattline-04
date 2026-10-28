# Retrospective

## Against the bar

| requirement | evidence | status |
|---|---|---|
| ≥ 120 days hourly data | 34,587 hours (≈ 47 months) | met |
| chronological split, ≥ 14 d validation, ≥ 21 d locked test | 44 d validation, 42 d test, 42 daily issue times | met |
| both naive baselines on identical origins, primary chosen on validation | seasonal naive day locked in `selection.json` before the test was loaded | met |
| ≥ 5 % test MAE gain vs primary; MAE, RMSE, worst origin | +22.7 %; 0.494 / 0.695 kWh; worst 2010-10-18 (0.98 kWh) | met |
| 24 horizon specific predictions, no unknown future inputs | per horizon table; boundary audit + corruption tests | met |
| empirical 80 % coverage ≥ 70 %, median width reported | 0.794, width 1.60 kWh (vs climatology band 2.07) | met |
| operator command without editing feature files; UNAVAILABLE/UNCERTAIN on bad input | `wattline forecast`, 9 demo failure inputs, exit codes 0/3/2 | met |
| tests, cards, plan, saved predictions, one frozen final evaluation | 57 tests; DATA_CARD, MODEL_CARD; lock + `final_run.log` | met |
| exceeds: ≥ 2 items with the risk each reduces | walk forward + shift, peak event study, learned versus rules ablation, adaptive monitor, operator impact simulation | 5 of 5 (the fifth is reported honestly as mixed) |

Status: the repository is pushed, CI is green on every run, the `v1.0.0` tag and release are
published, and the mid point check-in was sent. The demo video is recorded separately and its
link is added to the README when it is published. The script is `docs/DEMO_SCRIPT.md`.

## What went well

- **Writing tests before the first real validation run.** The DST test caught pandas 3's
  microsecond default before it could corrupt the monitor's time arithmetic.
- **Structural test blindness.** Truncating the series at the validation end before any pretest
  stage runs is simpler and stronger than remembering not to look.
- **Using voltage as a clock witness.** It turned an ambiguous timezone question into a clear
  physical answer and exposed the publisher's fabricated spring hour.
- **Serving equivalence check.** Running the operator code path for every test issue time removed
  any doubt that the evaluated system differs from the one a user runs.
- **The monitor paid off on the real test:** validation turned out calmer than autumn, and the
  adaptive interval recovered 6 points of coverage.
- **Pre registering the extension studies.** Writing the success criteria in `docs/EXTENSIONS.md`
  before computing anything meant that a negative result could be reported as a result and not
  explained away.

## What went badly or is still weak

- My first timezone rule in the plan was wrong; it needed an amendment. A better plan would have
  named the evidence (voltage) rather than a shortcut based on row counts.
- Two portability bugs (CRLF in a hashed file, cp1252 default) were found late. A CI run on Linux
  from day one would have caught them earlier.
- The interval is calibrated on average, not by hour: intervals at night are too wide and in the
  afternoon too narrow. The fix exists (study A) but is not part of the frozen release.
- The worst day was labelled OK. A system limited to information at issue time cannot see a single
  unusually busy Monday coming; I can only report it.
- `LOW` never fires. I kept it because the plan specified it, but in its current form it is dead UI.
- The drift alarm flickers around its threshold, and the hysteresis fix only helps modestly.
- The operator simulation is weak on the locked test: the seasonal naive policy chose better start
  times than the model on 41 days. Positive backtest results did not carry over, and the sample is
  small, so I do not claim that the model improves the operator decision.
- My own first plan was to treat all four extension ideas as improvements. Two of them were not.

## What the extension studies taught

`docs/EXTENSIONS.md` holds the pre registered criteria and the results. They run on the pre test
folds, with one descriptive pass over the saved test forecasts, and they do not change v1.0.0.

- **Per hour block calibration works.** The worst block coverage gap fell from 0.114 to 0.017 and
  the interval score improved, for 6.9 % wider intervals. This is the clearest next release item.
- **Hysteresis is a small gain.** Alarm switches fell from 13 to 11 at the cost of more alarm days.
- **Past only temperature is not worth it for this household.** Pooled backtest error got 0.4 %
  worse and temperature won in 11 of 21 folds, so the feature, its data feed and its failure modes
  are not justified here. This answers the learned versus rules question about weather.
- **A backtest win is not a test win.** The staggering simulation looked good on 601 backtest days
  and mixed on 41 test days. A simple baseline can be a strong operator rule.

## What I would do next

1. Ship the per hour block intervals as a documented v1.1 candidate, with a fresh plan and one new
   frozen evaluation, because the single scoring of v1.0.0 cannot be reused.
2. Alarm hysteresis with a minimum alarm duration, tested on more than one season.
3. Local clock lags for DST days.
4. An occupancy or vacancy detector (several days near base load) that switches the point forecast
   to a naive rule, tested on more than two holidays, ideally on a dataset with many households.
5. Run the same pipeline on a commercial building (for example Building Data Genome 2) to see
   whether the conclusions survive a less noisy, more scheduled load.
6. Redesign or remove the `LOW` label.

## Time

Built in one long working session on 2026-10-20 (day 1 of the 10 day window), including two
repeated downloads of the dataset over a slow connection. The extension studies were added
afterwards, on 2026-10-26, as separate small commits.
