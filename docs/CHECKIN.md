# Midpoint check-in (draft to paste into the mentorship group chat)

> wattline check-in
>
> **Dataset:** UCI Individual Household Electric Power Consumption (Sceaux, FR), CC BY 4.0
> (checked on the UCI page), DOI 10.24432/C58K54. Raw zip SHA-256 `9f84b46a…`, downloaded by a
> script, kept read-only.
>
> **Raw coverage:** 2,075,259 one-minute rows, 2006-12-16 → 2010-11-26; 25,979 rows `?` (1.25 %);
> 0 duplicate minutes; 0 impossible values. Units verified at source (active ≤ apparent power on
> 94 % of minutes, sub-meters ≤ total on 99.998 %).
>
> **Cleaned gaps:** 34,587 hourly rows, 444 missing (< 45 valid minutes) in 18 runs, longest
> 121 h (Aug 2010). Timezone finding: the logger is Paris wall clock and the publisher re-gridded
> DST days — the spring 02:00 hour is a copy of 03:00 (proved with grid-voltage profiles); I drop
> those and the ambiguous autumn hour (480 rows) instead of shifting anything.
>
> **Frozen split (Paris time, by target hour):** train → 2010-09-01, validation 09-01 → 10-15
> (44 d), locked test 10-15 → 11-26 (42 d, includes the 31 Oct DST change). Issue time 00:00, cutoff
> = issue time.
>
> **Naive baselines on validation (MAE kWh):** last-value 0.641, seasonal-naive-day **0.507**
> (primary), seasonal-naive-week 0.533; reference 7-day same-hour mean 0.398.
>
> **First trained model:** direct ridge (24 horizons, lags + calendar), validation MAE **0.373**
> (+26 % vs primary). Boosted trees 0.373 too (+0.1 %) — not worth it.
>
> **One leakage risk:** the monitor re-scores yesterday's forecast at issue time — on the spring DST
> day the 24th hour of yesterday's window ends *after* today's issue time, so it is masked (there
> is a test for it). Same family: never fit the imputation profile or thresholds on test hours.
>
> **Ugly forecast plot:** attached `worst_day.png` — a busy Monday the smooth forecast missed
> (MAE 0.98, 11 of 24 hours outside the interval).
>
> **Remaining days:** CI on Linux + push + tag, record the demo, tighten the interval per hour
> block only if it can be validated without touching test, write-up polish.

*Note: in this build the locked test was scored before the check-in date; if the check-in must
precede test scoring, use the validation stage numbers above and send `results/validation/selection.md`
plus the backtest `August 2010` fold as the ugly plot instead.*
