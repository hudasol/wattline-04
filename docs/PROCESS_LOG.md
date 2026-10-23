# Process log

Chronological, including the wrong turns. Commit hashes refer to this repository.

## 2026-10-20

**Brief and bar.** Read the brief and separated *meets* (≥ 120 days, ≥ 14 day validation, ≥ 21 day
locked test, both naive baselines on identical origins, primary chosen on validation, ≥ 5 % MAE gain,
≥ 70 % coverage, operator command with UNAVAILABLE/UNCERTAIN, tests, cards, one frozen evaluation)
from *exceeds* (at least two of: walk forward + shift, peak event study, learned versus rules ablation,
adaptive monitor, operator simulation). Decided to do four of the five and skip the peak staggering
simulation: its assumptions about which appliance moves would be invented, and I would rather
spend the time on the monitor, which directly addresses "never present stale values as good news".

**Dataset choice.** UCI household power consumption: CC BY 4.0, ~47 months of 1 minute data, real
gaps, explicitly suggested by the brief. Considered Building Data Genome 2 (closer to a "facility")
but its access path is heavier and I could not verify a clean single site story as quickly.

**Setup (`dce9486`).** Python 3.12 via the `py` launcher, `uv` for a lockfile. The UCI server is slow
and does not support HTTP range requests, so the first three downloads truncated; the downloader now
streams to `.part`, validates the zip and checks a SHA-256 before making the file read only.

**Plan first (`f7dff1a`).** Wrote `docs/PLAN.md` before any modelling code: target (kWh per clock
hour), issue time (00:00 Paris, cutoff = issue time), splits by calendar only, baselines, ridge first,
conformal intervals, status table, exceeds the bar items. At this point I had looked only at the dataset's
published description.

**Ingest and the timezone surprise (`6a283e2`, `1bede07`).** The plan's rule said "1440 rows on DST
dates ⇒ fixed offset clock". All 8 DST dates had 1440 rows, so I first concluded "fixed CET clock".
My behavioural check (does the household's morning ramp jump by 60 min at each transition?) did not
agree, but household routines were too noisy to be decisive (shifts from −58 to +20 min). I then
used **grid voltage**, which follows the whole neighbourhood's wall clock routine: zero shift at all
8 transitions, ±60 min fitting twice as badly. Looking at the raw minutes, the nonexistent spring
02:00–02:59 rows turned out to be **exact copies** of 03:00–03:59. So: wall clock logger, regridded by
the publisher. Rewrote the detection, dropped the fabricated and the ambiguous hours (480 rows),
and recorded **amendment A1** in the plan. No model existed yet.

Also caught: missing rows end with an empty field (`?;?;…;?;`), so pandas needed `""` as NA; and
Windows' default cp1252 codec broke a UTF-8 file edit, so all file I/O now names UTF-8 and LF.

**Core modules, then tests before real validation numbers (`4d28e16`).** Feature builder receives an
array that physically ends at the cutoff; the same function serves training, evaluation and the
operator tool. Wrote 56 tests before running `select` on real data (the 57th, for the skill alarm,
was added with amendment A2; the final suite has 57 tests, as stated in the README). The DST window test failed
immediately: pandas 3 defaults to **microsecond** timestamps, so my nanosecond hour arithmetic was
off by 1000×. Fixed by normalising to `ns` everywhere timestamps become integers.

**Validation stage.** Primary baseline seasonal naive day (0.507). Ridge lags+calendar 0.373
(+26 %). Boosted trees +0.1 % for 200× fit time, rejected by the predeclared 3 % bar.

**Backtest and amendment A2 (`45e6531`).** First backtest run: PSI values
above 1 next to KS ≈ 0.14 looked wrong: an empty winter bin with a 1e-6 epsilon dominated; switched
to 0.5-count smoothing. More importantly, **August 2010** showed the model 74 % worse than
seasonal naive (household away) while my drift alarm stayed silent because absolute errors on an
empty house are small. Added a skill alarm (model worse than the primary baseline over the last 7
days → UNCERTAIN), recorded as **amendment A2** with the evidence, adopted by the same predeclared
rule. Deliberately did *not* add a fallback to naive forecast fitted to two holiday months.

**Freeze (`680cdc8`, then `dfe0e8f`).** Committed the frozen manifest and all pretest outputs.
Then noticed `model.json` was written with CRLF while git stores LF; after a clone, the hash check
would refuse to serve. Fixed line endings and refroze; decisions hash unchanged
(`7ead74ad…`), only the model file's byte hash changed.

**Final evaluation, once (06:50 UTC).** MAE 0.494 vs 0.639 (+22.7 %), coverage 0.794 (static would
have been 0.734). Afterwards fixed a duplicated line in the horizon plot (reporting only) and
regenerated: `final_run.log` records `REGENERATION … headline identical: True`.

**Docs and demo assets.** `wattline demo-inputs` writes a clean file and eight broken exports;
`reports/examples/` holds the rendered result of each. Data card, forecast card, evaluation,
retrospective, demo script, check-in draft, CI workflow.

## Decisions I did not take, on purpose

- No weather: no archived forecast for this site/period; observed weather would be leakage.
- No retraining on train+validation before the test: it would turn the interval calibrated on
  validation into an in sample estimate.
- No alpha grid extension after seeing alpha 100 win at the edge: differences were in the third
  decimal and the validation set is 36 days.
- No LLM, agent or dashboard: the brief asked for an honest advisory, not a bigger demo.

**Fresh clone reproducibility check (after `30c6eaf`).** Cloned the repository into a scratch
directory, installed from `uv.lock`, copied in the raw zip (same SHA-256) and (1) served a forecast
from the *committed* frozen artifacts (the hash check passed after a git checkout), then (2) ran
`wattline reproduce` offline. `scripts/compare_metrics.py` matched every headline metric, and
`git status` showed only two changed files: the appended `results/final_run.log`
(`REGENERATION … headline identical: True`) and `results/validation/timings_seconds.json`
(wall clock fit times). Every other result file regenerated byte for byte. Not yet verified: the
same on Linux (that is what the CI `reproduce` job is for).
