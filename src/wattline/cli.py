"""wattline command line.

    wattline download                 fetch the raw zip (the ONLY networked command)
    wattline ingest                   raw minutes -> canonical hourly table + audit
    wattline select                   validation: baselines, model grid, intervals, thresholds
    wattline backtest                 walk-forward backtest on pre-validation folds
    wattline freeze                   write the frozen model + manifest
    wattline final-eval               score the locked test (once; later runs only regenerate)
    wattline reproduce                ingest -> ... -> final-eval, offline (sockets disabled)
    wattline forecast --issue-time T  24 h advisory for one issue time (JSON/CSV/PNG/HTML)

Exit codes for ``forecast``: 0 OK, 3 UNCERTAIN, 2 UNAVAILABLE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import load_config


def _print_forecast(rep: dict) -> None:
    bar = "=" * 72
    print(bar)
    print(f" wattline advisory | issued {rep['issue_time_local']} | STATUS {rep['status']}")
    print(bar)
    for r in rep["reasons"]:
        print("  " + r)
    if rep["hours"]:
        print(f"\n  {'h':>2}  {'local hour':<17} {'kWh':>6}  {'80% interval':>15}  advisory")
        for h in rep["hours"]:
            mark = "  <- likely peak" if h["likely_peak"] else ""
            print(f"  {h['horizon']:>2}  {h['hour_start_local'][:16].replace('T', ' '):<17} {h['forecast_kwh']:>6.3f}"
                  f"  {h['lower80_kwh']:>6.3f}-{h['upper80_kwh']:<7.3f}  {h['advisory']}{mark}")
        t = rep["thresholds"]
        print(f"\n  peak hours: {', '.join(rep['peak_hours_local'])}")
        print(f"  HIGH_DEMAND threshold {t['high_demand_kwh']:.3f} kWh ({t['high_demand_definition']}); {t['flag_rule']}")
    print(bar)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wattline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to wattline.toml (default: ./config/wattline.toml)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--force", action="store_true")
    for name in ["ingest", "select", "backtest", "freeze", "final-eval", "reproduce"]:
        sub.add_parser(name)
    di = sub.add_parser("demo-inputs", help="write clean + deliberately broken meter CSVs for the demo")
    di.add_argument("--issue-time", required=True)
    di.add_argument("--out", default="reports/demo_inputs")
    f = sub.add_parser("forecast")
    f.add_argument("--issue-time", required=True,
                   help="ISO time with offset, e.g. 2010-11-20T00:00+01:00 (must be 00:00 Europe/Paris)")
    f.add_argument("--input", default=None, help="hourly CSV (timestamp_utc, energy_kwh); default: canonical table")
    f.add_argument("--out", default=None, help="output directory (default reports/forecast_<date>)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "download":
        from .ingest import download
        download(cfg, force=args.force)
        return 0
    if args.cmd == "ingest":
        from .ingest import ingest
        rep = ingest(cfg)
        print(json.dumps(rep["hourly"] | {"timezone_decision": rep["timezone"]["decision"]["kind"]}, indent=1))
        return 0
    if args.cmd in ("select", "backtest", "freeze", "final-eval", "reproduce"):
        from . import pipeline
        fn = {"select": pipeline.run_select, "backtest": pipeline.run_backtest, "freeze": pipeline.run_freeze,
              "final-eval": pipeline.run_final_eval, "reproduce": pipeline.run_reproduce}[args.cmd]
        try:
            res = fn(cfg)
        except pipeline.FrozenMismatch as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 4
        brief = {"select": lambda r: {k: r[k] for k in ("primary_baseline", "chosen_model_name", "chosen_val_mae",
                                                         "val_improvement_vs_primary")},
                 "backtest": lambda r: r["pooled"] | {"monitor_adopted": r["monitor_adoption"]["adopted"]},
                 "freeze": lambda r: {"decisions_sha256": r["decisions_sha256"], "model_file": r["model_file"]},
                 "final-eval": lambda r: r["headline"] | {"coverage": r["uncertainty"]["coverage"]},
                 "reproduce": lambda r: r}[args.cmd](res)
        print(json.dumps(brief, indent=1, default=str))
        return 0
    if args.cmd == "demo-inputs":
        from .demo import write_demo_inputs
        for p in write_demo_inputs(cfg, pd.Timestamp(args.issue_time), Path(args.out)):
            print(p)
        return 0
    if args.cmd == "forecast":
        from .ingest import load_hourly
        from .report import write_forecast_outputs
        from .serving import EXIT_CODES, InputContractError, load_artifacts, make_forecast, read_meter_csv
        art = load_artifacts(cfg)
        issue = pd.Timestamp(args.issue_time)
        try:
            series = read_meter_csv(args.input) if args.input else load_hourly(cfg)
            rep = make_forecast(cfg, art, series, issue)
        except InputContractError as exc:
            rep = {"issue_time_local": str(issue), "issue_time_utc": str(issue), "status": "UNAVAILABLE",
                   "reasons": [f"[UNAVAILABLE] input contract violated: {exc}"], "checks": {},
                   "model": art.model.spec.name, "frozen_manifest_sha256": art.manifest_sha256,
                   "data_cutoff": "n/a", "hours": []}
            series = None
        stamp = issue.strftime("%Y%m%dT%H%M") if issue.tzinfo else "invalid"
        out = Path(args.out) if args.out else cfg.reports_dir / f"forecast_{stamp}"
        hist = None
        if series is not None and rep["hours"] and issue.tzinfo is not None:
            hist = series[series.index < issue]
        paths = write_forecast_outputs(rep, hist, out)
        _print_forecast(rep)
        print("written: " + ", ".join(str(p) for p in paths.values()))
        if rep["status"] == "UNAVAILABLE":
            print("UNAVAILABLE: no forecast was produced. See reasons above.", file=sys.stderr)
        return EXIT_CODES[rep["status"]]
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
