"""Figures and the operator-facing forecast report (JSON + CSV + PNG + HTML)."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .util import write_csv, write_json  # noqa: E402

COLORS = {"model": "#1f5fa8", "band": "#9cc0e8", "actual": "#222222", "base": "#d07a1f",
          "high": "#c0392b", "low": "#2e8b57", "grey": "#888888"}
STATUS_COLOR = {"OK": "#2e8b57", "UNCERTAIN": "#d68910", "UNAVAILABLE": "#c0392b"}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- final figures
def final_figures(cfg, prep, P, lo, hi, M, hdf, hour_df, worst_i, best_i, primary, rule, outdir: Path):
    tz = cfg.data.local_tz
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, lab, c, ls in [("model_mae", "model", COLORS["model"], "-"),
                            (f"{primary}_mae", f"{primary} (primary)", COLORS["base"], "-"),
                            ("seasonal_naive_day_mae", "seasonal_naive_day", COLORS["grey"], "--"),
                            ("last_value_mae", "last_value", COLORS["grey"], ":"),
                            ("mean_7d_same_hour_mae", "7-day mean rule", "#7d3c98", "-.")]:
        if col in hdf and not (col == f"{primary}_mae" and lab != f"{primary} (primary)"):
            ax.plot(hdf.horizon, hdf[col], ls, color=c, label=lab, lw=2 if lab == "model" else 1.3)
    ax.set_xlabel("horizon h (hours after issue)")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Locked test: MAE by forecast horizon (same origins for every line)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, outdir / "mae_by_horizon.png")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(hdf.horizon, hdf.coverage, color=COLORS["band"])
    ax.axhline(0.8, color=COLORS["model"], ls="--", label="nominal 80 %")
    ax.axhline(0.7, color=COLORS["high"], ls=":", label="bar 70 %")
    ax.set_ylim(0, 1)
    ax.set_xlabel("horizon h")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Locked test: 80 % interval coverage by horizon")
    ax.legend(fontsize=8)
    _save(fig, outdir / "coverage_by_horizon.png")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(hour_df.hour_local, hour_df.model_mae, "o-", color=COLORS["model"], label="model")
    ax.plot(hour_df.hour_local, hour_df[f"{primary}_mae"], "s--", color=COLORS["base"], label=primary)
    ax.set_xlabel("local hour of day (Europe/Paris)")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Locked test: error by hour of day")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, outdir / "mae_by_hour.png")

    for tag, i in [("worst_day", worst_i), ("best_day", best_i)]:
        tt = pd.to_datetime(prep.target_ns[i], utc=True).tz_convert(tz)
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(24)
        ax.fill_between(x, lo[i], hi[i], color=COLORS["band"], alpha=0.6, label="80 % interval")
        ax.plot(x, P[i], color=COLORS["model"], lw=2, label="forecast")
        ax.plot(x, prep.base[primary][i], "--", color=COLORS["base"], label=primary)
        ax.plot(x, prep.Y[i], "o-", color=COLORS["actual"], ms=3, label="actual")
        ax.axhline(rule.high_kwh, color=COLORS["high"], ls=":", label="HIGH_DEMAND threshold")
        ax.set_xticks(x[::3], [t.strftime("%H:%M") for t in tt[::3]])
        mae_i = np.nanmean(np.abs(prep.Y[i] - P[i]))
        ax.set_title(f"{tag.replace('_', ' ')} of the locked test: issued {prep.origins[i].tz_convert(tz):%a %Y-%m-%d %H:%M %Z}"
                     f" (MAE {mae_i:.2f} kWh)", fontsize=10)
        ax.set_ylabel("kWh in hour")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
        _save(fig, outdir / f"{tag}.png")


def backtest_figure(fdf: pd.DataFrame, primary: str, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = np.arange(len(fdf))
    ax1.plot(x, fdf.model_mae, "o-", color=COLORS["model"], label="model")
    ax1.plot(x, fdf[f"{primary}_mae"], "s--", color=COLORS["base"], label=primary)
    ax1.plot(x, fdf.mean_7d_same_hour_mae, "^-.", color="#7d3c98", label="7-day mean rule")
    ax1.set_ylabel("MAE (kWh)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title("Walk-forward backtest (monthly folds, pre-validation)")
    ax2.plot(x, fdf.coverage_static, "o-", color=COLORS["band"], label="coverage static")
    ax2.plot(x, fdf.coverage_adaptive, "o-", color=COLORS["model"], label="coverage adaptive")
    ax2.axhline(0.8, color="k", ls="--", lw=0.8)
    ax2b = ax2.twinx()
    ax2b.bar(x, fdf.ks_vs_train, alpha=0.25, color=COLORS["grey"], label="KS vs train window")
    ax2b.set_ylabel("KS shift")
    ax2.set_ylabel("80 % coverage")
    ax2.set_xticks(x, fdf.fold, rotation=60, fontsize=7)
    ax2.legend(fontsize=8, loc="lower left")
    return _save(fig, path)


# --------------------------------------------------------------------------- operator report
def forecast_figure(report: dict, history: pd.Series | None, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    status = report["status"]
    hours = report["hours"]
    if hours:
        t = pd.to_datetime([h["hour_start_utc"] for h in hours], utc=True)
        f = np.array([h["forecast_kwh"] for h in hours])
        lo = np.array([h["lower80_kwh"] for h in hours])
        hi = np.array([h["upper80_kwh"] for h in hours])
        if history is not None:
            hist = history.iloc[-48:]
            ax.plot(hist.index, hist.to_numpy(), color=COLORS["actual"], lw=1, label="observed (last 48 h)")
        ax.fill_between(t, lo, hi, step=None, color=COLORS["band"], alpha=0.6, label="80 % interval")
        ax.plot(t, f, color=COLORS["model"], lw=2, label="forecast")
        thr = report["thresholds"]["high_demand_kwh"]
        ax.axhline(thr, color=COLORS["high"], ls=":", label=f"HIGH_DEMAND threshold {thr:.2f} kWh")
        for h, tt in zip(hours, t):
            if h["advisory"] == "HIGH_DEMAND":
                ax.axvspan(tt, tt + pd.Timedelta(hours=1), color=COLORS["high"], alpha=0.15)
            if h["likely_peak"]:
                ax.plot(tt, h["forecast_kwh"], "v", color=COLORS["high"], ms=8)
        ax.axvline(pd.Timestamp(report["issue_time_utc"]), color="k", lw=1)
        ax.text(pd.Timestamp(report["issue_time_utc"]), ax.get_ylim()[1] * 0.95, " issue time / data cutoff",
                fontsize=8, va="top")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("kWh in hour")
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M", tz=report_tz(report)))
    else:
        ax.text(0.5, 0.5, "UNAVAILABLE - no forecast produced\n" + "\n".join(report["reasons"][:4]),
                ha="center", va="center", fontsize=9, wrap=True, transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title(f"wattline day-ahead advisory - issued {report['issue_time_local']} - status {status}",
                 color=STATUS_COLOR.get(status, "k"), fontsize=10)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def report_tz(report: dict):
    import zoneinfo
    return zoneinfo.ZoneInfo("Europe/Paris")


def write_forecast_outputs(report: dict, history: pd.Series | None, outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {"json": write_json(report, outdir / "forecast.json")}
    if report["hours"]:
        paths["csv"] = write_csv(pd.DataFrame(report["hours"]), outdir / "forecast.csv", float_format="%.4f")
    paths["png"] = forecast_figure(report, history, outdir / "forecast.png")
    paths["html"] = outdir / "forecast.html"
    paths["html"].write_text(render_html(report, paths["png"]), encoding="utf-8", newline="\n")
    return paths


def render_html(report: dict, png: Path) -> str:
    img = base64.b64encode(png.read_bytes()).decode()
    st = report["status"]
    esc = html.escape
    reasons = "".join(f"<li>{esc(r)}</li>" for r in report["reasons"]) or "<li>all input checks passed</li>"
    rows = "".join(
        f"<tr class='{h['advisory']}'><td>{h['horizon']}</td><td>{esc(h['hour_start_local'][5:16].replace('T', ' '))}</td>"
        f"<td>{h['forecast_kwh']:.3f}</td><td>{h['lower80_kwh']:.3f} - {h['upper80_kwh']:.3f}</td>"
        f"<td>{h['advisory']}</td><td>{'peak' if h['likely_peak'] else ''}</td></tr>"
        for h in report["hours"])
    prov = "".join(
        f"<tr><td>{esc(p['feature'])}</td><td>{esc(str(p['earliest_input_hour_start']))}</td>"
        f"<td>{esc(str(p['latest_input_hour_end']))}</td><td>{'yes' if p['within_cutoff'] else 'NO'}</td></tr>"
        for p in report.get("feature_provenance", []))
    thr = report.get("thresholds", {})
    mon = report.get("monitor", {})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>wattline advisory {esc(report['issue_time_local'])}</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;max-width:960px;margin:1.5rem auto;padding:0 16px;color:#222;background:#fff}}
.banner{{padding:.8rem 1rem;border-radius:6px;color:#fff;background:{STATUS_COLOR.get(st, '#555')};font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.6rem 0}}td,th{{border-bottom:1px solid #ddd;padding:.25rem .4rem;text-align:left}}
tr.HIGH_DEMAND td{{background:#fbe4e1}} tr.UNCERTAIN td{{background:#fdf2dc}} tr.LOW td{{background:#e3f4ea}}
img{{max-width:100%}} code{{background:#f3f3f3;padding:0 .2rem}} .muted{{color:#666;font-size:.85rem}}
</style></head><body>
<h1>Day-ahead electricity advisory</h1>
<div class="banner">STATUS: {st} &middot; issued {esc(report['issue_time_local'])}</div>
<p class="muted">{esc(report['data_cutoff'])}. Model <code>{esc(report['model'])}</code>; frozen manifest
<code>{esc(report['frozen_manifest_sha256'][:16])}</code>. Advisory only - never used to control equipment.</p>
<h2>Why this status</h2><ul>{reasons}</ul>
{'<img alt="forecast chart" src="data:image/png;base64,' + img + '">'}
{'<h2>Likely peak hours</h2><p>' + ', '.join(report.get('peak_hours_local', [])) + '</p>' if report['hours'] else ''}
{'<h2>Thresholds</h2><p>HIGH_DEMAND: ' + esc(thr.get('flag_rule', '')) + f" (threshold {thr.get('high_demand_kwh', 0):.3f} kWh = " + esc(thr.get('high_demand_definition', '')) + '). ' + esc(thr.get('low_definition', '')) + f" ({thr.get('low_kwh', 0):.3f} kWh). Interval stretch from monitor: x{thr.get('interval_stretch', 1):.2f}.</p>" if thr else ''}
{'<p class="muted">Monitor: ' + esc(str(mon.get('note', ''))) + '</p>' if mon else ''}
{'<h2>24 hourly forecasts (kWh)</h2><table><tr><th>h</th><th>local hour</th><th>forecast</th><th>80 % interval</th><th>advisory</th><th></th></tr>' + rows + '</table>' if rows else ''}
{'<h2>Data cutoff of every input</h2><table><tr><th>feature</th><th>earliest input</th><th>latest input ends</th><th>&le; issue time</th></tr>' + prov + '</table>' if prov else ''}
<h2>Input checks</h2><pre>{esc(json.dumps(report['checks'], indent=1))}</pre>
<p class="muted">The forecast can be wrong: an 80 % interval is expected to miss about 1 hour in 5, mostly on
sudden appliance peaks. When the status is UNCERTAIN no hour is called NORMAL or LOW.</p>
</body></html>"""
