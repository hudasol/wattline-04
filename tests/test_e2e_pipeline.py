"""End-to-end: synthetic raw zip -> ingest -> select -> backtest -> freeze -> final-eval
-> regeneration -> refusal after a decision change -> CLI forecasts. Sockets disabled."""

from __future__ import annotations

import json
import os
import stat

import pandas as pd
import pytest
from conftest import SYNTH_OVERRIDES, write_config
from synthetic import raw_frame, write_meter_csv, write_raw_zip

from wattline.cli import main
from wattline.config import load_config
from wattline.ingest import load_hourly
from wattline.pipeline import FrozenMismatch, run_final_eval, run_reproduce
from wattline.util import NetworkDisabled, no_network


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("wattline_e2e")
    df = raw_frame("2010-01-10", "2010-05-12", seed=7,
                   gaps=[("2010-02-03 10:00", "2010-02-03 16:30"),     # 6.5 h gap in train
                         ("2010-04-15 01:00", "2010-04-15 03:00")])    # 2 h gap in validation
    sha = write_raw_zip(df, root / "data" / "raw" / "household_power_consumption.zip")
    os.chmod(root / "data" / "raw" / "household_power_consumption.zip", stat.S_IREAD)
    cfg_path = write_config(root, SYNTH_OVERRIDES | {"sha256": sha})
    return root, cfg_path


@pytest.mark.e2e
def test_full_pipeline_offline(project):
    root, cfg_path = project
    cfg = load_config(cfg_path)
    steps = run_reproduce(cfg)                     # runs inside no_network()
    assert steps["ingest"] > 24 * 100
    rep = json.loads((cfg.results_dir / "data" / "ingest_report.json").read_text())
    assert rep["timezone"]["decision"]["kind"] == "wall_clock_regridded"
    assert rep["minute_audit"]["dst_rows_dropped"] == 60          # one spring transition
    assert rep["hourly"]["missing_hours"] >= 6
    for f in ["validation/selection.json", "backtest/backtest.json", "frozen/frozen_manifest.json",
              "frozen/artifacts.json", "final/metrics.json", "final/test_predictions.csv",
              "final/per_horizon.csv", "final/final_eval_lock.json", "final_run.log",
              "final/figures/worst_day.png"]:
        assert (cfg.results_dir / f).exists(), f
    m = json.loads((cfg.results_dir / "final" / "metrics.json").read_text())
    assert m["boundary_audit"]["test_starts_after_train_and_validation"]
    assert m["boundary_audit"]["all_feature_inputs_end_at_or_before_issue_time"]
    assert m["headline"]["test_origins"] >= 14
    assert m["serving_equivalence_max_abs_diff_kwh"] < 1e-3
    ph = pd.read_csv(cfg.results_dir / "final" / "per_horizon.csv")
    assert list(ph.horizon) == list(range(1, 25))
    preds = pd.read_csv(cfg.results_dir / "final" / "test_predictions.csv")
    assert (pd.to_datetime(preds.target_utc) >= cfg.val_end_utc).all()   # only locked-test targets

    # regeneration with identical decisions is allowed and identical
    m2 = run_final_eval(cfg)
    assert m2["headline"]["model_mae_kwh"] == pytest.approx(m["headline"]["model_mae_kwh"], abs=1e-9)
    log = (cfg.results_dir / "final_run.log").read_text().splitlines()
    assert "FIRST SCORING" in log[0] and "REGENERATION" in log[1]


@pytest.mark.e2e
def test_cli_forecast_ok_and_failures(project, tmp_path):
    root, cfg_path = project
    cfg = load_config(cfg_path)
    s = load_hourly(cfg)
    issue = pd.Timestamp("2010-05-05 00:00", tz="Europe/Paris")
    with no_network():
        code = main(["--config", str(cfg_path), "forecast", "--issue-time", issue.isoformat(),
                     "--out", str(tmp_path / "ok")])
    assert code in (0, 3)
    assert (tmp_path / "ok" / "forecast.html").exists() and (tmp_path / "ok" / "forecast.png").exists()
    out = json.loads((tmp_path / "ok" / "forecast.json").read_text())
    assert len(out["hours"]) == 24

    stale = write_meter_csv(s[s.index < issue - pd.Timedelta(hours=12)], tmp_path / "stale.csv")
    assert main(["--config", str(cfg_path), "forecast", "--issue-time", issue.isoformat(),
                 "--input", str(stale), "--out", str(tmp_path / "stale")]) == 2
    bad_unit = write_meter_csv(s * 1000, tmp_path / "wh.csv", value_col="energy_wh")
    assert main(["--config", str(cfg_path), "forecast", "--issue-time", issue.isoformat(),
                 "--input", str(bad_unit), "--out", str(tmp_path / "wh")]) == 2
    err = json.loads((tmp_path / "wh" / "forecast.json").read_text())
    assert err["status"] == "UNAVAILABLE" and err["hours"] == []


@pytest.mark.e2e
def test_rescoring_after_decision_change_is_refused(project):
    root, cfg_path = project
    from wattline.pipeline import run_freeze
    text = cfg_path.read_text(encoding="utf-8")
    try:
        cfg_path.write_text(text.replace("high_quantile = 0.90", "high_quantile = 0.80"), encoding="utf-8")
        cfg2 = load_config(cfg_path)
        with pytest.raises(FrozenMismatch):        # config changed after freeze
            run_final_eval(cfg2)
        from wattline.pipeline import run_select, run_backtest
        run_select(cfg2)
        run_backtest(cfg2)
        run_freeze(cfg2)
        with pytest.raises(FrozenMismatch):        # re-frozen with new decisions -> lock refuses
            run_final_eval(cfg2)
    finally:
        cfg_path.write_text(text, encoding="utf-8")


def test_no_network_guard_blocks_sockets():
    import socket
    with no_network(), pytest.raises(NetworkDisabled):
        socket.create_connection(("example.com", 80), timeout=1)
