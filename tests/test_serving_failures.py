"""Invalid, stale and degraded input must fail loudly - never look like good news."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import issue_after
from synthetic import write_meter_csv

from wattline.advisory import HIGH, LOW, NORMAL, UNCERTAIN
from wattline.serving import InputContractError, make_forecast, read_meter_csv


def _hist(series, issue):
    return series[series.index < issue]


def test_clean_input_gives_ok_forecast_with_24_hours(serving_setup):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3)
    rep = make_forecast(cfg, art, _hist(s, issue), issue)
    assert rep["status"] == "OK", rep["reasons"]
    assert len(rep["hours"]) == 24
    assert {h["advisory"] for h in rep["hours"]} <= {HIGH, NORMAL, LOW}
    assert all(h["lower80_kwh"] <= h["forecast_kwh"] <= h["upper80_kwh"] for h in rep["hours"])
    assert len(rep["peak_hours_local"]) == 3
    assert all(p["within_cutoff"] for p in rep["feature_provenance"])


def test_rows_after_issue_time_are_ignored_not_used(serving_setup):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3)
    a = make_forecast(cfg, art, _hist(s, issue), issue)
    future = s.copy()
    future[future.index >= issue] *= 50          # wildly different "future"
    b = make_forecast(cfg, art, future, issue)
    assert [h["forecast_kwh"] for h in a["hours"]] == [h["forecast_kwh"] for h in b["hours"]]
    assert b["checks"]["rows_after_issue_time_ignored"] > 0


@pytest.mark.parametrize("stale_hours, expected", [(3, "UNCERTAIN"), (10, "UNAVAILABLE")])
def test_stale_meter_is_uncertain_then_unavailable(serving_setup, stale_hours, expected):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3)
    h = _hist(s, issue).iloc[:-stale_hours]          # last readings never arrived
    rep = make_forecast(cfg, art, h, issue)
    assert rep["status"] == expected
    assert any("stale" in r or "last reading" in r for r in rep["reasons"])
    if expected == "UNCERTAIN":
        assert not {NORMAL, LOW} & {x["advisory"] for x in rep["hours"]}   # refuses to reassure
        assert UNCERTAIN in {x["advisory"] for x in rep["hours"]} or all(
            x["advisory"] == HIGH for x in rep["hours"])
    else:
        assert rep["hours"] == []


def test_stuck_meter_flatline_is_uncertain(serving_setup):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3)
    h = _hist(s, issue).copy()
    h.iloc[-8:] = 0.42                               # meter repeating one value
    rep = make_forecast(cfg, art, h, issue)
    assert rep["status"] == "UNCERTAIN"
    assert any("stuck meter" in r for r in rep["reasons"])
    assert not {NORMAL, LOW} & {x["advisory"] for x in rep["hours"]}


def test_missing_history_is_unavailable(serving_setup):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3)
    rep = make_forecast(cfg, art, _hist(s, issue).iloc[-100:], issue)
    assert rep["status"] == "UNAVAILABLE" and "insufficient history" in " ".join(rep["reasons"])


def test_gaps_in_recent_window(serving_setup):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3)
    h = _hist(s, issue).copy()
    h.iloc[-30:-28] = np.nan                         # 2 missing hours inside last 24 h? (no: 28-30 h ago)
    h.iloc[-10:-8] = np.nan                          # 2 missing hours inside last 24 h
    rep = make_forecast(cfg, art, h, issue)
    assert rep["status"] == "UNCERTAIN" and rep["imputed_input_hours_last_168h"] == 4
    h2 = _hist(s, issue).copy()
    h2.iloc[-160:-100] = np.nan                      # 36 % of the last week missing
    assert make_forecast(cfg, art, h2, issue)["status"] == "UNAVAILABLE"


def test_suspected_unit_error_is_unavailable(serving_setup):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3)
    rep = make_forecast(cfg, art, _hist(s, issue) * 1000, issue)   # Wh labelled as kWh
    assert rep["status"] == "UNAVAILABLE" and "unit" in " ".join(rep["reasons"])


def test_issue_time_outside_contract_is_unavailable(serving_setup):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3) + pd.Timedelta(hours=6)
    rep = make_forecast(cfg, art, _hist(s, issue), issue)
    assert rep["status"] == "UNAVAILABLE" and "contract" in " ".join(rep["reasons"])
    rep2 = make_forecast(cfg, art, s, pd.Timestamp("2010-03-01 00:00"))   # no timezone
    assert rep2["status"] == "UNAVAILABLE"


def test_monitor_alarm_downgrades_and_widens(serving_setup):
    cfg, art, s = serving_setup
    issue = issue_after(s, 3)
    h = _hist(s, issue).copy()
    rng = np.random.default_rng(9)
    h.iloc[-24 * 7:] = h.iloc[-24 * 7:] * rng.lognormal(0.6, 0.9, 24 * 7)   # regime change
    rep = make_forecast(cfg, art, h, issue)
    assert rep["monitor"]["alarm"] and rep["status"] == "UNCERTAIN"
    assert rep["thresholds"]["interval_stretch"] > 1.0


# ---------------------------------------------------------------- CSV input contract
def test_contract_accepts_canonical_csv(tmp_path, serving_setup):
    _, _, s = serving_setup
    got = read_meter_csv(write_meter_csv(s, tmp_path / "ok.csv", extra={"unit": "kWh"}))
    assert len(got) == len(s) and got.index.tz is not None
    np.testing.assert_allclose(got.to_numpy(), s.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("kind, message", [
    ("unit_column", "unexpected unit"),
    ("unit_value", "only 'kWh'"),
    ("naive", "without explicit UTC offset"),
    ("duplicate", "duplicate timestamp"),
    ("misaligned", "not on the hour"),
    ("negative", "negative"),
    ("text", "non-numeric"),
])
def test_contract_violations_raise(tmp_path, serving_setup, kind, message):
    _, _, s = serving_setup
    s = s.iloc[:300].copy()
    path = tmp_path / f"{kind}.csv"
    if kind == "unit_column":
        write_meter_csv(s * 1000, path, value_col="energy_wh")
    elif kind == "unit_value":
        write_meter_csv(s, path, extra={"unit": "Wh"})
    elif kind == "naive":
        write_meter_csv(s, path, ts_format="naive")
    elif kind == "duplicate":
        write_meter_csv(pd.concat([s, s.iloc[[10]]]), path)
    elif kind == "misaligned":
        s.index = s.index + pd.Timedelta(minutes=15)
        write_meter_csv(s, path)
    elif kind == "negative":
        s.iloc[5] = -1.0
        write_meter_csv(s, path)
    elif kind == "text":
        write_meter_csv(s, path)
        txt = path.read_text().splitlines()
        txt[3] = txt[3].rsplit(",", 1)[0] + ",n/a"
        path.write_text("\n".join(txt) + "\n")
    with pytest.raises(InputContractError, match=message):
        read_meter_csv(path)
