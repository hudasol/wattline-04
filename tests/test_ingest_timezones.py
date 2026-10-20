"""Resampling, timezone and DST handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import raw_frame, write_raw_zip

from wattline.calendar_fr import day_type, easter_sunday, is_holiday
from wattline.features import target_times
from wattline.ingest import (audit_and_clean_minutes, detect_clock, dst_transition_dates,
                             logger_to_utc, minutes_to_hourly, missing_runs, read_raw_minutes)
from wattline.pipeline import daily_origins

TZ = "Europe/Paris"


def test_read_raw_parses_question_marks_and_empty_trailing_field(tmp_path):
    df = raw_frame("2010-01-01", "2010-01-01 00:10", gaps=[("2010-01-01 00:03", "2010-01-01 00:05")])
    write_raw_zip(df, tmp_path / "r.zip")
    m = read_raw_minutes(tmp_path / "r.zip", "household_power_consumption.txt")
    assert len(m) == 10
    assert m["Global_active_power"].isna().sum() == 2
    assert m["logger_time"].iloc[0] == pd.Timestamp("2010-01-01 00:00")
    assert m["logger_time"].iloc[-1] == pd.Timestamp("2010-01-01 00:09")  # dd/mm/yyyy parsed day-first


def test_minutes_to_hourly_45_minute_rule_and_units():
    utc = pd.Series(pd.date_range("2010-01-01 00:00", periods=180, freq="min", tz="UTC"))
    kw = pd.Series(np.r_[np.full(60, 2.0), np.full(60, 1.0), np.full(60, 3.0)])
    kw[60:60 + 20] = np.nan      # hour 2: 40 present minutes -> missing
    kw[120:120 + 10] = np.nan    # hour 3: 50 present minutes -> kept, mean of present
    h = minutes_to_hourly(utc, kw, 45)
    assert list(h["n_minutes"]) == [60, 40, 50]
    assert h["energy_kwh"].iloc[0] == pytest.approx(2.0)   # 2 kW for 1 h = 2 kWh
    assert np.isnan(h["energy_kwh"].iloc[1])
    assert h["energy_kwh"].iloc[2] == pytest.approx(3.0)


def test_hourly_grid_is_complete_and_gaps_are_logged():
    utc = pd.Series(pd.to_datetime(["2010-01-01 00:00", "2010-01-01 00:30", "2010-01-01 03:10"], utc=True))
    kw = pd.Series([1.0, 1.0, 1.0])
    h = minutes_to_hourly(utc, kw, 1)
    assert len(h) == 4 and h["energy_kwh"].isna().sum() == 2   # 01:00 and 02:00 absent -> missing
    runs = missing_runs(h["energy_kwh"], TZ)
    assert runs.iloc[0]["hours"] == 2


def test_dst_transition_dates_found_from_tz_database():
    d = dst_transition_dates(range(2010, 2011), TZ)
    assert [str(x.date()) for x in d] == ["2010-03-28", "2010-10-31"]


def test_detect_clock_identifies_regridded_wall_clock():
    df = raw_frame("2010-02-15", "2010-05-01")
    minutes = df.rename(columns={"kw": "Global_active_power", "volt": "Voltage"})[
        ["logger_time", "Global_active_power", "Voltage"]]
    dec = detect_clock(minutes, TZ)
    assert dec.kind == "wall_clock_regridded"
    assert dec.voltage_shift_minutes["2010-03-28"] == 0
    assert dec.spring_hour_duplicate_frac["2010-03-28"] == pytest.approx(1.0)


def test_detect_clock_identifies_fixed_offset_logger():
    df = raw_frame("2010-02-15", "2010-05-01", fixed_clock=True)
    minutes = df.rename(columns={"kw": "Global_active_power", "volt": "Voltage"})[
        ["logger_time", "Global_active_power", "Voltage"]]
    assert detect_clock(minutes, TZ).kind == "fixed_offset"


def test_wall_clock_conversion_drops_nonexistent_and_ambiguous_minutes():
    from wattline.ingest import ClockDecision
    dec = ClockDecision("wall_clock_regridded", float("nan"), {}, {}, {}, "")
    t = pd.Series(pd.to_datetime([
        "2010-01-15 12:00", "2010-07-15 12:00",        # CET / CEST
        "2010-03-28 02:30",                             # does not exist
        "2010-10-31 02:30",                             # happens twice
        "2010-03-28 03:00", "2010-10-31 03:00",
    ]))
    u = logger_to_utc(t, dec, TZ)
    assert u.iloc[0] == pd.Timestamp("2010-01-15 11:00", tz="UTC")
    assert u.iloc[1] == pd.Timestamp("2010-07-15 10:00", tz="UTC")
    assert pd.isna(u.iloc[2]) and pd.isna(u.iloc[3])  # never guessed
    assert u.iloc[4] == pd.Timestamp("2010-03-28 01:00", tz="UTC")
    assert u.iloc[5] == pd.Timestamp("2010-10-31 02:00", tz="UTC")


def test_audit_counts_duplicates_and_physical_violations():
    m = pd.DataFrame({
        "logger_time": pd.to_datetime(["2010-01-01 00:00", "2010-01-01 00:00", "2010-01-01 00:01",
                                       "2010-01-01 00:02", "2010-01-01 00:02"]),
        "Global_active_power": [1.0, 1.0, 99.0, 1.0, 2.0],
        "Global_reactive_power": 0.1, "Voltage": 240.0, "Global_intensity": 5.0,
        "Sub_metering_1": 0.0, "Sub_metering_2": 0.0, "Sub_metering_3": 0.0,
    })
    clean, audit = audit_and_clean_minutes(m, max_kw=15.0)
    assert audit.duplicate_timestamps == 2
    assert audit.conflicting_duplicates == 1          # 00:02 has 1.0 vs 2.0
    assert audit.physical_range_violations == 1        # 99 kW
    assert len(clean) == 3
    assert clean["Global_active_power"].isna().sum() == 2   # conflict + violation set missing


def test_daily_origins_are_local_midnights_across_dst(real_cfg):
    idx = pd.date_range("2010-03-20", "2010-04-05", freq="h", tz="UTC")
    o = daily_origins(idx, real_cfg)
    loc = o.tz_convert(TZ)
    assert (loc.hour == 0).all() and (loc.minute == 0).all()
    assert o[o.tz_convert(TZ).date == pd.Timestamp("2010-03-27").date()][0].hour == 23  # CET
    assert o[o.tz_convert(TZ).date == pd.Timestamp("2010-03-29").date()][0].hour == 22  # CEST


def test_forecast_window_on_autumn_dst_day_is_24_absolute_hours():
    origin = pd.Timestamp("2010-10-31 00:00", tz=TZ).tz_convert("UTC")
    tt = target_times(origin)
    assert len(tt) == 24 and (np.diff(tt.asi8) == 3600 * 10**9).all()
    labels = [t.strftime("%H:%M%z") for t in tt.tz_convert(TZ)]
    assert labels.count("02:00+0200") == 1 and labels.count("02:00+0100") == 1   # repeated local hour
    assert tt[-1].tz_convert(TZ).strftime("%H:%M") == "22:00"   # 25 h local day: ends before 23:00


def test_forecast_window_on_spring_dst_day_skips_missing_local_hour():
    origin = pd.Timestamp("2010-03-28 00:00", tz=TZ).tz_convert("UTC")
    loc = target_times(origin).tz_convert(TZ)
    assert "02:00" not in [t.strftime("%H:%M") for t in loc]
    assert loc[-1].strftime("%Y-%m-%d %H:%M") == "2010-03-29 00:00"   # 23 h local day


def test_french_holidays_and_day_type():
    assert easter_sunday(2010) == pd.Timestamp("2010-04-04").date()
    assert is_holiday(pd.Timestamp("2010-04-05").date())   # Easter Monday
    assert is_holiday(pd.Timestamp("2010-11-01").date()) and is_holiday(pd.Timestamp("2010-11-11").date())
    assert day_type(pd.Timestamp("2010-11-01 10:00", tz=TZ)) == "holiday"
    assert day_type(pd.Timestamp("2010-11-06 10:00", tz=TZ)) == "weekend"
    assert day_type(pd.Timestamp("2010-11-05 10:00", tz=TZ)) == "weekday"


def test_config_split_boundaries_convert_wall_clock_to_utc(real_cfg):
    assert real_cfg.train_end_utc == pd.Timestamp("2010-08-31 22:00", tz="UTC")   # CEST
    assert real_cfg.test_end_utc == pd.Timestamp("2010-11-25 23:00", tz="UTC")    # CET
    assert real_cfg.train_end_utc < real_cfg.val_end_utc < real_cfg.test_end_utc
