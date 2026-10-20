"""No-leakage guards and forecast-window alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wattline.baselines import baseline_forecast, baseline_matrix
from wattline.clean import causal_fill, fit_profile
from wattline.features import (ALL_FEATURES, LAG_FEATURES, InsufficientHistory, build_design,
                               origin_features, positions, provenance, target_times, targets)

TZ = "Europe/Paris"


def _grid(n=24 * 30, start="2010-01-01T23:00Z"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


def test_corrupting_the_future_never_changes_features():
    """Change every value at or after the issue time -> identical features."""
    idx = _grid()
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 0.5, len(idx))
    imp = np.zeros(len(idx), bool)
    p = 24 * 20
    X1 = build_design(y, imp, idx, np.array([p]), TZ)
    y2 = y.copy()
    y2[p:] = 1e6                     # absurd future
    imp2 = imp.copy()
    imp2[p:] = True
    X2 = build_design(y2, imp2, idx, np.array([p]), TZ)
    np.testing.assert_array_equal(X1, X2)


def test_lags_point_at_exactly_target_minus_lag():
    """With y[i] = i the lag value reveals the exact hour it was read from."""
    idx = _grid()
    y = np.arange(len(idx), dtype=float)
    p = 24 * 15
    X = origin_features(y[:p], np.zeros(p), idx[p], TZ)
    col = {n: X[:, ALL_FEATURES.index(n)] for n in LAG_FEATURES}
    target = p + np.arange(24)            # horizon h -> hour p + h - 1
    np.testing.assert_array_equal(col["lag_24"], target - 24)
    np.testing.assert_array_equal(col["lag_48"], target - 48)
    np.testing.assert_array_equal(col["lag_168"], target - 168)
    np.testing.assert_array_equal(col["same_hour_mean_7d"], target - 24 * 4)   # mean of -24..-168
    np.testing.assert_array_equal(col["last_value"], np.full(24, p - 1))
    assert (np.c_[col["lag_24"], col["lag_168"], col["last_value"]] < p).all()  # all strictly before issue


def test_position_specs_match_computed_features():
    idx = _grid()
    y = np.random.default_rng(3).random(len(idx))
    p = 24 * 12
    X = origin_features(y[:p], np.zeros(p), idx[p], TZ)
    for name in LAG_FEATURES:
        for h in range(1, 25):
            expected = y[:p][positions(name, h)].mean()
            assert X[h - 1, ALL_FEATURES.index(name)] == pytest.approx(expected)
            assert positions(name, h).max() <= -1     # never index >= issue


def test_provenance_reports_every_input_ending_by_the_issue_time():
    origin = pd.Timestamp("2010-11-20 00:00", tz=TZ).tz_convert("UTC")
    rows = provenance(origin, TZ)
    assert all(r["within_cutoff"] for r in rows)
    lv = next(r for r in rows if r["feature"] == "last_value")
    assert pd.Timestamp(lv["latest_input_hour_end"]) == origin
    l168 = next(r for r in rows if r["feature"] == "lag_168")
    assert pd.Timestamp(l168["earliest_input_hour_start"]) == origin - pd.Timedelta(hours=168)


def test_baselines_agree_with_feature_columns_and_definitions():
    idx = _grid()
    y = np.random.default_rng(5).random(len(idx))
    p = 24 * 10
    X = origin_features(y[:p], np.zeros(p), idx[p], TZ)
    np.testing.assert_allclose(baseline_forecast("seasonal_naive_day", y[:p]), X[:, ALL_FEATURES.index("lag_24")])
    np.testing.assert_allclose(baseline_forecast("seasonal_naive_week", y[:p]), X[:, ALL_FEATURES.index("lag_168")])
    np.testing.assert_allclose(baseline_forecast("last_value", y[:p]), np.full(24, y[p - 1]))
    np.testing.assert_allclose(baseline_forecast("mean_7d_same_hour", y[:p]),
                               X[:, ALL_FEATURES.index("same_hour_mean_7d")])
    # baselines over many origins also only see the past
    y2 = y.copy()
    y2[p:] = -999
    np.testing.assert_array_equal(baseline_matrix("seasonal_naive_week", y, np.array([p])),
                                  baseline_matrix("seasonal_naive_week", y2, np.array([p])))


def test_targets_align_with_forecast_window_and_keep_missing_as_nan():
    idx = _grid()
    y = np.arange(len(idx), dtype=float)
    y[24 * 10 + 5] = np.nan
    Y = targets(y, np.array([24 * 10]))
    assert Y.shape == (1, 24)
    assert Y[0, 0] == 24 * 10 and np.isnan(Y[0, 5]) and Y[0, 23] == 24 * 10 + 23
    tt = target_times(idx[24 * 10])
    assert tt[0] == idx[24 * 10] and tt[23] == idx[24 * 10 + 23]


def test_feature_builder_refuses_short_or_unfilled_history():
    idx = _grid()
    with pytest.raises(InsufficientHistory):
        origin_features(np.ones(100), np.zeros(100), idx[100], TZ)
    h = np.ones(200)
    h[-3] = np.nan
    with pytest.raises(ValueError):
        origin_features(h, np.zeros(200), idx[200], TZ)


def test_causal_fill_uses_only_the_past_and_train_profile():
    idx = _grid(24 * 21)
    y = np.random.default_rng(1).random(len(idx)) + 1
    y[5] = np.nan              # first week -> profile
    y[200] = np.nan            # later -> value 168 h earlier
    y[200 + 168] = np.nan      # chained: uses the already-filled 200
    prof = fit_profile(y, idx, idx[24 * 14], TZ)
    f = causal_fill(y, idx, prof, TZ)
    assert f.n_profile == 1 and f.n_lag168 == 2
    assert f.filled[200] == y[200 - 168]
    assert f.filled[368] == f.filled[200]
    # future corruption cannot change any filled past value
    y2 = y.copy()
    y2[300:] = 50.0
    f2 = causal_fill(y2, idx, prof, TZ)
    np.testing.assert_array_equal(f.filled[:300], f2.filled[:300])
    assert f.imputed.sum() == 3


def test_fill_profile_ignores_non_training_hours():
    idx = _grid(24 * 21)
    y = np.ones(len(idx))
    y[24 * 14:] = 100.0        # "validation/test" values must not influence the profile
    prof = fit_profile(y, idx, idx[24 * 14], TZ)
    assert np.allclose(prof, 1.0)
