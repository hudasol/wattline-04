"""Evaluation math against hand calculations; intervals; advisory; monitor; lock."""

from __future__ import annotations

import numpy as np
import pytest

from wattline.advisory import (HIGH, LOW, NORMAL, UNCERTAIN, AdvisoryRule, false_confidence,
                               hour_labels, label_matrix)
from wattline.evaluate import (coverage, flag_scores, improvement, mae, median_width, peak_hit_rate,
                               per_horizon, per_origin_mae, rmse)
from wattline.intervals import LogConformal, order_stat_quantile
from wattline.monitor import HOUR_NS, MonitorRef, batch_monitor, monitor_state
from wattline.pipeline import FrozenMismatch, _check_lock
from wattline.util import write_json


def test_mae_rmse_by_hand_with_missing_and_mask():
    y = np.array([[1.0, 2.0, np.nan, 4.0]])
    p = np.array([[2.0, 2.0, 9.0, 1.0]])
    assert mae(y, p) == pytest.approx((1 + 0 + 3) / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt((1 + 0 + 9) / 3))
    mask = np.array([[True, True, True, False]])
    assert mae(y, p, mask) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        mae(y, p, np.zeros_like(mask))


def test_improvement_sign_and_scale():
    assert improvement(0.5, 0.4) == pytest.approx(0.2)
    assert improvement(0.5, 0.6) == pytest.approx(-0.2)


def test_coverage_width_by_hand():
    y = np.array([[1.0, 2.0, 3.0, 4.0]])
    lo = np.array([[0.5, 2.5, 2.0, 3.0]])
    hi = np.array([[1.5, 3.0, 4.0, 3.5]])
    assert coverage(y, lo, hi) == pytest.approx(2 / 4)   # 1 in, 2 below, 3 in, 4 above
    assert median_width(lo, hi) == pytest.approx(np.median([1.0, 0.5, 2.0, 0.5]))


def test_flag_scores_by_hand():
    actual = np.array([[True, True, False, False, True]])
    flagged = np.array([[True, False, True, False, True]])
    s = flag_scores(actual, flagged, np.ones_like(actual))
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (2, 1, 1, 1)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)


def test_peak_hit_rate_and_per_origin():
    y = np.zeros((2, 24))
    y[0, 18] = 5
    y[1, 7] = 5
    p = np.zeros((2, 24))
    p[0, [17, 18, 19]] = [3, 2, 1]     # day 0: true peak in top-3
    p[1, [10, 11, 12]] = [3, 2, 1]     # day 1: miss
    m = np.ones_like(y, bool)
    assert peak_hit_rate(y, p, m, 3) == pytest.approx(0.5)
    np.testing.assert_allclose(per_origin_mae(y, p, m), np.abs(y - p).mean(axis=1))
    assert per_horizon(y, {"m": p}, m).shape == (24, 3)


def test_conformal_order_statistics():
    r = np.arange(1, 10, dtype=float)    # n = 9
    assert order_stat_quantile(r, 0.9) == 9.0    # ceil(10*0.9) = 9th smallest
    assert order_stat_quantile(r, 0.1) == 1.0    # floor(10*0.1) = 1st smallest
    assert order_stat_quantile(r, 0.5) == 5.0


def test_log_conformal_bounds_are_ordered_nonnegative_and_widen_only():
    rng = np.random.default_rng(0)
    yhat = rng.uniform(0.2, 3, 2000)
    y = yhat * rng.lognormal(0, 0.3, 2000)
    c = LogConformal.fit(y, yhat, extra_levels=[0.7])
    lo, hi = c.bounds(yhat)
    assert (lo >= 0).all() and (lo <= hi).all()
    assert 0.77 <= coverage(y[None], lo[None], hi[None]) <= 0.83
    lo2, hi2 = c.bounds(yhat, scale=2.0)
    assert (lo2 <= lo + 1e-12).all() and (hi2 >= hi - 1e-12).all()
    assert (c.quantile(yhat, 0.7) <= hi).all()


def test_labels_vectorised_equals_scalar_and_uncertain_never_says_normal_or_low():
    rule = AdvisoryRule(high_kwh=2.0, low_kwh=0.5, flag_level=0.7, high_quantile=0.9, low_quantile=0.25)
    fq = np.array([[2.5, 1.0, 0.2]])
    up = np.array([[3.0, 1.5, 0.4]])
    assert hour_labels(fq[0], up[0], rule, "OK") == [HIGH, NORMAL, LOW]
    assert hour_labels(fq[0], up[0], rule, "UNCERTAIN") == [HIGH, UNCERTAIN, UNCERTAIN]
    np.testing.assert_array_equal(label_matrix(fq, up, rule, np.array([False]))[0], [HIGH, NORMAL, LOW])
    np.testing.assert_array_equal(label_matrix(fq, up, rule, np.array([True]))[0], [HIGH, UNCERTAIN, UNCERTAIN])


def test_false_confidence_counts_confident_but_wrong_hours():
    y = np.array([[3.0, 1.0, 0.2, 2.5]])
    labels = np.array([[NORMAL, NORMAL, LOW, UNCERTAIN]])
    upper = np.array([[2.0, 1.5, 0.4, 2.0]])
    fc = false_confidence(y, labels, upper, high_kwh=2.4, mask=np.ones_like(y, bool))
    assert fc["confident_hours"] == 3 and fc["confident_but_wrong"] == 1   # only the 3.0 kWh "NORMAL"


def test_monitor_widens_and_alarms_only_on_real_degradation():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.5, 2, (7, 24))
    y_val = p * rng.lognormal(0, 0.2, p.shape)
    ref = MonitorRef.fit(y_val, p)
    calm = monitor_state(p * rng.lognormal(0, 0.2, p.shape), p, ref, 72, 1.5, 3.0)
    assert calm.active and not calm.alarm and 1.0 <= calm.scale < 1.5
    wild = monitor_state(p * rng.lognormal(0, 0.8, p.shape), p, ref, 72, 1.5, 3.0)
    assert wild.alarm and wild.scale > 1.5 and wild.scale <= 3.0
    few = monitor_state(np.full((7, 24), np.nan), p, ref, 72, 1.5, 3.0)
    assert not few.active and few.scale == 1.0


def test_batch_monitor_never_uses_hours_that_end_after_the_issue_time():
    n = 9
    origin_ns = np.arange(n) * 24 * HOUR_NS
    target_ns = origin_ns[:, None] + np.arange(24)[None] * HOUR_NS
    target_ns[3, 23] = origin_ns[4] + 0          # spring-DST style: last hour ends 1 h after next issue
    p = np.ones((n, 24))
    y = np.ones((n, 24))
    y[3, 23] = 1000.0                             # a huge error hidden in that hour
    ref = MonitorRef(val_mae=0.1, val_halfwidth=0.1, val_median_resid=0.0)
    st = batch_monitor(y, p, target_ns, origin_ns, ref, 7, 24, 1.5, 3.0)
    assert not st[4].alarm                        # the not-yet-finished hour was masked
    assert st[5].alarm                            # one day later it is legitimately known


def test_final_eval_lock_refuses_changed_decisions(tmp_path):
    manifest = {"decisions_sha256": "a" * 64, "numeric_fingerprint": {"high_kwh": 2.0, "levels": {"0.1": -0.4}}}
    lock = tmp_path / "lock.json"
    assert _check_lock(manifest, lock) is None             # first scoring allowed
    write_json(manifest | {"first_scored_utc": "x", "headline": {}}, lock)
    assert _check_lock(manifest, lock) is not None         # identical -> regeneration allowed
    near = manifest | {"numeric_fingerprint": {"high_kwh": 2.0 + 1e-12, "levels": {"0.1": -0.4}}}
    assert _check_lock(near, lock) is not None             # float noise tolerated
    with pytest.raises(FrozenMismatch):
        _check_lock(manifest | {"decisions_sha256": "b" * 64}, lock)
    with pytest.raises(FrozenMismatch):
        _check_lock(manifest | {"numeric_fingerprint": {"high_kwh": 2.2, "levels": {"0.1": -0.4}}}, lock)
