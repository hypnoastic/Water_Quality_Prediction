import numpy as np
import pytest

from water_quality.evaluation import interval_metrics, intervals, metrics, residual_quantile


def test_metrics_use_index_units():
    score = metrics(np.array([0, 2, 4]), np.array([1, 2, 3]))
    assert score["mae"] == pytest.approx(2 / 3)
    assert score["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert score["r2"] == pytest.approx(0.75)


def test_calibration_uses_finite_sample_order_statistic():
    # n=9, 90%: ceil((9+1)*.9)=9, so choose maximum, not interpolated p90.
    assert residual_quantile(np.arange(9), np.zeros(9), 0.1) == 8
    assert residual_quantile(np.array([1]), np.array([0]), 0.1) == float("inf")
    with pytest.raises(ValueError):
        residual_quantile(np.array([1]), np.array([0]), 0)


def test_intervals_respect_bounded_target_and_count_coverage():
    prediction = np.array([1, 99])
    lo, hi = intervals(prediction, 3)
    np.testing.assert_array_equal(lo, [0, 96])
    np.testing.assert_array_equal(hi, [4, 100])
    assert interval_metrics(np.array([0, 95]), prediction, 3)["coverage"] == 0.5
