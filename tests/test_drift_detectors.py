import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from industrial_evolution.config import MonitoringConfig
from industrial_evolution.monitoring.feature_drift import (
    FeatureDriftDetector,
    _ks_statistic,
    population_stability_index,
)
from industrial_evolution.monitoring.performance_drift import PerformanceDriftDetector


def test_shifted_features_have_higher_drift_score():
    rng = np.random.default_rng(7)
    reference = pd.DataFrame(rng.normal(size=(1200, 4)), columns=list("abcd"))
    stable = pd.DataFrame(rng.normal(size=(1200, 4)), columns=list("abcd"))
    shifted = stable.copy()
    shifted["a"] += 2.0
    shifted["b"] *= 2.2
    config = MonitoringConfig(
        psi_threshold=0.18,
        ks_threshold=0.18,
        performance_drop_threshold=0.06,
        label_shift_threshold=0.12,
        consecutive_windows=2,
        reference_update_rate=0.15,
    )

    detector = FeatureDriftDetector(reference, config)
    stable_result = detector.detect(stable)
    shifted_result = detector.detect(shifted)

    assert shifted_result.is_drift
    assert shifted_result.overall_score > stable_result.overall_score
    assert {"a", "b"}.intersection(shifted_result.top_features)


def test_fast_ks_statistic_matches_scipy():
    samples = [
        (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 4])),
        (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 2])),
        (np.array([0, 1, np.nan, 3]), np.array([0, np.nan, 2, 4])),
    ]

    for reference, current in samples:
        finite_reference = np.sort(reference[np.isfinite(reference)])
        expected = ks_2samp(reference, current, nan_policy="omit").statistic
        assert np.isclose(_ks_statistic(finite_reference, current), expected)


def test_reference_cache_refreshes_after_update():
    rng = np.random.default_rng(19)
    reference = pd.DataFrame(rng.normal(size=(500, 2)), columns=["a", "b"])
    current = pd.DataFrame(rng.normal(2.0, 1.0, size=(500, 2)), columns=["a", "b"])
    config = MonitoringConfig(
        psi_threshold=0.18,
        ks_threshold=0.18,
        performance_drop_threshold=0.06,
        label_shift_threshold=0.12,
        consecutive_windows=2,
        reference_update_rate=0.50,
    )
    detector = FeatureDriftDetector(reference, config)

    detector.update_reference(current)
    result = detector.detect(current)
    expected_ks = ks_2samp(detector.reference["a"], current["a"]).statistic
    expected_psi = population_stability_index(detector.reference["a"], current["a"])

    assert np.isclose(result.per_feature["a"]["ks"], expected_ks, atol=1e-6)
    assert np.isclose(result.per_feature["a"]["psi"], expected_psi, atol=1e-6)


def test_all_non_finite_values_are_data_quality_failures():
    reference = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100)})
    config = MonitoringConfig(
        psi_threshold=0.18,
        ks_threshold=0.18,
        performance_drop_threshold=0.06,
        label_shift_threshold=0.12,
        consecutive_windows=1,
        reference_update_rate=0.15,
    )
    detector = FeatureDriftDetector(reference, config)

    nan_result = detector.detect(pd.DataFrame({"a": [np.nan] * 30, "b": [np.nan] * 30}))
    inf_result = detector.detect(pd.DataFrame({"a": [np.inf] * 30, "b": [-np.inf] * 30}))

    assert not nan_result.data_quality_valid
    assert not inf_result.data_quality_valid
    assert nan_result.data_quality_issues
    assert inf_result.data_quality_issues


def test_feature_reference_snapshot_restores_previous_distribution():
    rng = np.random.default_rng(31)
    reference = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    shifted = pd.DataFrame(rng.normal(5.0, 1.0, size=(200, 2)), columns=["a", "b"])
    config = MonitoringConfig(
        psi_threshold=0.18,
        ks_threshold=0.18,
        performance_drop_threshold=0.06,
        label_shift_threshold=0.12,
        consecutive_windows=1,
        reference_update_rate=0.5,
    )
    detector = FeatureDriftDetector(reference, config)
    snapshot = detector.snapshot_state()
    detector.update_reference(shifted)
    detector.restore_state(snapshot)

    pd.testing.assert_frame_equal(detector.reference, reference)


def test_regression_performance_drift_detects_rmse_rise():
    # 负 RMSE 分数：基线 -25，阈值按 RMSE 量级取 2.0。稳定时不应误报，持续变差才触发。
    detector = PerformanceDriftDetector(baseline_score=-25.0, drop_threshold=2.0)
    for _ in range(3):
        result = detector.update(-25.0)
    assert not result.is_drift

    result = detector.update(-30.0)  # 滚动 drop 1.67，尚不足阈值
    assert not result.is_drift
    result = detector.update(-30.0)  # 滚动 drop 3.33，超过阈值
    assert result.is_drift
    assert result.score_drop >= 2.0


def test_regression_performance_drift_stable_scores_do_not_alarm():
    # 回归下 Page-Hinkley 不再用 1.0 - score，稳定负 RMSE 不应触发 PH 报警。
    detector = PerformanceDriftDetector(baseline_score=-30.0, drop_threshold=2.0)
    for _ in range(5):
        result = detector.update(-30.0)
    assert not result.is_drift
    assert not result.page_hinkley_alarm
