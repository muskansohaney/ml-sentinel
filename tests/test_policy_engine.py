from ml_sentinel.policy.engine import (
    PolicyAction,
    ReliabilitySignals,
    evaluate_policy,
)


def test_healthy_system_keeps_current_model():
    signals = ReliabilitySignals()

    action = evaluate_policy(signals)

    assert action == PolicyAction.KEEP


def test_data_drift_triggers_retraining():
    signals = ReliabilitySignals(
        drift_detected=True,
        drifted_feature_count=3,
        total_feature_count=4,
    )

    action = evaluate_policy(signals)

    assert action == PolicyAction.RETRAIN


def test_poor_model_quality_triggers_retraining():
    signals = ReliabilitySignals(
        model_quality=0.55,
        quality_threshold=0.60,
    )

    action = evaluate_policy(signals)

    assert action == PolicyAction.RETRAIN


def test_high_latency_triggers_rollback():
    signals = ReliabilitySignals(
        latency_ms=750.0,
        latency_threshold_ms=500.0,
    )

    action = evaluate_policy(signals)

    assert action == PolicyAction.ROLLBACK


def test_latency_at_threshold_is_allowed():
    signals = ReliabilitySignals(
        latency_ms=500.0,
        latency_threshold_ms=500.0,
    )

    action = evaluate_policy(signals)

    assert action == PolicyAction.KEEP


def test_policy_prioritizes_rollback_over_retraining():
    signals = ReliabilitySignals(
        drift_detected=True,
        model_quality=0.40,
        latency_ms=800.0,
    )

    action = evaluate_policy(signals)

    assert action == PolicyAction.ROLLBACK

from datetime import datetime, timezone

from ml_sentinel.data.generator import DataScenario, generate_data
from ml_sentinel.drift.detector import detect_drift
def test_drift_detector_triggers_policy_retraining():
    start_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    reference = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=42,
        start_time=start_time,
    )

    production = generate_data(
        n_samples=1000,
        scenario=DataScenario.DRIFT,
        seed=43,
        start_time=start_time,
    )

    drift_result = detect_drift(
        reference,
        production,
    )

    signals = ReliabilitySignals(
        drift_detected=drift_result["drift_detected"],
        drifted_feature_count=drift_result["drifted_feature_count"],
        total_feature_count=drift_result["total_feature_count"],
    )

    action = evaluate_policy(signals)

    assert drift_result["drift_detected"] is True
    assert action == PolicyAction.RETRAIN

def test_normal_data_keeps_current_model():
    start_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    reference = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=42,
        start_time=start_time,
    )

    production = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=43,
        start_time=start_time,
    )

    drift_result = detect_drift(
        reference,
        production,
    )

    signals = ReliabilitySignals(
        drift_detected=drift_result["drift_detected"],
        drifted_feature_count=drift_result["drifted_feature_count"],
        total_feature_count=drift_result["total_feature_count"],
    )

    action = evaluate_policy(signals)

    assert drift_result["drift_detected"] is False
    assert action == PolicyAction.KEEP