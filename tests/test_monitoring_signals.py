from ml_sentinel.monitoring.signals import (
    QualitySnapshot,
    build_reliability_signals,
)


def test_quality_snapshot_calculates_accuracy():
    quality = QualitySnapshot(correct=80, incorrect=20)

    assert quality.accuracy == 0.8


def test_quality_snapshot_without_predictions_returns_none():
    quality = QualitySnapshot()

    assert quality.accuracy is None


def test_build_reliability_signals():
    quality = QualitySnapshot(correct=75, incorrect=25)

    signals = build_reliability_signals(
        quality,
        drift_detected=True,
        drifted_feature_count=2,
        total_feature_count=4,
        latency_ms=120.0,
    )

    assert signals.model_quality == 0.75
    assert signals.drift_detected is True
    assert signals.drifted_feature_count == 2
    assert signals.total_feature_count == 4
    assert signals.latency_ms == 120.0


def test_no_quality_data_is_preserved_as_none():
    signals = build_reliability_signals(
        QualitySnapshot(),
        latency_ms=100.0,
    )

    assert signals.model_quality is None


from ml_sentinel.policy.engine import PolicyAction, evaluate_policy


def test_poor_quality_triggers_retraining():
    signals = build_reliability_signals(
        QualitySnapshot(correct=50, incorrect=50),
    )

    assert evaluate_policy(signals) == PolicyAction.RETRAIN


def test_high_latency_triggers_rollback():
    signals = build_reliability_signals(
        QualitySnapshot(correct=90, incorrect=10),
        latency_ms=600.0,
    )

    assert evaluate_policy(signals) == PolicyAction.ROLLBACK


def test_drift_triggers_retraining():
    signals = build_reliability_signals(
        QualitySnapshot(correct=90, incorrect=10),
        drift_detected=True,
        drifted_feature_count=2,
        total_feature_count=4,
    )

    assert evaluate_policy(signals) == PolicyAction.RETRAIN


def test_healthy_signals_keep_model():
    signals = build_reliability_signals(
        QualitySnapshot(correct=90, incorrect=10),
        latency_ms=100.0,
    )

    assert evaluate_policy(signals) == PolicyAction.KEEP
