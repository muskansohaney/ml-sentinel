from prometheus_client import REGISTRY

from ml_sentinel.monitoring.metrics import (
    DRIFT_DETECTED,
    DRIFTED_FEATURE_COUNT,
    MODEL_LATENCY_MS,
    MODEL_QUALITY,
    MONITORING_DRIFT_SHARE,
    MONITORING_EVALUATED,
    MONITORING_OBSERVATION_COUNT,
    MONITORING_QUALITY_OBSERVATION_COUNT,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    QUALITY_COUNT,
    RELIABILITY_ACTION,
)


def test_prediction_metrics_are_registered():
    assert PREDICTION_COUNT is not None
    assert PREDICTION_LATENCY is not None
    assert PREDICTION_ERRORS is not None
    assert QUALITY_COUNT is not None


def test_reliability_metrics_are_registered():
    assert DRIFT_DETECTED is not None
    assert DRIFTED_FEATURE_COUNT is not None
    assert MODEL_QUALITY is not None
    assert MODEL_LATENCY_MS is not None
    assert RELIABILITY_ACTION is not None
    assert MONITORING_OBSERVATION_COUNT is not None
    assert MONITORING_QUALITY_OBSERVATION_COUNT is not None
    assert MONITORING_DRIFT_SHARE is not None
    assert MONITORING_EVALUATED is not None


def test_reliability_metrics_can_record_values():
    DRIFT_DETECTED.set(1)
    DRIFTED_FEATURE_COUNT.set(2)
    MODEL_QUALITY.set(0.82)
    MODEL_LATENCY_MS.set(125.5)

    RELIABILITY_ACTION.labels(
        action="RETRAIN",
    ).set(1)

    assert DRIFT_DETECTED._value.get() == 1
    assert DRIFTED_FEATURE_COUNT._value.get() == 2
    assert MODEL_QUALITY._value.get() == 0.82
    assert MODEL_LATENCY_MS._value.get() == 125.5
    assert (
        RELIABILITY_ACTION.labels(
            action="RETRAIN",
        )._value.get()
        == 1
    )


def test_prometheus_registry_contains_reliability_metrics():
    metric_names = {
        metric.name
        for metric in REGISTRY.collect()
    }

    assert "ml_sentinel_drift_detected" in metric_names
    assert "ml_sentinel_drifted_feature_count" in metric_names
    assert "ml_sentinel_model_quality" in metric_names
    assert "ml_sentinel_model_latency_ms" in metric_names
    assert "ml_sentinel_reliability_action" in metric_names
    assert "ml_sentinel_monitoring_observation_count" in metric_names
    assert "ml_sentinel_monitoring_quality_observation_count" in metric_names
    assert "ml_sentinel_monitoring_drift_share" in metric_names
    assert "ml_sentinel_monitoring_evaluated" in metric_names
