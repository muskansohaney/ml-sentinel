import pandas as pd

from ml_sentinel.monitoring.service import (
    collect_monitoring_snapshot,
    load_and_collect_monitoring_snapshot,
)
from ml_sentinel.monitoring.signals import QualitySnapshot
from ml_sentinel.policy.engine import PolicyAction


def make_reference_data():
    return pd.DataFrame(
        {
            "temperature": [50, 51, 49, 50, 52],
            "pressure": [100, 101, 99, 100, 102],
            "vibration": [0.5, 0.51, 0.49, 0.5, 0.52],
            "load": [50, 51, 49, 50, 52],
            "target": [0, 1, 0, 1, 0],
        }
    )


def make_production_data():
    return pd.DataFrame(
        {
            "temperature": [70, 72, 71, 73, 74],
            "pressure": [120, 122, 121, 123, 124],
            "vibration": [0.8, 0.82, 0.81, 0.83, 0.84],
            "load": [70, 72, 71, 73, 74],
            "target": [0, 1, 1, 1, 0],
        }
    )


def test_collect_monitoring_snapshot_detects_drift():
    snapshot = collect_monitoring_snapshot(
        make_reference_data(),
        make_production_data(),
        QualitySnapshot(correct=8, incorrect=2),
    )

    assert snapshot.signals.drift_detected is True
    assert snapshot.drifted_feature_count > 0
    assert snapshot.signals.model_quality == 0.8


def test_collect_monitoring_snapshot_combines_latency():
    snapshot = collect_monitoring_snapshot(
        make_reference_data(),
        make_reference_data(),
        QualitySnapshot(correct=9, incorrect=1),
        latency_ms=750.0,
    )

    assert snapshot.signals.latency_ms == 750.0
    assert snapshot.signals.model_quality == 0.9


def test_drift_causes_retrain_decision():
    snapshot = collect_monitoring_snapshot(
        make_reference_data(),
        make_production_data(),
        QualitySnapshot(correct=9, incorrect=1),
    )

    assert snapshot.signals.drift_detected is True


def test_load_and_collect_monitoring_snapshot(tmp_path):
    reference_path = tmp_path / "reference.csv"
    production_path = tmp_path / "production.csv"

    make_reference_data().to_csv(reference_path, index=False)
    make_production_data().to_csv(production_path, index=False)

    snapshot = load_and_collect_monitoring_snapshot(
        reference_path,
        production_path,
        QualitySnapshot(correct=7, incorrect=3),
    )

    assert snapshot.signals.drift_detected is True
    assert snapshot.signals.model_quality == 0.7
