from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_sentinel.drift.detector import detect_drift
from ml_sentinel.monitoring.signals import (
    QualitySnapshot,
    build_reliability_signals,
)
from ml_sentinel.policy.engine import ReliabilitySignals


@dataclass(frozen=True)
class MonitoringSnapshot:
    signals: ReliabilitySignals
    drifted_feature_count: int


def collect_monitoring_snapshot(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    quality: QualitySnapshot,
    *,
    latency_ms: float | None = None,
    latency_threshold_ms: float = 500.0,
    quality_threshold: float = 0.60,
) -> MonitoringSnapshot:
    drift_result = detect_drift(
        reference_data=reference_data,
        production_data=production_data,
    )

    signals = build_reliability_signals(
        quality,
        drift_detected=drift_result["drift_detected"],
        drifted_feature_count=drift_result["drifted_feature_count"],
        total_feature_count=drift_result["total_feature_count"],
        latency_ms=latency_ms,
        latency_threshold_ms=latency_threshold_ms,
        quality_threshold=quality_threshold,
    )

    return MonitoringSnapshot(
        signals=signals,
        drifted_feature_count=drift_result["drifted_feature_count"],
    )


def load_and_collect_monitoring_snapshot(
    reference_path: str | Path,
    production_path: str | Path,
    quality: QualitySnapshot,
    *,
    latency_ms: float | None = None,
    latency_threshold_ms: float = 500.0,
    quality_threshold: float = 0.60,
) -> MonitoringSnapshot:
    reference_data = pd.read_csv(reference_path)
    production_data = pd.read_csv(production_path)

    return collect_monitoring_snapshot(
        reference_data,
        production_data,
        quality,
        latency_ms=latency_ms,
        latency_threshold_ms=latency_threshold_ms,
        quality_threshold=quality_threshold,
    )
