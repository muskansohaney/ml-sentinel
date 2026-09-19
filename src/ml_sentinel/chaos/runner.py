from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ml_sentinel.control.controller import (
    ControlDecision,
    ReliabilityController,
)
from ml_sentinel.data.generator import DataScenario, generate_data
from ml_sentinel.monitoring.service import MonitoringSnapshot
from ml_sentinel.monitoring.signals import QualitySnapshot
from ml_sentinel.monitoring.service import collect_monitoring_snapshot


@dataclass(frozen=True)
class ChaosResult:
    """Result of an intentionally injected production failure."""

    scenario: DataScenario
    data: pd.DataFrame


def inject_data_drift(
    n_samples: int = 1000,
    seed: int | None = None,
) -> ChaosResult:
    """Generate production data with intentional distribution drift."""

    data = generate_data(
        n_samples=n_samples,
        scenario=DataScenario.DRIFT,
        seed=seed,
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    return ChaosResult(
        scenario=DataScenario.DRIFT,
        data=data,
    )


def run_drift_chaos(
    reference_data: pd.DataFrame,
    *,
    n_samples: int = 1000,
    seed: int | None = None,
    quality: QualitySnapshot | None = None,
    latency_ms: float | None = None,
    controller: ReliabilityController | None = None,
) -> ControlDecision:
    """Inject data drift and run it through the reliability control loop."""

    chaos = inject_data_drift(
        n_samples=n_samples,
        seed=seed,
    )

    monitoring_quality = quality or QualitySnapshot()

    snapshot = collect_monitoring_snapshot(
        reference_data=reference_data,
        production_data=chaos.data,
        quality=monitoring_quality,
        latency_ms=latency_ms,
    )

    reliability_controller = controller or ReliabilityController()

    return reliability_controller.monitor_and_act(snapshot)