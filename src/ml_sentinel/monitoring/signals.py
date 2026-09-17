from __future__ import annotations

from dataclasses import dataclass

from ml_sentinel.policy.engine import ReliabilitySignals


@dataclass(frozen=True)
class QualitySnapshot:
    """Production prediction quality counts."""

    correct: int = 0
    incorrect: int = 0

    @property
    def accuracy(self) -> float | None:
        total = self.correct + self.incorrect

        if total == 0:
            return None

        return self.correct / total


def build_reliability_signals(
    quality: QualitySnapshot,
    *,
    drift_detected: bool = False,
    drifted_feature_count: int = 0,
    total_feature_count: int = 0,
    latency_ms: float | None = None,
    latency_threshold_ms: float = 500.0,
    quality_threshold: float = 0.60,
) -> ReliabilitySignals:
    """Convert monitoring observations into policy signals."""

    return ReliabilitySignals(
        drift_detected=drift_detected,
        drifted_feature_count=drifted_feature_count,
        total_feature_count=total_feature_count,
        latency_ms=latency_ms,
        latency_threshold_ms=latency_threshold_ms,
        model_quality=quality.accuracy,
        quality_threshold=quality_threshold,
    )
