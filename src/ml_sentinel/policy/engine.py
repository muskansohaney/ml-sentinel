from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PolicyAction(str, Enum):
    """Actions that ML Sentinel can recommend."""

    KEEP = "KEEP"
    RETRAIN = "RETRAIN"
    BLOCK_DEPLOYMENT = "BLOCK_DEPLOYMENT"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True)
class ReliabilitySignals:
    """Signals collected from the ML system."""

    drift_detected: bool = False
    drifted_feature_count: int = 0
    total_feature_count: int = 0

    latency_ms: float | None = None
    latency_threshold_ms: float = 500.0

    model_quality: float | None = None
    quality_threshold: float = 0.60


def evaluate_policy(signals: ReliabilitySignals) -> PolicyAction:
    """
    Convert reliability signals into a policy decision.

    Decision priority:
        1. Severe latency -> ROLLBACK
        2. Poor model quality -> RETRAIN
        3. Significant data drift -> RETRAIN
        4. Otherwise -> KEEP
    """

    # Severe serving degradation.
    if (
        signals.latency_ms is not None
        and signals.latency_ms > signals.latency_threshold_ms
    ):
        return PolicyAction.ROLLBACK

    # Model quality has degraded below the acceptable threshold.
    if (
        signals.model_quality is not None
        and signals.model_quality < signals.quality_threshold
    ):
        return PolicyAction.RETRAIN

    # Data distribution has changed significantly.
    if signals.drift_detected:
        return PolicyAction.RETRAIN

    return PolicyAction.KEEP

