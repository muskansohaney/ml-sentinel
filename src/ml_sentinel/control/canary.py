from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanaryResult:
    """Result of evaluating a candidate model in canary mode."""

    decision: str
    candidate_version: str
    candidate_metric: float
    baseline_metric: float
    threshold: float


def evaluate_canary(
    candidate_version: str,
    candidate_metric: float,
    baseline_metric: float,
    threshold: float = 0.0,
) -> CanaryResult:
    """
    Evaluate a candidate model against the current production model.

    The candidate passes canary evaluation when its metric is not worse
    than the baseline by more than the configured threshold.

    Example:
        baseline = 0.80
        candidate = 0.79
        threshold = 0.02

        Difference = -0.01
        Candidate passes because the degradation is within tolerance.
    """

    metric_difference = candidate_metric - baseline_metric

    if metric_difference >= -threshold:
        decision = "PROMOTE"
    else:
        decision = "BLOCK"

    return CanaryResult(
        decision=decision,
        candidate_version=str(candidate_version),
        candidate_metric=candidate_metric,
        baseline_metric=baseline_metric,
        threshold=threshold,
    )
