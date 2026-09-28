from __future__ import annotations

from dataclasses import dataclass

from ml_sentinel.control.canary import evaluate_canary
from ml_sentinel.models.validation import (
    ValidationResult,
    validate_registered_models,
)
from ml_sentinel.registry.model_registry import promote_model


@dataclass(frozen=True)
class PromotionResult:
    """Result of validating, canary-testing, and potentially promoting a model."""

    decision: str
    candidate_version: str
    validation: ValidationResult
    canary_decision: str | None = None


def validate_and_promote(
    candidate_version: str,
    current_version: str | None,
    metric_name: str = "f1",
    canary_threshold: float = 0.0,
) -> PromotionResult:
    """
    Validate a candidate model, evaluate it in canary mode,
    and promote it only when it passes the allowed threshold.

    The canary threshold represents the maximum acceptable
    degradation relative to the current production model.
    """

    validation = validate_registered_models(
        candidate_version=candidate_version,
        current_version=current_version,
        metric_name=metric_name,
    )

    # No current production model exists.
    # The first valid candidate can be promoted directly.
    if current_version is None:
        promote_model(candidate_version)

        return PromotionResult(
            decision="PROMOTE",
            candidate_version=candidate_version,
            validation=validation,
            canary_decision="PROMOTE",
        )

    # Get the candidate and baseline metrics from validation.
    candidate_metric = validation.candidate_metric
    baseline_metric = validation.current_metric

    if baseline_metric is None:
        raise ValueError(
            "Current model metric is required for canary evaluation."
        )

    # Canary evaluation is intentionally separate from the
    # strict validation decision. This allows a small,
    # explicitly configured degradation.
    canary = evaluate_canary(
        candidate_version=candidate_version,
        candidate_metric=candidate_metric,
        baseline_metric=baseline_metric,
        threshold=canary_threshold,
    )

    if canary.decision == "PROMOTE":
        promote_model(candidate_version)

    return PromotionResult(
        decision=canary.decision,
        candidate_version=candidate_version,
        validation=validation,
        canary_decision=canary.decision,
    )
