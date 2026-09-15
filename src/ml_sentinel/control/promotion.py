from __future__ import annotations

from dataclasses import dataclass

from ml_sentinel.models.validation import (
    ValidationResult,
    validate_registered_models,
)
from ml_sentinel.registry.model_registry import promote_model


@dataclass(frozen=True)
class PromotionResult:
    """Result of validating and potentially promoting a model."""

    decision: str
    candidate_version: str
    validation: ValidationResult


def validate_and_promote(
    candidate_version: str,
    current_version: str | None,
    metric_name: str = "f1",
) -> PromotionResult:
    """
    Validate a candidate model and promote it when approved.

    A blocked candidate is never assigned the production alias.
    """

    validation = validate_registered_models(
        candidate_version=candidate_version,
        current_version=current_version,
        metric_name=metric_name,
    )

    if validation.decision == "PROMOTE":
        promote_model(candidate_version)

    return PromotionResult(
        decision=validation.decision,
        candidate_version=candidate_version,
        validation=validation,
    )
