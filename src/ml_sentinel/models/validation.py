from __future__ import annotations

from dataclasses import dataclass

from ml_sentinel.registry.model_registry import get_model_metrics


@dataclass(frozen=True)
class ValidationResult:
    """Result of comparing a candidate model with the active model."""

    decision: str
    candidate_metric: float
    current_metric: float | None
    metric_name: str


def validate_candidate(
    candidate_metric: float,
    current_metric: float | None,
    metric_name: str = "f1",
) -> ValidationResult:
    """
    Decide whether a candidate model is good enough for promotion.

    A candidate is accepted when:
    - there is no current model, or
    - the candidate metric is at least as good as the current metric.
    """

    if current_metric is None:
        decision = "PROMOTE"
    elif candidate_metric >= current_metric:
        decision = "PROMOTE"
    else:
        decision = "BLOCK"

    return ValidationResult(
        decision=decision,
        candidate_metric=candidate_metric,
        current_metric=current_metric,
        metric_name=metric_name,
    )


def validate_registered_models(
    candidate_version: str,
    current_version: str | None,
    metric_name: str = "f1",
) -> ValidationResult:
    """
    Validate a registered candidate model against the current model.

    Metrics are retrieved from MLflow for both model versions.
    """

    candidate_metrics = get_model_metrics(candidate_version)

    if metric_name not in candidate_metrics:
        raise ValueError(
            f"Candidate model version {candidate_version} "
            f"does not contain metric '{metric_name}'."
        )

    candidate_metric = candidate_metrics[metric_name]

    if current_version is None:
        current_metric = None
    else:
        current_metrics = get_model_metrics(current_version)

        if metric_name not in current_metrics:
            raise ValueError(
                f"Current model version {current_version} "
                f"does not contain metric '{metric_name}'."
            )

        current_metric = current_metrics[metric_name]

    return validate_candidate(
        candidate_metric=candidate_metric,
        current_metric=current_metric,
        metric_name=metric_name,
    )