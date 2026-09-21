from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_sentinel.control.promotion import (
    PromotionResult,
    validate_and_promote,
)
from ml_sentinel.models.train import train_and_register
from ml_sentinel.registry.model_registry import (
    MODEL_NAME,
    get_latest_version,
)


@dataclass(frozen=True)
class SelfHealingResult:
    """Result of the automated retraining and promotion workflow."""

    training: dict[str, Any]
    promotion: PromotionResult


def retrain_and_validate(
    training_data_path: str | Path,
    model_output_path: str | Path = "models/model.joblib",
    metric_name: str = "f1",
) -> SelfHealingResult:
    """
    Train a candidate model, compare it with the current model,
    and promote it only when validation approves it.
    """

    current_version = get_latest_version()

    training_result = train_and_register(
        data_path=training_data_path,
        output_path=model_output_path,
    )

    candidate_version = training_result["model_version"]

    promotion_result = validate_and_promote(
        candidate_version=candidate_version,
        current_version=(
            str(current_version.version)
            if current_version is not None
            else None
        ),
        metric_name=metric_name,
    )

    return SelfHealingResult(
        training=training_result,
        promotion=promotion_result,
    )
