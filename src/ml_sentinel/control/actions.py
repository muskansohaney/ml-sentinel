from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_sentinel.control.self_healing import retrain_and_validate
from ml_sentinel.policy.engine import PolicyAction
from ml_sentinel.registry.model_registry import (
    get_previous_version,
    get_production_version,
    promote_model,
)


@dataclass(frozen=True)
class ActionResult:
    """Result returned after executing a control-plane action."""

    action: PolicyAction
    status: str
    message: str
    details: dict[str, Any] | None = None


class ActionExecutor:
    """
    Execute policy actions.

    Dry-run mode is enabled by default. When disabled,
    supported actions are executed against the ML system.
    """

    def __init__(
        self,
        dry_run: bool = True,
        training_data_path: str | Path = "data/production/production.csv",
        model_output_path: str | Path = "models/model.joblib",
    ) -> None:
        self.dry_run = dry_run
        self.training_data_path = Path(training_data_path)
        self.model_output_path = Path(model_output_path)

    def execute(self, action: PolicyAction) -> ActionResult:
        """Execute a policy action."""

        if action == PolicyAction.KEEP:
            return self._keep()

        if action == PolicyAction.RETRAIN:
            return self._retrain()

        if action == PolicyAction.ROLLBACK:
            return self._rollback()

        if action == PolicyAction.BLOCK_DEPLOYMENT:
            return self._block_deployment()

        raise ValueError(f"Unsupported policy action: {action}")

    def _keep(self) -> ActionResult:
        return ActionResult(
            action=PolicyAction.KEEP,
            status="NO_OP",
            message="Current model remains active.",
        )

    def _retrain(self) -> ActionResult:
        if self.dry_run:
            return ActionResult(
                action=PolicyAction.RETRAIN,
                status="DRY_RUN",
                message="Retraining workflow would be started.",
            )

        if not self.training_data_path.exists():
            return ActionResult(
                action=PolicyAction.RETRAIN,
                status="FAILED",
                message=(
                    "Retraining failed because the training dataset "
                    f"does not exist: {self.training_data_path}"
                ),
            )

        try:
            result = retrain_and_validate(
                training_data_path=self.training_data_path,
                model_output_path=self.model_output_path,
            )
        except Exception as exc:
            return ActionResult(
                action=PolicyAction.RETRAIN,
                status="FAILED",
                message=f"Retraining failed: {exc}",
            )

        return ActionResult(
            action=PolicyAction.RETRAIN,
            status="COMPLETED",
            message=(
                "Retraining completed and the candidate model "
                f"was {result.promotion.decision.lower()}."
            ),
            details={
                "training": result.training,
                "promotion": {
                    "decision": result.promotion.decision,
                    "candidate_version": result.promotion.candidate_version,
                    "validation": {
                        "candidate_metric": (
                            result.promotion.validation.candidate_metric
                        ),
                        "current_metric": (
                            result.promotion.validation.current_metric
                        ),
                        "metric_name": (
                            result.promotion.validation.metric_name
                        ),
                    },
                },
            },
        )

    def _rollback(self) -> ActionResult:
        if self.dry_run:
            return ActionResult(
                action=PolicyAction.ROLLBACK,
                status="DRY_RUN",
                message="Rollback workflow would be started.",
            )

        try:
            current_version = get_production_version()

            if current_version is None:
                return ActionResult(
                    action=PolicyAction.ROLLBACK,
                    status="FAILED",
                    message="Rollback failed: no production model is assigned.",
                )

            previous_version = get_previous_version(
                current_version=str(current_version.version),
            )

            if previous_version is None:
                return ActionResult(
                    action=PolicyAction.ROLLBACK,
                    status="FAILED",
                    message=(
                        "Rollback failed: no previous model version "
                        f"exists before production version {current_version.version}."
                    ),
                )

            promoted_version = promote_model(
                model_version=str(previous_version.version),
            )

            return ActionResult(
                action=PolicyAction.ROLLBACK,
                status="COMPLETED",
                message=(
                    f"Rolled back production model from version "
                    f"{current_version.version} to version "
                    f"{promoted_version.version}."
                ),
                details={
                    "previous_production_version": str(current_version.version),
                    "rollback_version": str(promoted_version.version),
                },
            )

        except Exception as exc:
            return ActionResult(
                action=PolicyAction.ROLLBACK,
                status="FAILED",
                message=f"Rollback failed: {exc}",
            )

    def _block_deployment(self) -> ActionResult:
        if self.dry_run:
            return ActionResult(
                action=PolicyAction.BLOCK_DEPLOYMENT,
                status="DRY_RUN",
                message="Deployment would be blocked.",
            )

        return ActionResult(
            action=PolicyAction.BLOCK_DEPLOYMENT,
            status="NOT_IMPLEMENTED",
            message="Deployment blocking is not implemented yet.",
        )
