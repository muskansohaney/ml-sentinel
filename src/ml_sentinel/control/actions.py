
from __future__ import annotations

from dataclasses import dataclass

from ml_sentinel.policy.engine import PolicyAction


@dataclass(frozen=True)
class ActionResult:
    """Result returned after executing a control-plane action."""

    action: PolicyAction
    status: str
    message: str


class ActionExecutor:
    """
    Execute policy actions safely.

    The first version operates in dry-run mode by default.
    It records what ML Sentinel would do without changing
    production state.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run

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

        # Real retraining will be implemented later.
        return ActionResult(
            action=PolicyAction.RETRAIN,
            status="NOT_IMPLEMENTED",
            message="Retraining workflow is not implemented yet.",
        )

    def _rollback(self) -> ActionResult:
        if self.dry_run:
            return ActionResult(
                action=PolicyAction.ROLLBACK,
                status="DRY_RUN",
                message="Rollback workflow would be started.",
            )

        # Real rollback will be implemented later.
        return ActionResult(
            action=PolicyAction.ROLLBACK,
            status="NOT_IMPLEMENTED",
            message="Rollback workflow is not implemented yet.",
        )

    def _block_deployment(self) -> ActionResult:
        if self.dry_run:
            return ActionResult(
                action=PolicyAction.BLOCK_DEPLOYMENT,
                status="DRY_RUN",
                message="Deployment would be blocked.",
            )

        # Real deployment blocking will be implemented later.
        return ActionResult(
            action=PolicyAction.BLOCK_DEPLOYMENT,
            status="NOT_IMPLEMENTED",
            message="Deployment blocking is not implemented yet.",
        )
