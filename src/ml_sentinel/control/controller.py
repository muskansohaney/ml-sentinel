from __future__ import annotations

from dataclasses import dataclass

from ml_sentinel.control.actions import ActionExecutor, ActionResult
from ml_sentinel.monitoring.audit import record_decision
from ml_sentinel.monitoring.service import MonitoringSnapshot
from ml_sentinel.policy.engine import (
    ReliabilitySignals,
    PolicyAction,
    evaluate_policy,
)


@dataclass(frozen=True)
class ControlDecision:
    """Policy decision and resulting control action."""

    action: PolicyAction
    result: ActionResult


class ReliabilityController:
    """Connect reliability signals, policy, and control actions."""

    def __init__(
        self,
        executor: ActionExecutor | None = None,
        audit_log_path: str = "logs/decisions.jsonl",
    ) -> None:
        self.executor = executor or ActionExecutor()
        self.audit_log_path = audit_log_path

    def evaluate_and_act(
        self,
        signals: ReliabilitySignals,
    ) -> ControlDecision:
        """Evaluate reliability signals, execute, and audit the decision."""

        action = evaluate_policy(signals)
        result = self.executor.execute(action)

        record_decision(
            signals,
            action,
            result,
            self.audit_log_path,
        )

        return ControlDecision(
            action=action,
            result=result,
        )

    def monitor_and_act(
        self,
        snapshot: MonitoringSnapshot,
    ) -> ControlDecision:
        """Execute the control loop using a monitoring snapshot."""

        return self.evaluate_and_act(snapshot.signals)
