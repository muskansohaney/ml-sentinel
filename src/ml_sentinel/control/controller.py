from __future__ import annotations

from dataclasses import dataclass

from ml_sentinel.control.actions import ActionExecutor, ActionResult
from ml_sentinel.monitoring.audit import record_decision
from ml_sentinel.monitoring.metrics import (
    DRIFT_DETECTED,
    DRIFTED_FEATURE_COUNT,
    MODEL_LATENCY_MS,
    MODEL_QUALITY,
    RELIABILITY_ACTION,
)
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
        """Evaluate reliability signals, publish metrics, execute, and audit."""

        # Publish current reliability signals.
        DRIFT_DETECTED.set(
            1 if signals.drift_detected else 0
        )

        DRIFTED_FEATURE_COUNT.set(
            signals.drifted_feature_count
        )

        if signals.model_quality is not None:
            MODEL_QUALITY.set(
                signals.model_quality
            )

        if signals.latency_ms is not None:
            MODEL_LATENCY_MS.set(
                signals.latency_ms
            )

        action = evaluate_policy(signals)

        # Reset action gauges so only the latest action is active.
        for policy_action in PolicyAction:
            RELIABILITY_ACTION.labels(
                action=policy_action.value,
            ).set(0)

        RELIABILITY_ACTION.labels(
            action=action.value,
        ).set(1)

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
