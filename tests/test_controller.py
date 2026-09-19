from ml_sentinel.control.actions import ActionResult
from ml_sentinel.control.controller import ReliabilityController
from ml_sentinel.policy.engine import PolicyAction, ReliabilitySignals
from ml_sentinel.monitoring.service import MonitoringSnapshot

class FakeExecutor:
    def __init__(self):
        self.actions = []

    def execute(self, action):
        self.actions.append(action)

        return ActionResult(
            action=action,
            status="TEST",
            message="Test action executed.",
        )


def test_healthy_system_keeps_model():
    executor = FakeExecutor()
    controller = ReliabilityController(executor=executor)

    decision = controller.evaluate_and_act(
        ReliabilitySignals(
            model_quality=0.90,
            latency_ms=100.0,
        )
    )

    assert decision.action == PolicyAction.KEEP
    assert decision.result.action == PolicyAction.KEEP
    assert executor.actions == [PolicyAction.KEEP]


def test_poor_quality_triggers_retraining():
    executor = FakeExecutor()
    controller = ReliabilityController(executor=executor)

    decision = controller.evaluate_and_act(
        ReliabilitySignals(
            model_quality=0.40,
            latency_ms=100.0,
        )
    )

    assert decision.action == PolicyAction.RETRAIN
    assert decision.result.action == PolicyAction.RETRAIN
    assert executor.actions == [PolicyAction.RETRAIN]


def test_high_latency_triggers_rollback():
    executor = FakeExecutor()
    controller = ReliabilityController(executor=executor)

    decision = controller.evaluate_and_act(
        ReliabilitySignals(
            model_quality=0.90,
            latency_ms=700.0,
        )
    )

    assert decision.action == PolicyAction.ROLLBACK
    assert decision.result.action == PolicyAction.ROLLBACK
    assert executor.actions == [PolicyAction.ROLLBACK]


def test_controller_uses_real_executor_in_dry_run():
    from ml_sentinel.control.actions import ActionExecutor

    controller = ReliabilityController(
        executor=ActionExecutor(dry_run=True)
    )

    decision = controller.evaluate_and_act(
        ReliabilitySignals(
            model_quality=0.40,
            latency_ms=100.0,
        )
    )

    assert decision.action == PolicyAction.RETRAIN
    assert decision.result.action == PolicyAction.RETRAIN
    assert decision.result.status == "DRY_RUN"


def test_controller_records_audit_event(tmp_path):
    audit_log = tmp_path / "decisions.jsonl"

    controller = ReliabilityController(
        executor=FakeExecutor(),
        audit_log_path=str(audit_log),
    )

    decision = controller.evaluate_and_act(
        ReliabilitySignals(
            model_quality=0.40,
            latency_ms=100.0,
        )
    )

    assert decision.action == PolicyAction.RETRAIN
    assert audit_log.exists()

    content = audit_log.read_text()

    assert '"action": "RETRAIN"' in content
    assert '"status": "TEST"' in content
    assert '"model_quality": 0.4' in content
def test_controller_accepts_monitoring_snapshot(tmp_path):
    audit_log = tmp_path / "decisions.jsonl"

    controller = ReliabilityController(
        executor=FakeExecutor(),
        audit_log_path=str(audit_log),
    )

    snapshot = MonitoringSnapshot(
        signals=ReliabilitySignals(
            model_quality=0.40,
            latency_ms=100.0,
        ),
        drifted_feature_count=0,
    )

    decision = controller.monitor_and_act(snapshot)

    assert decision.action == PolicyAction.RETRAIN
    assert decision.result.status == "TEST"
