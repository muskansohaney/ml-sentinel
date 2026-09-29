from ml_sentinel.control.self_healing import SelfHealingResult
from ml_sentinel.control.promotion import PromotionResult
from ml_sentinel.models.validation import ValidationResult
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
def test_controller_triggers_self_healing_retraining(
    tmp_path,
    monkeypatch,
):
    training_data = tmp_path / "production.csv"
    training_data.write_text("dummy")

    model_output = tmp_path / "model.joblib"

    expected_training = {
        "run_id": "test-run",
        "model_name": "ml-sentinel-model",
        "model_version": "101",
        "output_path": str(model_output),
        "metrics": {
            "f1": 0.82,
        },
    }

    def fake_retrain_and_validate(
        training_data_path,
        model_output_path,
    ):
        assert training_data_path == training_data
        assert model_output_path == model_output

        return SelfHealingResult(
            training=expected_training,
            promotion=PromotionResult(
                decision="PROMOTE",
                candidate_version="101",
                validation=ValidationResult(
                    decision="PROMOTE",
                    candidate_metric=0.82,
                    current_metric=0.75,
                    metric_name="f1",
                ),
            ),
        )

    monkeypatch.setattr(
        "ml_sentinel.control.actions.retrain_and_validate",
        fake_retrain_and_validate,
    )

    from ml_sentinel.control.actions import ActionExecutor

    executor = ActionExecutor(
        dry_run=False,
        training_data_path=training_data,
        model_output_path=model_output,
    )

    controller = ReliabilityController(
        executor=executor,
        audit_log_path=str(tmp_path / "decisions.jsonl"),
    )

    decision = controller.evaluate_and_act(
        ReliabilitySignals(
            model_quality=0.40,
            latency_ms=100.0,
        )
    )

    assert decision.action == PolicyAction.RETRAIN
    assert decision.result.action == PolicyAction.RETRAIN
    assert decision.result.status == "COMPLETED"

    assert decision.result.details["training"] == expected_training

    assert (
        decision.result.details["promotion"]["decision"]
        == "PROMOTE"
    )

    assert (
        decision.result.details["promotion"]["candidate_version"]
        == "101"
    )
def test_controller_publishes_reliability_metrics(tmp_path):
    from ml_sentinel.monitoring.metrics import (
        DRIFT_DETECTED,
        DRIFTED_FEATURE_COUNT,
        MODEL_LATENCY_MS,
        MODEL_QUALITY,
        RELIABILITY_ACTION,
    )

    controller = ReliabilityController(
        executor=FakeExecutor(),
        audit_log_path=str(tmp_path / "decisions.jsonl"),
    )

    decision = controller.evaluate_and_act(
        ReliabilitySignals(
            drift_detected=True,
            drifted_feature_count=3,
            total_feature_count=4,
            model_quality=0.42,
            latency_ms=150.0,
        )
    )

    assert decision.action == PolicyAction.RETRAIN

    assert DRIFT_DETECTED._value.get() == 1
    assert DRIFTED_FEATURE_COUNT._value.get() == 3
    assert MODEL_QUALITY._value.get() == 0.42
    assert MODEL_LATENCY_MS._value.get() == 150.0

    assert (
        RELIABILITY_ACTION.labels(
            action="RETRAIN",
        )._value.get()
        == 1
    )

    assert (
        RELIABILITY_ACTION.labels(
            action="KEEP",
        )._value.get()
        == 0
    )
