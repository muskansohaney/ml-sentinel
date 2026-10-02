from pathlib import Path

from ml_sentinel.control.actions import ActionExecutor
from ml_sentinel.control.promotion import PromotionResult
from ml_sentinel.control.self_healing import SelfHealingResult
from ml_sentinel.models.validation import ValidationResult
from datetime import datetime, timezone
from pathlib import Path
from ml_sentinel.data.generator import DataScenario, generate_data
from ml_sentinel.drift.detector import detect_drift
from ml_sentinel.policy.engine import (
    ReliabilitySignals,
    evaluate_policy,
)
from ml_sentinel.control.actions import ActionExecutor
from ml_sentinel.policy.engine import PolicyAction


def test_keep_action_is_no_op():
    executor = ActionExecutor()

    result = executor.execute(PolicyAction.KEEP)

    assert result.action == PolicyAction.KEEP
    assert result.status == "NO_OP"


def test_retrain_action_is_dry_run_by_default():
    executor = ActionExecutor()

    result = executor.execute(PolicyAction.RETRAIN)

    assert result.action == PolicyAction.RETRAIN
    assert result.status == "DRY_RUN"


def test_rollback_action_is_dry_run_by_default():
    executor = ActionExecutor()

    result = executor.execute(PolicyAction.ROLLBACK)

    assert result.action == PolicyAction.ROLLBACK
    assert result.status == "DRY_RUN"


def test_block_deployment_action_is_dry_run_by_default():
    executor = ActionExecutor()

    result = executor.execute(PolicyAction.BLOCK_DEPLOYMENT)

    assert result.action == PolicyAction.BLOCK_DEPLOYMENT
    assert result.status == "DRY_RUN"


def test_retrain_completes_when_not_dry_run(tmp_path, monkeypatch):
    training_data = tmp_path / "training.csv"
    training_data.write_text("dummy")

    expected_result = {
        "run_id": "test-run",
        "model_name": "ml-sentinel-model",
        "model_version": "100",
        "output_path": str(tmp_path / "model.joblib"),
        "metrics": {
            "accuracy": 0.80,
        },
    }

    def fake_retrain_and_validate(training_data_path, model_output_path):
        assert Path(training_data_path) == training_data
        assert Path(model_output_path) == tmp_path / "model.joblib"

        return SelfHealingResult(
            training=expected_result,
            promotion=PromotionResult(
                decision="PROMOTE",
                candidate_version="100",
                validation=ValidationResult(
                    decision="PROMOTE",
                    candidate_metric=0.80,
                    current_metric=0.75,
                    metric_name="f1",
                ),
            ),
        )

    monkeypatch.setattr(
        "ml_sentinel.control.actions.retrain_and_validate",
        fake_retrain_and_validate,
    )

    executor = ActionExecutor(
        dry_run=False,
        training_data_path=training_data,
        model_output_path=tmp_path / "model.joblib",
    )

    result = executor.execute(PolicyAction.RETRAIN)

    assert result.action == PolicyAction.RETRAIN
    assert result.status == "COMPLETED"

    assert result.details["training"] == expected_result

    assert result.details["promotion"]["decision"] == "PROMOTE"
    assert result.details["promotion"]["candidate_version"] == "100"
    assert (
        result.details["promotion"]["validation"]["candidate_metric"]
        == 0.80
    )
    assert (
        result.details["promotion"]["validation"]["current_metric"]
        == 0.75
    )
    assert (
        result.details["promotion"]["validation"]["metric_name"]
        == "f1"
    )

def test_rollback_completes_when_not_dry_run(monkeypatch):
    class FakeVersion:
        version = "5"

    class FakePreviousVersion:
        version = "4"

    class FakePromotedVersion:
        version = "4"

    monkeypatch.setattr(
        "ml_sentinel.control.actions.get_production_version",
        lambda: FakeVersion(),
    )

    monkeypatch.setattr(
        "ml_sentinel.control.actions.get_previous_version",
        lambda current_version: FakePreviousVersion(),
    )

    monkeypatch.setattr(
        "ml_sentinel.control.actions.promote_model",
        lambda model_version: FakePromotedVersion(),
    )

    executor = ActionExecutor(dry_run=False)

    result = executor.execute(PolicyAction.ROLLBACK)

    assert result.action == PolicyAction.ROLLBACK
    assert result.status == "COMPLETED"
    assert result.details["previous_production_version"] == "5"
    assert result.details["rollback_version"] == "4"

def test_drift_triggers_retraining_action():
    start_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    reference = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=42,
        start_time=start_time,
    )

    production = generate_data(
        n_samples=1000,
        scenario=DataScenario.DRIFT,
        seed=43,
        start_time=start_time,
    )

    drift_result = detect_drift(
        reference,
        production,
    )

    signals = ReliabilitySignals(
        drift_detected=drift_result["drift_detected"],
        drifted_feature_count=drift_result["drifted_feature_count"],
        total_feature_count=drift_result["total_feature_count"],
    )

    policy_action = evaluate_policy(signals)

    executor = ActionExecutor()

    action_result = executor.execute(policy_action)

    assert policy_action == PolicyAction.RETRAIN
    assert action_result.action == PolicyAction.RETRAIN
    assert action_result.status == "DRY_RUN"

def test_healthy_data_keeps_current_model():
    start_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    reference = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=42,
        start_time=start_time,
    )

    production = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=43,
        start_time=start_time,
    )

    drift_result = detect_drift(
        reference,
        production,
    )

    signals = ReliabilitySignals(
        drift_detected=drift_result["drift_detected"],
        drifted_feature_count=drift_result["drifted_feature_count"],
        total_feature_count=drift_result["total_feature_count"],
    )

    policy_action = evaluate_policy(signals)

    executor = ActionExecutor()

    action_result = executor.execute(policy_action)

    assert policy_action == PolicyAction.KEEP
    assert action_result.action == PolicyAction.KEEP
    assert action_result.status == "NO_OP"

def test_retrain_fails_when_training_data_does_not_exist(tmp_path):
    executor = ActionExecutor(
        dry_run=False,
        training_data_path=tmp_path / "missing.csv",
    )

    result = executor.execute(PolicyAction.RETRAIN)

    assert result.action == PolicyAction.RETRAIN
    assert result.status == "FAILED"
    assert "does not exist" in result.message

def test_retrain_executes_training_workflow(tmp_path, monkeypatch):
    training_data = tmp_path / "training.csv"
    training_data.write_text("dummy")

    expected_result = {
        "run_id": "test-run",
        "model_name": "ml-sentinel-model",
        "model_version": "99",
        "output_path": str(tmp_path / "model.joblib"),
        "metrics": {
            "accuracy": 0.75,
        },
    }

    def fake_retrain_and_validate(training_data_path, model_output_path):
        assert Path(training_data_path) == training_data
        assert Path(model_output_path) == tmp_path / "model.joblib"

        return SelfHealingResult(
            training=expected_result,
            promotion=PromotionResult(
                decision="BLOCK",
                candidate_version="99",
                validation=ValidationResult(
                    decision="BLOCK",
                    candidate_metric=0.70,
                    current_metric=0.80,
                    metric_name="f1",
                ),
            ),
        )

    monkeypatch.setattr(
        "ml_sentinel.control.actions.retrain_and_validate",
        fake_retrain_and_validate,
    )

    executor = ActionExecutor(
        dry_run=False,
        training_data_path=training_data,
        model_output_path=tmp_path / "model.joblib",
    )

    result = executor.execute(PolicyAction.RETRAIN)

    assert result.action == PolicyAction.RETRAIN
    assert result.status == "COMPLETED"

    assert result.details["training"] == expected_result

    assert result.details["promotion"]["decision"] == "BLOCK"
    assert result.details["promotion"]["candidate_version"] == "99"
    assert (
        result.details["promotion"]["validation"]["candidate_metric"]
        == 0.70
    )
    assert (
        result.details["promotion"]["validation"]["current_metric"]
        == 0.80
    )
    assert (
        result.details["promotion"]["validation"]["metric_name"]
        == "f1"
    )
