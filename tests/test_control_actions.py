from datetime import datetime, timezone

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


def test_retrain_is_not_implemented_when_not_dry_run():
    executor = ActionExecutor(dry_run=False)

    result = executor.execute(PolicyAction.RETRAIN)

    assert result.status == "NOT_IMPLEMENTED"


def test_rollback_is_not_implemented_when_not_dry_run():
    executor = ActionExecutor(dry_run=False)

    result = executor.execute(PolicyAction.ROLLBACK)

    assert result.status == "NOT_IMPLEMENTED"

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