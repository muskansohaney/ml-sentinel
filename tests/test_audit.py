import json

from ml_sentinel.control.actions import ActionResult
from ml_sentinel.monitoring.audit import record_decision
from ml_sentinel.policy.engine import PolicyAction, ReliabilitySignals


def test_record_decision_creates_jsonl_event(tmp_path):
    log_path = tmp_path / "decisions.jsonl"

    signals = ReliabilitySignals(
        model_quality=0.45,
        latency_ms=120.0,
        drift_detected=True,
        drifted_feature_count=2,
        total_feature_count=4,
    )

    result = ActionResult(
        action=PolicyAction.RETRAIN,
        status="DRY_RUN",
        message="Retraining workflow would be started.",
    )

    record_decision(
        signals,
        PolicyAction.RETRAIN,
        result,
        log_path,
    )

    assert log_path.exists()

    event = json.loads(log_path.read_text())

    assert event["action"] == "RETRAIN"
    assert event["status"] == "DRY_RUN"
    assert event["signals"]["model_quality"] == 0.45
    assert event["signals"]["drift_detected"] is True
    assert event["signals"]["drifted_feature_count"] == 2
    assert event["signals"]["total_feature_count"] == 4
    assert "timestamp" in event


def test_record_decision_appends_events(tmp_path):
    log_path = tmp_path / "decisions.jsonl"

    signals = ReliabilitySignals()

    result = ActionResult(
        action=PolicyAction.KEEP,
        status="NO_OP",
        message="Current model remains active.",
    )

    record_decision(signals, PolicyAction.KEEP, result, log_path)
    record_decision(signals, PolicyAction.KEEP, result, log_path)

    lines = log_path.read_text().splitlines()

    assert len(lines) == 2
