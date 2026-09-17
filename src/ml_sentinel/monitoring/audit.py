from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ml_sentinel.control.actions import ActionResult
from ml_sentinel.policy.engine import PolicyAction, ReliabilitySignals


def record_decision(
    signals: ReliabilitySignals,
    action: PolicyAction,
    result: ActionResult,
    log_path: str | Path = "logs/decisions.jsonl",
) -> None:
    """Record a reliability decision as a JSON Lines event."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action.value,
        "status": result.status,
        "message": result.message,
        "signals": {
            "drift_detected": signals.drift_detected,
            "drifted_feature_count": signals.drifted_feature_count,
            "total_feature_count": signals.total_feature_count,
            "latency_ms": signals.latency_ms,
            "model_quality": signals.model_quality,
        },
    }

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")
