import pandas as pd
from ml_sentinel.control.actions import ActionResult
from ml_sentinel.control.controller import ReliabilityController
from ml_sentinel.policy.engine import PolicyAction
from ml_sentinel.chaos.runner import (
    ChaosResult,
    inject_data_drift,
    run_drift_chaos,
)
from ml_sentinel.data.generator import DataScenario


def test_inject_data_drift_returns_chaos_result():
    result = inject_data_drift(
        n_samples=100,
        seed=42,
    )

    assert isinstance(result, ChaosResult)
    assert result.scenario == DataScenario.DRIFT


def test_inject_data_drift_returns_expected_shape():
    result = inject_data_drift(
        n_samples=100,
        seed=42,
    )

    assert result.data.shape == (100, 6)


def test_inject_data_drift_returns_expected_columns():
    result = inject_data_drift(
        n_samples=100,
        seed=42,
    )

    expected_columns = {
        "timestamp",
        "temperature",
        "pressure",
        "vibration",
        "load",
        "target",
    }

    assert set(result.data.columns) == expected_columns


def test_inject_data_drift_is_reproducible():
    first = inject_data_drift(
        n_samples=100,
        seed=42,
    )

    second = inject_data_drift(
        n_samples=100,
        seed=42,
    )

    pd.testing.assert_frame_equal(
        first.data,
        second.data,
    )
class FakeExecutor:
    def execute(self, action):
        return ActionResult(
            action=action,
            status="TEST",
            message="test action executed",
        )


def test_run_drift_chaos_triggers_retraining():
    reference = pd.DataFrame(
        {
            "temperature": [50, 51, 49, 50, 52],
            "pressure": [100, 101, 99, 100, 102],
            "vibration": [0.5, 0.51, 0.49, 0.5, 0.52],
            "load": [50, 51, 49, 50, 52],
            "target": [0, 1, 0, 1, 0],
        }
    )

    controller = ReliabilityController(
        executor=FakeExecutor(),
    )

    decision = run_drift_chaos(
        reference_data=reference,
        n_samples=100,
        seed=42,
        controller=controller,
    )

    assert decision.action == PolicyAction.RETRAIN
    assert decision.result.status == "TEST"
