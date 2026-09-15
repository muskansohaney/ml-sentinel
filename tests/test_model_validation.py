import mlflow
import pytest

from ml_sentinel.models.validation import (
    validate_candidate,
    validate_registered_models,
)
from ml_sentinel.registry.model_registry import MODEL_NAME


@pytest.fixture
def mlflow_client(tmp_path):
    """Create an isolated MLflow database for testing."""
    db_path = tmp_path / "mlflow_test.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{db_path}"
    )

    client = mlflow.MlflowClient()

    experiment = client.create_experiment(
        "test-experiment"
    )

    mlflow.set_experiment(
        experiment_id=experiment
    )

    yield client

    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )


def test_validate_candidate_promotes_when_candidate_is_better():
    result = validate_candidate(
        candidate_metric=0.85,
        current_metric=0.80,
    )

    assert result.decision == "PROMOTE"
    assert result.candidate_metric == 0.85
    assert result.current_metric == 0.80
    assert result.metric_name == "f1"


def test_validate_candidate_promotes_when_metrics_are_equal():
    result = validate_candidate(
        candidate_metric=0.80,
        current_metric=0.80,
    )

    assert result.decision == "PROMOTE"


def test_validate_candidate_blocks_when_candidate_is_worse():
    result = validate_candidate(
        candidate_metric=0.75,
        current_metric=0.80,
    )

    assert result.decision == "BLOCK"


def test_validate_candidate_promotes_without_current_model():
    result = validate_candidate(
        candidate_metric=0.70,
        current_metric=None,
    )

    assert result.decision == "PROMOTE"


def test_validate_registered_models(mlflow_client):
    try:
        mlflow_client.create_registered_model(MODEL_NAME)
    except Exception:
        pass

    with mlflow.start_run() as current_run:
        mlflow.log_metric("f1", 0.75)
        current_run_id = current_run.info.run_id

    current_version = mlflow_client.create_model_version(
        name=MODEL_NAME,
        source="test-current-model",
        run_id=current_run_id,
    )

    with mlflow.start_run() as candidate_run:
        mlflow.log_metric("f1", 0.85)
        candidate_run_id = candidate_run.info.run_id

    candidate_version = mlflow_client.create_model_version(
        name=MODEL_NAME,
        source="test-candidate-model",
        run_id=candidate_run_id,
    )

    result = validate_registered_models(
        candidate_version=str(candidate_version.version),
        current_version=str(current_version.version),
    )

    assert result.decision == "PROMOTE"
    assert result.candidate_metric == 0.85
    assert result.current_metric == 0.75
