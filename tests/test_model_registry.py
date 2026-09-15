import mlflow
import pytest
from mlflow import MlflowClient

from ml_sentinel.registry.model_registry import (
    MODEL_NAME,
    get_latest_version,
    get_model_metrics,
)


@pytest.fixture
def mlflow_client(tmp_path):
    """Create an isolated MLflow database for testing."""
    db_path = tmp_path / "mlflow_test.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{db_path}"
    )

    client = MlflowClient()

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


def test_model_name():
    assert MODEL_NAME == "ml-sentinel-model"


def test_get_latest_version_empty(mlflow_client):
    assert get_latest_version() is None

def test_get_model_metrics(mlflow_client):
    try:
        mlflow_client.create_registered_model(MODEL_NAME)
    except Exception:
        pass

    with mlflow.start_run() as run:
        mlflow.log_metric("f1", 0.85)
        run_id = run.info.run_id

    model_version = mlflow_client.create_model_version(
        name=MODEL_NAME,
        source="test-model-artifact",
        run_id=run_id,
    )

    metrics = get_model_metrics(
        model_version.version
    )

    assert metrics["f1"] == 0.85