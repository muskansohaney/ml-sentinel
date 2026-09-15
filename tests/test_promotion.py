import mlflow
import pytest

from ml_sentinel.control.promotion import validate_and_promote
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


def create_model_version(client, f1):
    """Create a registered model version with an F1 metric."""
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass

    with mlflow.start_run() as run:
        mlflow.log_metric("f1", f1)
        run_id = run.info.run_id

    return client.create_model_version(
        name=MODEL_NAME,
        source="test-model",
        run_id=run_id,
    )


def test_validate_and_promote_promotes_better_candidate(mlflow_client):
    current = create_model_version(
        mlflow_client,
        f1=0.75,
    )

    candidate = create_model_version(
        mlflow_client,
        f1=0.85,
    )

    result = validate_and_promote(
        candidate_version=str(candidate.version),
        current_version=str(current.version),
    )

    assert result.decision == "PROMOTE"
    assert result.candidate_version == str(candidate.version)

    production = mlflow_client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="production",
    )

    assert production.version == candidate.version


def test_validate_and_promote_blocks_worse_candidate(mlflow_client):
    current = create_model_version(
        mlflow_client,
        f1=0.85,
    )

    candidate = create_model_version(
        mlflow_client,
        f1=0.75,
    )

    # Establish the current model as production first.
    from ml_sentinel.registry.model_registry import promote_model

    promote_model(current.version)

    result = validate_and_promote(
        candidate_version=str(candidate.version),
        current_version=str(current.version),
    )

    assert result.decision == "BLOCK"
    assert result.candidate_version == str(candidate.version)

    production = mlflow_client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="production",
    )

    assert production.version == current.version
