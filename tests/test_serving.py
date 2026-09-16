import mlflow
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("mlflow") / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.MlflowClient()
    model_name = "ml-sentinel-model"
    mlflow.set_experiment("serving-test")
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            model,
            name="model",
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=model_version.version,
    )

    from ml_sentinel.serving.app import app

    return TestClient(app)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_name"] == "ml-sentinel-model"
    assert data["model_version"].isdigit()


def test_predict(client):
    response = client.post(
        "/predict",
        json={
            "temperature": 50.0,
            "pressure": 100.0,
            "vibration": 0.5,
            "load": 50.0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["model_name"] == "ml-sentinel-model"
    assert data["model_version"].isdigit()
    assert data["latency_ms"] >= 0

def test_predict_with_actual_target_records_quality(client):
    response = client.post(
        "/predict",
        json={
            "temperature": 50.0,
            "pressure": 100.0,
            "vibration": 0.5,
            "load": 50.0,
            "actual_target": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["model_name"] == "ml-sentinel-model"
    assert data["model_version"].isdigit()