import mlflow
import numpy as np
import pandas as pd
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

    import ml_sentinel.serving.app as app_module

    app_module.MONITORING_DB_PATH = (
        tmp_path_factory.mktemp("monitoring") / "observations.sqlite3"
    )
    return TestClient(app_module.app)


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


def test_monitoring_drift_requires_minimum_observations(
    client,
    tmp_path,
    monkeypatch,
):
    import ml_sentinel.serving.app as app_module

    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "temperature": [50, 51, 49, 50, 52],
            "pressure": [100, 101, 99, 100, 102],
            "vibration": [0.5, 0.51, 0.49, 0.5, 0.52],
            "load": [50, 51, 49, 50, 52],
        }
    ).to_csv(reference_path, index=False)
    monkeypatch.setattr(
        app_module,
        "REFERENCE_DATA_PATH",
        reference_path,
    )
    monkeypatch.setattr(
        app_module,
        "MONITORING_DB_PATH",
        tmp_path / "observations.sqlite3",
    )

    response = client.get("/monitoring/drift")

    assert response.status_code == 200
    data = response.json()
    assert data["monitoring_status"] == "insufficient_data"
    assert data["observation_count"] == 0
    assert data["policy_action"] is None
    metrics = client.get("/metrics").text
    assert "ml_sentinel_monitoring_observation_count 0.0" in metrics
    assert "ml_sentinel_monitoring_evaluated 0.0" in metrics


def test_monitoring_drift_uses_recent_prediction_requests(
    client,
    tmp_path,
    monkeypatch,
):
    import ml_sentinel.serving.app as app_module

    class ConstantModel:
        def predict(self, features):
            return np.zeros(len(features), dtype=int)

    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "temperature": np.linspace(48, 52, 200),
            "pressure": np.linspace(96, 104, 200),
            "vibration": np.linspace(0.45, 0.55, 200),
            "load": np.linspace(45, 55, 200),
        }
    ).to_csv(reference_path, index=False)
    monkeypatch.setattr(app_module, "REFERENCE_DATA_PATH", reference_path)
    monkeypatch.setattr(app_module, "MODEL", ConstantModel())
    monkeypatch.setattr(
        app_module,
        "MONITORING_DB_PATH",
        tmp_path / "observations.sqlite3",
    )

    for index in range(30):
        prediction_response = client.post(
            "/predict",
            json={
                "temperature": 70 + index * 0.1,
                "pressure": 120 + index * 0.2,
                "vibration": 0.8 + index * 0.01,
                "load": 70 + index * 0.2,
                "actual_target": 0,
            },
        )
        assert prediction_response.status_code == 200

    response = client.get("/monitoring/drift")

    assert response.status_code == 200
    data = response.json()
    assert data["monitoring_status"] == "evaluated"
    assert data["observation_count"] == 30
    assert data["quality_observation_count"] == 30
    assert data["drift_detected"] is True
    assert data["drifted_feature_count"] >= 2
    assert data["total_feature_count"] == 4
    assert data["drift_share"] >= data["drift_share_threshold"]
    assert data["model_quality"] is not None
    assert data["model_quality"] == 1.0
    assert data["policy_action"] == "RETRAIN"
    assert set(data["features"]) == {
        "temperature",
        "pressure",
        "vibration",
        "load",
    }
    metrics = client.get("/metrics").text
    assert "ml_sentinel_monitoring_observation_count 30.0" in metrics
    assert "ml_sentinel_monitoring_quality_observation_count 30.0" in metrics
    assert "ml_sentinel_monitoring_last_evaluation_timestamp_seconds" in metrics
    assert (
        f"ml_sentinel_monitoring_drift_share {data['drift_share']}"
        in metrics
    )
    assert "ml_sentinel_monitoring_evaluated 1.0" in metrics
    assert "ml_sentinel_drift_detected 1.0" in metrics
    assert "ml_sentinel_drifted_feature_count" in metrics
    assert "ml_sentinel_model_quality 1.0" in metrics