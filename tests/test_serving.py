from fastapi.testclient import TestClient

from ml_sentinel.serving.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_name"] == "ml-sentinel-model"
    assert data["model_version"].isdigit()


def test_predict():
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
