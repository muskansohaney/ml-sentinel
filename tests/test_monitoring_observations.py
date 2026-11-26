import pytest

from ml_sentinel.monitoring.observations import (
    load_recent_predictions,
    record_prediction,
)


def test_load_recent_predictions_limits_rows_and_scopes_quality_by_version(
    tmp_path,
):
    database_path = tmp_path / "monitoring.sqlite3"

    for index in range(5):
        record_prediction(
            database_path,
            {
                "temperature": float(index),
                "pressure": float(index + 1),
                "vibration": float(index + 2),
                "load": float(index + 3),
            },
            prediction=index % 2,
            actual_target=(index + 1) % 2,
            model_version="previous" if index == 3 else "current",
        )

    observations, quality = load_recent_predictions(
        database_path,
        "current",
        limit=3,
    )

    assert len(observations) == 3
    assert observations["temperature"].tolist() == [2.0, 3.0, 4.0]
    assert quality.correct == 0
    assert quality.incorrect == 2


def test_load_recent_predictions_without_database_returns_empty_data(tmp_path):
    observations, quality = load_recent_predictions(
        tmp_path / "missing.sqlite3",
        "current",
    )

    assert observations.empty
    assert quality.accuracy is None


def test_record_prediction_bounds_local_observation_history(
    tmp_path,
    monkeypatch,
):
    import ml_sentinel.monitoring.observations as observations_module

    monkeypatch.setattr(observations_module, "MAX_PRODUCTION_OBSERVATIONS", 2)
    database_path = tmp_path / "monitoring.sqlite3"

    for index in range(3):
        record_prediction(
            database_path,
            {
                "temperature": float(index),
                "pressure": float(index),
                "vibration": float(index),
                "load": float(index),
            },
            prediction=0,
            actual_target=None,
            model_version="current",
        )

    production_data, _ = load_recent_predictions(
        database_path,
        "current",
    )

    assert production_data["temperature"].tolist() == [1.0, 2.0]


def test_record_prediction_rejects_missing_features(tmp_path):
    with pytest.raises(ValueError, match="Missing prediction features"):
        record_prediction(
            tmp_path / "monitoring.sqlite3",
            {"temperature": 50.0},
            prediction=0,
            actual_target=None,
            model_version="1",
        )
