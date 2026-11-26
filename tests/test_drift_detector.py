from datetime import datetime, timezone

import pytest

from ml_sentinel.data.generator import DataScenario, generate_data
from ml_sentinel.drift.detector import detect_drift


def test_normal_data_has_no_drift():
    start_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    reference = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=42,
        start_time=start_time,
    )

    production = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=43,
        start_time=start_time,
    )

    result = detect_drift(
        reference,
        production,
    )

    assert result["drift_detected"] is False


def test_drifted_data_is_detected():
    start_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    reference = generate_data(
        n_samples=1000,
        scenario=DataScenario.NORMAL,
        seed=42,
        start_time=start_time,
    )

    production = generate_data(
        n_samples=1000,
        scenario=DataScenario.DRIFT,
        seed=43,
        start_time=start_time,
    )

    result = detect_drift(
        reference,
        production,
    )

    assert result["drift_detected"] is True
    assert result["drifted_feature_count"] >= 2


def test_drift_result_contains_all_features():
    start_time = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    reference = generate_data(
        n_samples=500,
        scenario=DataScenario.NORMAL,
        seed=42,
        start_time=start_time,
    )

    production = generate_data(
        n_samples=500,
        scenario=DataScenario.DRIFT,
        seed=43,
        start_time=start_time,
    )

    result = detect_drift(
        reference,
        production,
    )

    assert set(result["features"]) == {
        "temperature",
        "pressure",
        "vibration",
        "load",
    }
    assert result["drift_share_threshold"] == 0.5
    assert result["feature_drift_threshold"] == 0.1
    for feature in result["features"].values():
        assert set(feature) == {
            "drift_score",
            "drift_detected",
            "drift_threshold",
        }


@pytest.mark.parametrize(
    ("feature_threshold", "share_threshold"),
    [
        (0.0, 0.5),
        (1.1, 0.5),
        (0.1, 0.0),
        (0.1, 1.1),
    ],
)
def test_drift_rejects_invalid_thresholds(
    feature_threshold,
    share_threshold,
):
    reference = generate_data(n_samples=100, seed=42)
    production = generate_data(n_samples=100, seed=43)

    with pytest.raises(ValueError):
        detect_drift(
            reference,
            production,
            feature_drift_threshold=feature_threshold,
            drift_share_threshold=share_threshold,
        )
