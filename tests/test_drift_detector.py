from datetime import datetime, timezone

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
