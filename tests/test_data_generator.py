from ml_sentinel.data.generator import (
    DataScenario,
    generate_data,
    save_data,
)

from datetime import datetime, timezone


def test_generator_returns_expected_rows():
    df = generate_data(
        n_samples=500,
        scenario=DataScenario.NORMAL,
    )

    assert len(df) == 500


def test_generator_contains_expected_columns():
    df = generate_data()

    expected_columns = {
        "timestamp",
        "temperature",
        "pressure",
        "vibration",
        "load",
        "target",
    }

    assert set(df.columns) == expected_columns


def test_target_is_binary():
    df = generate_data()

    assert set(df["target"].dropna().unique()).issubset({0, 1})


def test_drift_changes_distribution():
    normal = generate_data(
        n_samples=2000,
        scenario=DataScenario.NORMAL,
    )

    drift = generate_data(
        n_samples=2000,
        scenario=DataScenario.DRIFT,
    )

    normal_mean = normal["temperature"].mean()
    drift_mean = drift["temperature"].mean()

    assert drift_mean > normal_mean


def test_corruption_introduces_missing_values():
    df = generate_data(
        n_samples=1000,
        scenario=DataScenario.CORRUPTION,
    )

    assert df["temperature"].isna().sum() > 0


def test_generation_is_reproducible():
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    df1 = generate_data(
        n_samples=100,
        random_state=42,
        start_time=start_time,
    )

    df2 = generate_data(
        n_samples=100,
        random_state=42,
        start_time=start_time,
    )

    assert df1.equals(df2)

def test_generator_contains_timestamp():
    df = generate_data(
        n_samples=100,
        scenario=DataScenario.NORMAL,
    )

    assert "timestamp" in df.columns


def test_timestamps_are_ordered():
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    df = generate_data(
        n_samples=100,
        scenario=DataScenario.NORMAL,
        start_time=start_time,
        interval_seconds=60,
    )

    timestamps = df["timestamp"]

    assert timestamps.is_monotonic_increasing


def test_timestamp_interval_is_correct():
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    df = generate_data(
        n_samples=10,
        start_time=start_time,
        interval_seconds=60,
    )

    differences = df["timestamp"].diff().dropna()

    assert all(
        difference.total_seconds() == 60
        for difference in differences
    )

def test_save_data_creates_parent_directory(tmp_path):
    output_path = (
        tmp_path
        / "nested"
        / "directory"
        / "data.csv"
    )

    df = generate_data(
        n_samples=10,
        scenario=DataScenario.NORMAL,
        start_time=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    save_data(
        df,
        output_path,
    )

    assert output_path.exists()