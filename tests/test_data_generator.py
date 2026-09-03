import numpy as np

from ml_sentinel.data.generator import (
    DataScenario,
    generate_data,
)


def test_generator_returns_expected_rows():
    df = generate_data(
        n_samples=500,
        scenario=DataScenario.NORMAL,
    )

    assert len(df) == 500


def test_generator_contains_expected_columns():
    df = generate_data()

    expected_columns = {
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
    df1 = generate_data(
        n_samples=100,
        random_state=42,
    )

    df2 = generate_data(
        n_samples=100,
        random_state=42,
    )

    assert np.array_equal(
        df1.to_numpy(),
        df2.to_numpy(),
        equal_nan=True,
    )