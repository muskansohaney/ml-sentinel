from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd


class DataScenario(str, Enum):
    NORMAL = "normal"
    DRIFT = "drift"
    SPIKE = "spike"
    SHIFT = "shift"
    CORRUPTION = "corruption"


def generate_data(
    n_samples: int = 1000,
    scenario: DataScenario = DataScenario.NORMAL,
    seed: int | None = None,
    start_time: datetime | None = None,
    interval_seconds: int = 1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic production data for ML Sentinel.

    The generated dataset represents a binary classification problem
    with numerical production features.

    Args:
        n_samples: Number of samples to generate.
        scenario: Data generation scenario.
        seed: Random seed for reproducible generation.
        start_time: Timestamp of the first generated sample.
        interval_seconds: Time interval between samples.
        random_state: Backward-compatible alias for seed.
    """

    # Support the existing random_state API while allowing seed.
    if seed is not None and random_state is not None:
        raise ValueError(
            "Provide either 'seed' or 'random_state', not both."
        )

    effective_seed = seed if seed is not None else random_state

    rng = np.random.default_rng(effective_seed)

    if start_time is None:
        start_time = datetime.now(timezone.utc)

    timestamps = [
        start_time + timedelta(seconds=i * interval_seconds)
        for i in range(n_samples)
    ]

    # Baseline production distribution
    temperature = rng.normal(50, 5, n_samples)
    pressure = rng.normal(100, 10, n_samples)
    vibration = rng.normal(0.5, 0.1, n_samples)
    load = rng.normal(70, 15, n_samples)

    if scenario == DataScenario.DRIFT:
        drift_strength = np.linspace(0, 1, n_samples)

        temperature += 8 * drift_strength
        pressure += 15 * drift_strength
        vibration += 0.15 * drift_strength

    elif scenario == DataScenario.SPIKE:
        # Introduce extreme observations
        spike_indices = rng.choice(
            n_samples,
            size=max(1, n_samples // 20),
            replace=False,
        )

        temperature[spike_indices] += 30
        pressure[spike_indices] += 50
        vibration[spike_indices] += 0.8

    elif scenario == DataScenario.SHIFT:
        # Change the overall population distribution
        temperature = rng.normal(65, 8, n_samples)
        pressure = rng.normal(125, 15, n_samples)
        vibration = rng.normal(0.8, 0.15, n_samples)
        load = rng.normal(85, 10, n_samples)

    elif scenario == DataScenario.CORRUPTION:
        # Introduce missing/invalid values
        corruption_indices = rng.choice(
            n_samples,
            size=max(1, n_samples // 20),
            replace=False,
        )

        temperature[corruption_indices] = np.nan

    # Synthetic target
    # Use a clean copy only for generating the target.
    # The returned dataset still contains the corrupted values.
    target_temperature = np.nan_to_num(
        temperature,
        nan=50.0,
    )

    risk_score = (
        0.04 * (target_temperature - 50)
        + 0.025 * (pressure - 100)
        + 3.0 * (vibration - 0.5)
        + 0.015 * (load - 50)
    )

    probability = 1 / (1 + np.exp(-(risk_score - 0.5)))

    target = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": temperature,
            "pressure": pressure,
            "vibration": vibration,
            "load": load,
            "target": target,
        }
    )


def save_data(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save generated data to disk."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic production data for ML Sentinel."
    )

    parser.add_argument(
        "--scenario",
        choices=[scenario.value for scenario in DataScenario],
        default=DataScenario.NORMAL.value,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/production.csv",
    )

    args = parser.parse_args()

    scenario = DataScenario(args.scenario)

    df = generate_data(
        n_samples=args.samples,
        scenario=scenario,
    )

    save_data(
        df,
        args.output,
    )

    print(f"Generated {len(df)} samples")
    print(f"Scenario: {scenario.value}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()