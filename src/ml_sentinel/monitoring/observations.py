from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ml_sentinel.data.schema import FEATURE_COLUMNS
from ml_sentinel.monitoring.signals import QualitySnapshot


MAX_PRODUCTION_OBSERVATIONS = 1000
MIN_PRODUCTION_OBSERVATIONS = 30


def _ensure_schema(connection: sqlite3.Connection) -> None:
    feature_columns = ", ".join(
        f'"{column}" REAL NOT NULL' for column in FEATURE_COLUMNS
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS prediction_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {feature_columns},
            prediction INTEGER NOT NULL,
            actual_target INTEGER,
            model_version TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def record_prediction(
    database_path: str | Path,
    features: dict[str, float],
    prediction: int,
    actual_target: int | None,
    model_version: str,
) -> None:
    missing_features = [
        column for column in FEATURE_COLUMNS if column not in features
    ]
    if missing_features:
        raise ValueError(f"Missing prediction features: {missing_features}")

    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    placeholders = ", ".join("?" for _ in range(len(FEATURE_COLUMNS) + 3))
    insert_columns = ", ".join(
        [
            *(f'"{column}"' for column in FEATURE_COLUMNS),
            "prediction",
            "actual_target",
            "model_version",
        ]
    )
    values = [float(features[column]) for column in FEATURE_COLUMNS]
    values.extend([prediction, actual_target, model_version])

    with sqlite3.connect(database_path, timeout=10) as connection:
        _ensure_schema(connection)
        connection.execute(
            f"""
            INSERT INTO prediction_observations ({insert_columns})
            VALUES ({placeholders})
            """,
            values,
        )
        connection.execute(
            """
            DELETE FROM prediction_observations
            WHERE id NOT IN (
                SELECT id
                FROM prediction_observations
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (MAX_PRODUCTION_OBSERVATIONS,),
        )


def load_recent_predictions(
    database_path: str | Path,
    model_version: str,
    *,
    limit: int = MAX_PRODUCTION_OBSERVATIONS,
) -> tuple[pd.DataFrame, QualitySnapshot]:
    if limit < 1:
        raise ValueError("Observation limit must be at least one.")

    columns = [*FEATURE_COLUMNS, "prediction", "actual_target", "model_version"]
    database_path = Path(database_path)
    if not database_path.exists():
        return pd.DataFrame(columns=columns), QualitySnapshot()

    feature_names = ", ".join(f'"{column}"' for column in FEATURE_COLUMNS)
    with sqlite3.connect(database_path, timeout=10) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            f"""
            SELECT {feature_names}, prediction, actual_target, model_version
            FROM prediction_observations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    rows.reverse()
    observations = pd.DataFrame(rows, columns=columns)
    current_model = observations[
        (observations["model_version"] == model_version)
        & observations["actual_target"].notna()
    ]
    correct = int(
        (current_model["prediction"] == current_model["actual_target"]).sum()
    )
    incorrect = len(current_model) - correct

    return (
        observations[FEATURE_COLUMNS],
        QualitySnapshot(correct=correct, incorrect=incorrect),
    )
