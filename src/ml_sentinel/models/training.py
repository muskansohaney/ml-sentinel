from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_sentinel.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def build_model() -> Pipeline:
    """Build the baseline ML model."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )


def train_model(
    dataset: pd.DataFrame,
) -> Pipeline:
    """Train the baseline model."""

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    model = build_model()

    model.fit(X, y)

    return model


def save_model(
    model: Pipeline,
    output_path: str | Path,
) -> None:
    """Persist a trained model."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )
