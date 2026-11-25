from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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


def build_xgboost_model() -> XGBClassifier:
    """Build a CPU-only XGBoost candidate model."""

    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
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
