from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_sentinel.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def evaluate_model(
    model: Any,
    dataset: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate a trained model."""

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
    }
