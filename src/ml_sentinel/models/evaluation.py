from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)

from ml_sentinel.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from ml_sentinel.models.training import (
    build_lightgbm_model,
    build_model,
    build_xgboost_model,
)


DEFAULT_CROSS_VALIDATION_FOLDS = 5


def _stratified_cv(y: pd.Series, cv: int) -> StratifiedKFold:
    if cv < 2:
        raise ValueError("Cross-validation requires at least 2 folds.")

    class_counts = y.value_counts()
    if len(class_counts) != 2:
        raise ValueError("Cross-validation requires exactly two target classes.")

    min_class_count = int(class_counts.min())
    if min_class_count < cv:
        raise ValueError(
            f"Cross-validation with {cv} folds requires at least "
            f"{cv} samples in each target class; found {min_class_count}."
        )

    return StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
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


def cross_validate_model(
    dataset: pd.DataFrame,
    cv: int = DEFAULT_CROSS_VALIDATION_FOLDS,
    estimator: BaseEstimator | None = None,
) -> dict[str, float]:
    """Return mean and standard deviation of stratified fold metrics."""

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        estimator if estimator is not None else build_model(),
        X,
        y,
        cv=_stratified_cv(y, cv),
        scoring=scoring,
        error_score="raise",
        n_jobs=1,
    )

    return {
        f"cv_{metric}_{stat}": float(values)
        for metric in scoring
        for stat, values in (
            ("mean", scores[f"test_{metric}"].mean()),
            ("std", scores[f"test_{metric}"].std()),
        )
    }


def tune_logistic_regression(
    dataset: pd.DataFrame,
    cv: int = DEFAULT_CROSS_VALIDATION_FOLDS,
) -> GridSearchCV:
    """Tune baseline logistic-regression parameters using ROC-AUC."""

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    search = GridSearchCV(
        estimator=build_model(),
        param_grid={
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=_stratified_cv(y, cv),
        n_jobs=1,
        error_score="raise",
        refit=True,
    )
    search.fit(X, y)

    return search


def benchmark_model_candidates(
    dataset: pd.DataFrame,
    cv: int = DEFAULT_CROSS_VALIDATION_FOLDS,
) -> dict[str, dict[str, float]]:
    """Compare the baseline, XGBoost, and LightGBM using the same CV folds."""

    candidates = {
        "logistic_regression": build_model(),
        "xgboost": build_xgboost_model(),
        "lightgbm": build_lightgbm_model(),
    }

    return {
        name: cross_validate_model(
            dataset,
            cv=cv,
            estimator=estimator,
        )
        for name, estimator in candidates.items()
    }
