import numpy as np
import pandas as pd
import pytest

from ml_sentinel.models.explainability import explain_model
from ml_sentinel.models.evaluation import (
    benchmark_model_candidates,
    cross_validate_model,
    evaluate_model,
    tune_logistic_regression,
)
from ml_sentinel.models.training import (
    build_lightgbm_model,
    build_model,
    train_model,
)


def create_dataset(n_samples=200):
    return pd.DataFrame(
        {
            "temperature": range(n_samples),
            "pressure": range(n_samples),
            "vibration": [
                value / 100
                for value in range(n_samples)
            ],
            "load": range(n_samples),
            "target": [
                int(value % 2 == 0)
                for value in range(n_samples)
            ],
        }
    )


def test_build_model():
    model = build_model()

    assert model is not None


def test_train_model():
    dataset = create_dataset()

    model = train_model(dataset)

    predictions = model.predict(
        dataset[
            [
                "temperature",
                "pressure",
                "vibration",
                "load",
            ]
        ]
    )

    assert len(predictions) == len(dataset)


def test_evaluate_model():
    dataset = create_dataset()

    model = train_model(dataset)

    metrics = evaluate_model(
        model,
        dataset,
    )

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert set(metrics.keys()) == expected_metrics

    for value in metrics.values():
        assert 0 <= value <= 1


def test_cross_validate_model():
    dataset = create_dataset()

    metrics = cross_validate_model(dataset)

    expected_metrics = {
        f"cv_{metric}_{stat}"
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
        for stat in ("mean", "std")
    }
    assert set(metrics) == expected_metrics

    for value in metrics.values():
        assert 0 <= value <= 1


def test_cross_validate_model_rejects_too_many_folds():
    dataset = create_dataset(n_samples=8)

    with pytest.raises(ValueError, match="requires at least 5 samples"):
        cross_validate_model(dataset)


def test_cross_validate_model_requires_at_least_two_folds():
    dataset = create_dataset()

    with pytest.raises(ValueError, match="at least 2 folds"):
        cross_validate_model(dataset, cv=1)


def test_benchmark_model_candidates_compares_all_models():
    dataset = create_dataset()

    results = benchmark_model_candidates(dataset, cv=3)

    assert set(results) == {"logistic_regression", "xgboost", "lightgbm"}
    for metrics in results.values():
        assert "cv_roc_auc_mean" in metrics
        assert 0 <= metrics["cv_roc_auc_mean"] <= 1


def test_build_lightgbm_model_is_cpu_only():
    model = build_lightgbm_model()

    assert model.get_params()["device_type"] == "cpu"
    assert model.get_params()["n_jobs"] == 1


def test_tune_logistic_regression_returns_best_search():
    dataset = create_dataset()

    search = tune_logistic_regression(dataset, cv=3)

    assert set(search.best_params_) == {
        "classifier__C",
        "classifier__class_weight",
    }
    assert 0 <= search.best_score_ <= 1
    assert len(search.cv_results_["params"]) == 6


def test_explain_model_returns_additive_feature_attributions():
    dataset = create_dataset()
    model = train_model(dataset)

    explanation = explain_model(model, dataset, background_size=50)
    scaled_features = model.named_steps["scaler"].transform(
        dataset[
            [
                "temperature",
                "pressure",
                "vibration",
                "load",
            ]
        ]
    )
    decision_scores = model.named_steps["classifier"].decision_function(
        scaled_features
    )

    assert explanation.values.shape == (len(dataset), 4)
    assert explanation.feature_names == [
        "temperature",
        "pressure",
        "vibration",
        "load",
    ]
    assert np.allclose(
        explanation.values.sum(axis=1) + explanation.base_values,
        decision_scores,
    )


def test_explain_model_rejects_empty_data():
    dataset = create_dataset().iloc[0:0]
    model = train_model(create_dataset())

    with pytest.raises(ValueError, match="empty dataset"):
        explain_model(model, dataset)
