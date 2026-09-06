import pandas as pd

from ml_sentinel.models.evaluation import evaluate_model
from ml_sentinel.models.training import (
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
