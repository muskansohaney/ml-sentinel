
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from ml_sentinel.data.loader import load_dataset
from ml_sentinel.data.schema import TARGET_COLUMN
from ml_sentinel.models.evaluation import evaluate_model
from ml_sentinel.models.training import (
    save_model,
    train_model,
)
from ml_sentinel.registry.model_registry import MODEL_NAME


def train_and_register(
    data_path: str | Path,
    output_path: str | Path = "models/model.joblib",
) -> dict[str, Any]:
    """
    Train, evaluate, save, and register an ML Sentinel model.

    Returns metadata about the completed training run.
    """

    print("[1/6] Loading dataset...", flush=True)

    dataset = load_dataset(data_path)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    print("[2/6] Splitting dataset...", flush=True)

    train_data, validation_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset[TARGET_COLUMN],
    )

    print("[3/6] Starting MLflow run...", flush=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ml-sentinel")

    with mlflow.start_run() as run:

        mlflow.log_param(
            "model_type",
            "logistic_regression",
        )

        mlflow.log_param(
            "training_samples",
            len(train_data),
        )

        mlflow.log_param(
            "validation_samples",
            len(validation_data),
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        print("[4/6] Training model...", flush=True)

        model = train_model(train_data)

        print("[5/6] Evaluating model...", flush=True)

        metrics = evaluate_model(
            model,
            validation_data,
        )

        mlflow.log_metrics(metrics)

        print("[6/6] Saving model...", flush=True)

        save_model(
            model,
            output_path,
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
        )

        run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/model"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
        )

        result = {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            "model_version": str(registered_model.version),
            "output_path": str(output_path),
            "metrics": metrics,
        }

        print("\nTraining completed.")
        print(f"Model saved to: {output_path}")
        print(f"MLflow run ID: {run_id}")
        print(f"Registered model: {MODEL_NAME}")
        print(f"Model version: {registered_model.version}")

        print("\nMetrics:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an ML Sentinel model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to training dataset.",
    )

    parser.add_argument(
        "--output",
        default="models/model.joblib",
        help="Path to save trained model.",
    )

    args = parser.parse_args()

    train_and_register(
        data_path=args.data,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
