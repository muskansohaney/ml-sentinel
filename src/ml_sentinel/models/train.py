import argparse

import mlflow
from sklearn.model_selection import train_test_split

from ml_sentinel.data.loader import load_dataset
from ml_sentinel.data.schema import TARGET_COLUMN
from ml_sentinel.models.evaluation import evaluate_model
from ml_sentinel.models.training import (
    save_model,
    train_model,
)


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

    # Load dataset
    dataset = load_dataset(args.data)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    # Split into training and validation datasets
    train_data, validation_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=dataset[TARGET_COLUMN],
    )

    # Configure MLflow experiment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ml-sentinel")
    # Start an MLflow run
    with mlflow.start_run():

        # Log experiment parameters
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

        # Train model
        model = train_model(train_data)

        # Evaluate model on validation data
        metrics = evaluate_model(
            model,
            validation_data,
        )

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Save model artifact
        save_model(
            model,
            args.output,
        )

        # Log model artifact to MLflow
        mlflow.log_artifact(
            args.output,
        )

        # Display results
        run_id = mlflow.active_run().info.run_id

        print("Training completed.")
        print(f"Model saved to: {args.output}")
        print(f"MLflow run ID: {run_id}")

        print("\nMetrics:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
