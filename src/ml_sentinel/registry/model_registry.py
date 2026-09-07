from __future__ import annotations

import mlflow
from mlflow import MlflowClient


MODEL_NAME = "ml-sentinel-model"


def get_client() -> MlflowClient:
    """Return an MLflow client using the configured tracking backend."""
    return MlflowClient()


def register_model(
    model_uri: str,
    model_name: str = MODEL_NAME,
):
    """Register a trained MLflow model and return its model version."""
    client = get_client()

    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=None,
    )

    return model_version


def get_latest_version(
    model_name: str = MODEL_NAME,
):
    """Return the latest version of a registered model."""
    client = get_client()

    versions = list(
        client.search_model_versions(
            f"name='{model_name}'"
        )
    )

    if not versions:
        return None

    return max(
        versions,
        key=lambda version: int(version.version),
    )