from __future__ import annotations

import os
import time
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from ml_sentinel.config.settings import (
    PRODUCTION_DATA_PATH,
    REFERENCE_DATA_PATH,
)
from ml_sentinel.monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    QUALITY_COUNT,
)
from ml_sentinel.monitoring.service import (
    load_and_collect_monitoring_snapshot,
)
from ml_sentinel.monitoring.signals import QualitySnapshot
from ml_sentinel.policy.engine import evaluate_policy
from ml_sentinel.registry.model_registry import MODEL_NAME


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


app = FastAPI(
    title="ML Sentinel",
    description="Production ML reliability platform",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    temperature: float
    pressure: float
    vibration: float
    load: float
    actual_target: int | None = None


class PredictionResponse(BaseModel):
    prediction: int
    model_name: str
    model_version: str
    latency_ms: float


class FeatureDriftResponse(BaseModel):
    drift_score: float | None
    drift_detected: bool
    drift_threshold: float


class DriftMonitoringResponse(BaseModel):
    drift_detected: bool
    drifted_feature_count: int
    total_feature_count: int
    drift_share: float
    drift_share_threshold: float
    feature_drift_threshold: float
    model_quality: float | None
    policy_action: str
    features: dict[str, FeatureDriftResponse]


def load_model():
    """Load the model assigned to the MLflow production alias."""
    client = mlflow.MlflowClient()

    try:
        production_version = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="production",
        )
    except Exception as exc:
        raise RuntimeError(
            f"No production model configured for {MODEL_NAME}"
        ) from exc

    model_uri = (
        f"models:/{MODEL_NAME}@production"
    )

    model = mlflow.sklearn.load_model(model_uri)

    return model, str(production_version.version)


MODEL, MODEL_VERSION = load_model()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get(
    "/monitoring/drift",
    response_model=DriftMonitoringResponse,
)
def monitoring_drift(
    correct: int = Query(default=0, ge=0),
    incorrect: int = Query(default=0, ge=0),
):
    """Report local data drift and the resulting policy recommendation."""
    snapshot = load_and_collect_monitoring_snapshot(
        REFERENCE_DATA_PATH,
        PRODUCTION_DATA_PATH,
        QualitySnapshot(
            correct=correct,
            incorrect=incorrect,
        ),
    )
    action = evaluate_policy(snapshot.signals)

    return DriftMonitoringResponse(
        drift_detected=snapshot.signals.drift_detected,
        drifted_feature_count=snapshot.drifted_feature_count,
        total_feature_count=snapshot.total_feature_count,
        drift_share=snapshot.drift_share,
        drift_share_threshold=snapshot.drift_share_threshold,
        feature_drift_threshold=snapshot.feature_drift_threshold,
        model_quality=snapshot.signals.model_quality,
        policy_action=action.value,
        features={
            name: FeatureDriftResponse(**details)
            for name, details in snapshot.feature_drift.items()
        },
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.perf_counter()

    try:
        features = pd.DataFrame(
            [
                {
                    "temperature": request.temperature,
                    "pressure": request.pressure,
                    "vibration": request.vibration,
                    "load": request.load,
                }
            ]
        )

        prediction = int(MODEL.predict(features)[0])
        if request.actual_target is not None:
            result = (
                "correct"
                if prediction == request.actual_target
                else "incorrect"
            )

            QUALITY_COUNT.labels(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                result=result,
            ).inc()

        latency_seconds = (
            time.perf_counter() - start_time
        )

        PREDICTION_COUNT.labels(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            prediction=str(prediction),
        ).inc()

        PREDICTION_LATENCY.labels(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        ).observe(latency_seconds)

        return PredictionResponse(
            prediction=prediction,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            latency_ms=latency_seconds * 1000,
        )

    except Exception:
        PREDICTION_ERRORS.labels(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        ).inc()

        raise
