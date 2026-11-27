from __future__ import annotations

import os
import time
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ml_sentinel.config.settings import (
    MONITORING_DB_PATH,
    REFERENCE_DATA_PATH,
)
from ml_sentinel.data.schema import FEATURE_COLUMNS
from ml_sentinel.drift.detector import (
    DRIFT_SHARE_THRESHOLD,
    FEATURE_DRIFT_THRESHOLD,
)
from ml_sentinel.monitoring.metrics import (
    DRIFT_DETECTED,
    DRIFTED_FEATURE_COUNT,
    MODEL_QUALITY,
    MONITORING_DRIFT_SHARE,
    MONITORING_EVALUATED,
    MONITORING_EVALUATION_TIMESTAMP,
    MONITORING_OBSERVATION_COUNT,
    MONITORING_QUALITY_OBSERVATION_COUNT,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    QUALITY_COUNT,
)
from ml_sentinel.monitoring.observations import (
    MAX_PRODUCTION_OBSERVATIONS,
    MIN_PRODUCTION_OBSERVATIONS,
    load_recent_predictions,
    record_prediction,
)
from ml_sentinel.monitoring.service import (
    collect_monitoring_snapshot,
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
    actual_target: int | None = Field(default=None, ge=0, le=1)


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
    monitoring_status: str
    observation_count: int
    quality_observation_count: int
    drift_detected: bool
    drifted_feature_count: int
    total_feature_count: int
    drift_share: float
    drift_share_threshold: float
    feature_drift_threshold: float
    model_quality: float | None
    policy_action: str | None
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


def update_monitoring_metrics(
    *,
    observation_count: int,
    quality_observation_count: int,
    drift_detected: bool,
    drifted_feature_count: int,
    drift_share: float,
    evaluated: bool,
    model_quality: float | None,
) -> None:
    MONITORING_OBSERVATION_COUNT.set(observation_count)
    MONITORING_QUALITY_OBSERVATION_COUNT.set(quality_observation_count)
    MONITORING_DRIFT_SHARE.set(drift_share)
    MONITORING_EVALUATED.set(1 if evaluated else 0)
    MONITORING_EVALUATION_TIMESTAMP.set(time.time())
    DRIFT_DETECTED.set(1 if drift_detected else 0)
    DRIFTED_FEATURE_COUNT.set(drifted_feature_count)
    MODEL_QUALITY.set(
        float("nan") if model_quality is None else model_quality
    )


@app.get(
    "/monitoring/drift",
    response_model=DriftMonitoringResponse,
)
def monitoring_drift():
    """Report drift and quality from recent predictions served locally."""
    production_data, quality = load_recent_predictions(
        MONITORING_DB_PATH,
        MODEL_VERSION,
        limit=MAX_PRODUCTION_OBSERVATIONS,
    )
    quality_observation_count = quality.correct + quality.incorrect

    if len(production_data) < MIN_PRODUCTION_OBSERVATIONS:
        update_monitoring_metrics(
            observation_count=len(production_data),
            quality_observation_count=quality_observation_count,
            drift_detected=False,
            drifted_feature_count=0,
            drift_share=0.0,
            evaluated=False,
            model_quality=quality.accuracy,
        )
        return DriftMonitoringResponse(
            monitoring_status="insufficient_data",
            observation_count=len(production_data),
            quality_observation_count=quality_observation_count,
            drift_detected=False,
            drifted_feature_count=0,
            total_feature_count=len(FEATURE_COLUMNS),
            drift_share=0.0,
            drift_share_threshold=DRIFT_SHARE_THRESHOLD,
            feature_drift_threshold=FEATURE_DRIFT_THRESHOLD,
            model_quality=quality.accuracy,
            policy_action=None,
            features={},
        )

    reference_data = pd.read_csv(REFERENCE_DATA_PATH)
    snapshot = collect_monitoring_snapshot(
        reference_data,
        production_data,
        quality,
    )
    action = evaluate_policy(snapshot.signals)
    update_monitoring_metrics(
        observation_count=len(production_data),
        quality_observation_count=quality_observation_count,
        drift_detected=snapshot.signals.drift_detected,
        drifted_feature_count=snapshot.drifted_feature_count,
        drift_share=snapshot.drift_share,
        evaluated=True,
        model_quality=snapshot.signals.model_quality,
    )

    return DriftMonitoringResponse(
        monitoring_status="evaluated",
        observation_count=len(production_data),
        quality_observation_count=quality_observation_count,
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
        record_prediction(
            MONITORING_DB_PATH,
            features.iloc[0].to_dict(),
            prediction,
            request.actual_target,
            MODEL_VERSION,
        )
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
