from prometheus_client import Counter, Gauge, Histogram


PREDICTION_COUNT = Counter(
    "ml_sentinel_predictions_total",
    "Total number of predictions served.",
    ["model_name", "model_version", "prediction"],
)


PREDICTION_LATENCY = Histogram(
    "ml_sentinel_prediction_latency_seconds",
    "Prediction request latency in seconds.",
    ["model_name", "model_version"],
)


PREDICTION_ERRORS = Counter(
    "ml_sentinel_prediction_errors_total",
    "Total number of prediction errors.",
    ["model_name", "model_version"],
)


QUALITY_COUNT = Counter(
    "ml_sentinel_prediction_quality_total",
    "Total number of correct and incorrect predictions.",
    ["model_name", "model_version", "result"],
)


# ------------------------------------------------------------------
# Reliability metrics
# ------------------------------------------------------------------

DRIFT_DETECTED = Gauge(
    "ml_sentinel_drift_detected",
    "Whether production data drift is currently detected.",
)


DRIFTED_FEATURE_COUNT = Gauge(
    "ml_sentinel_drifted_feature_count",
    "Number of production features currently detected as drifted.",
)

MONITORING_OBSERVATION_COUNT = Gauge(
    "ml_sentinel_monitoring_observation_count",
    "Recent production observations considered by drift monitoring.",
)

MONITORING_QUALITY_OBSERVATION_COUNT = Gauge(
    "ml_sentinel_monitoring_quality_observation_count",
    "Recent labeled predictions for the currently served model.",
)

MONITORING_DRIFT_SHARE = Gauge(
    "ml_sentinel_monitoring_drift_share",
    "Share of monitored production features detected as drifted.",
)

MONITORING_EVALUATED = Gauge(
    "ml_sentinel_monitoring_evaluated",
    "Whether enough production observations were available for evaluation.",
)


MODEL_QUALITY = Gauge(
    "ml_sentinel_model_quality",
    "Current observed model quality.",
)


MODEL_LATENCY_MS = Gauge(
    "ml_sentinel_model_latency_ms",
    "Current observed model latency in milliseconds.",
)


RELIABILITY_ACTION = Gauge(
    "ml_sentinel_reliability_action",
    "Last reliability policy action taken.",
    ["action"],
)
