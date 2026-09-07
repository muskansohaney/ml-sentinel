from prometheus_client import Counter, Histogram


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
