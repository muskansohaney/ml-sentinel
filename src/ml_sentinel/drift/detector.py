
from __future__ import annotations

from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from ml_sentinel.data.schema import FEATURE_COLUMNS


DRIFT_SHARE_THRESHOLD = 0.5
FEATURE_DRIFT_THRESHOLD = 0.1


def detect_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    *,
    feature_drift_threshold: float = FEATURE_DRIFT_THRESHOLD,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
) -> dict[str, Any]:
    """
    Compare reference and production feature distributions.

    Returns overall drift status and per-feature drift information.
    """

    missing_reference = [
        column
        for column in FEATURE_COLUMNS
        if column not in reference_data.columns
    ]

    missing_production = [
        column
        for column in FEATURE_COLUMNS
        if column not in production_data.columns
    ]

    if missing_reference:
        raise ValueError(
            f"Missing features in reference data: {missing_reference}"
        )

    if missing_production:
        raise ValueError(
            f"Missing features in production data: {missing_production}"
        )

    reference = reference_data[FEATURE_COLUMNS].copy()
    production = production_data[FEATURE_COLUMNS].copy()

    if not 0 < feature_drift_threshold <= 1:
        raise ValueError(
            "Feature drift threshold must be greater than zero and at most one."
        )

    if not 0 < drift_share_threshold <= 1:
        raise ValueError(
            "Drift share threshold must be greater than zero and at most one."
        )

    report = Report(
        [
            DataDriftPreset(
                drift_share=drift_share_threshold,
            ),
        ]
    )

    snapshot = report.run(
        current_data=production,
        reference_data=reference,
    )

    results = snapshot.dict()

    feature_drift: dict[str, dict[str, Any]] = {}

    for metric in results.get("metrics", []):
        metric_name = metric.get("metric_name", "")

        if not metric_name.startswith("ValueDrift("):
            continue

        config = metric.get("config", {})
        column = config.get("column")
        value = metric.get("value")

        if column not in FEATURE_COLUMNS:
            continue

        drift_score = float(value)
        feature_drift[column] = {
            "drift_score": drift_score,
            "drift_detected": drift_score >= feature_drift_threshold,
            "drift_threshold": feature_drift_threshold,
        }

    drifted_columns = 0

    # Mark individual features using Evidently's drift score.
    #
    # For the per-feature result, we use the ValueDrift
    # metric value as an informational score rather than
    # interpreting it as a universal probability threshold.
    for column in FEATURE_COLUMNS:
        if column not in feature_drift:
            feature_drift[column] = {
                "drift_score": None,
                "drift_detected": False,
                "drift_threshold": feature_drift_threshold,
            }

    for metric in results.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        if metric_name.startswith("DriftedColumnsCount("):
            value = metric.get("value", {})
            if isinstance(value, dict):
                drifted_columns = int(value.get("count", 0))
            break

    drift_detected = (
        drifted_columns / len(FEATURE_COLUMNS)
        >= drift_share_threshold
    )

    return {
        "drift_detected": drift_detected,
        "drifted_feature_count": drifted_columns,
        "total_feature_count": len(FEATURE_COLUMNS),
        "drift_share": (
            drifted_columns / len(FEATURE_COLUMNS)
        ),
        "drift_share_threshold": drift_share_threshold,
        "feature_drift_threshold": feature_drift_threshold,
        "features": feature_drift,
    }
