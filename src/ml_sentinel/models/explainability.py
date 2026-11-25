import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from ml_sentinel.data.schema import FEATURE_COLUMNS


def explain_model(
    model: Pipeline,
    dataset: pd.DataFrame,
    background_size: int = 100,
) -> shap.Explanation:
    """Explain a fitted scaled logistic-regression model in log-odds units."""

    if dataset.empty:
        raise ValueError("Cannot explain an empty dataset.")
    if background_size < 1:
        raise ValueError("background_size must be at least 1.")

    features = dataset[FEATURE_COLUMNS]
    background = features.sample(
        n=min(background_size, len(features)),
        random_state=42,
    )
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    scaled_features = scaler.transform(features)
    scaled_background = scaler.transform(background)

    explainer = shap.LinearExplainer(
        classifier,
        shap.maskers.Independent(scaled_background),
    )
    values = explainer(scaled_features)

    return shap.Explanation(
        values=values.values,
        base_values=values.base_values,
        data=features.to_numpy(),
        feature_names=list(FEATURE_COLUMNS),
    )
