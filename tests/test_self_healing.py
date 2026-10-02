from ml_sentinel.control.self_healing import retrain_and_validate
from ml_sentinel.models.validation import ValidationResult
from ml_sentinel.control.promotion import PromotionResult


def test_retrain_and_validate_promotes_better_candidate(
    monkeypatch,
    tmp_path,
):
    training_result = {
        "run_id": "run-123",
        "model_name": "ml-sentinel-model",
        "model_version": "5",
        "output_path": str(tmp_path / "model.joblib"),
        "metrics": {
            "f1": 0.90,
        },
    }

    class FakeVersion:
        version = "4"

    promotion_result = PromotionResult(
        decision="PROMOTE",
        candidate_version="5",
        validation=ValidationResult(
            decision="PROMOTE",
            candidate_metric=0.90,
            current_metric=0.80,
            metric_name="f1",
        ),
    )

    monkeypatch.setattr(
        "ml_sentinel.control.self_healing.get_production_version",
        lambda: FakeVersion(),
    )

    monkeypatch.setattr(
        "ml_sentinel.control.self_healing.train_and_register",
        lambda data_path, output_path: training_result,
    )

    monkeypatch.setattr(
        "ml_sentinel.control.self_healing.validate_and_promote",
        lambda candidate_version, current_version, metric_name:
            promotion_result,
    )

    result = retrain_and_validate(
        training_data_path=tmp_path / "production.csv",
    )

    assert result.training["model_version"] == "5"
    assert result.promotion.decision == "PROMOTE"
    assert result.promotion.candidate_version == "5"


def test_retrain_and_validate_blocks_worse_candidate(
    monkeypatch,
    tmp_path,
):
    training_result = {
        "run_id": "run-456",
        "model_name": "ml-sentinel-model",
        "model_version": "6",
        "output_path": str(tmp_path / "model.joblib"),
        "metrics": {
            "f1": 0.60,
        },
    }

    class FakeVersion:
        version = "5"

    promotion_result = PromotionResult(
        decision="BLOCK",
        candidate_version="6",
        validation=ValidationResult(
            decision="BLOCK",
            candidate_metric=0.60,
            current_metric=0.80,
            metric_name="f1",
        ),
    )

    monkeypatch.setattr(
        "ml_sentinel.control.self_healing.get_production_version",
        lambda: FakeVersion(),
    )

    monkeypatch.setattr(
        "ml_sentinel.control.self_healing.train_and_register",
        lambda data_path, output_path: training_result,
    )

    monkeypatch.setattr(
        "ml_sentinel.control.self_healing.validate_and_promote",
        lambda candidate_version, current_version, metric_name:
            promotion_result,
    )

    result = retrain_and_validate(
        training_data_path=tmp_path / "production.csv",
    )

    assert result.training["model_version"] == "6"
    assert result.promotion.decision == "BLOCK"
    assert result.promotion.candidate_version == "6"
