from ml_sentinel.control.canary import evaluate_canary


def test_canary_promotes_candidate_with_better_metric():
    result = evaluate_canary(
        candidate_version="101",
        candidate_metric=0.85,
        baseline_metric=0.80,
    )

    assert result.decision == "PROMOTE"
    assert result.candidate_version == "101"
    assert result.candidate_metric == 0.85
    assert result.baseline_metric == 0.80


def test_canary_promotes_candidate_within_allowed_degradation():
    result = evaluate_canary(
        candidate_version="102",
        candidate_metric=0.79,
        baseline_metric=0.80,
        threshold=0.02,
    )

    assert result.decision == "PROMOTE"


def test_canary_blocks_candidate_below_threshold():
    result = evaluate_canary(
        candidate_version="103",
        candidate_metric=0.75,
        baseline_metric=0.80,
        threshold=0.02,
    )

    assert result.decision == "BLOCK"


def test_canary_promotes_equal_metric():
    result = evaluate_canary(
        candidate_version="104",
        candidate_metric=0.80,
        baseline_metric=0.80,
    )

    assert result.decision == "PROMOTE"


def test_canary_threshold_zero_requires_no_degradation():
    result = evaluate_canary(
        candidate_version="105",
        candidate_metric=0.799,
        baseline_metric=0.80,
        threshold=0.0,
    )

    assert result.decision == "BLOCK"
