from __future__ import annotations

import numpy as np

from errorquake.analyze import FitResult
from errorquake.synthetic import generate_synthetic_scores, validate_pipeline_recovery


def _counts(scores: np.ndarray) -> list[int]:
    support = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)
    return [int(np.sum(np.isclose(scores, point))) for point in support]


def test_generate_synthetic_scores_power_law_decreases_monotonically() -> None:
    counts = _counts(generate_synthetic_scores("power_law", n=1000))
    assert counts == sorted(counts, reverse=True)


def test_generate_synthetic_scores_exponential_decreases_monotonically() -> None:
    counts = _counts(generate_synthetic_scores("exponential", n=1000))
    assert counts == sorted(counts, reverse=True)


def test_generate_synthetic_scores_uniform_is_close_to_even() -> None:
    scores = generate_synthetic_scores("uniform", n=1000)
    expected = 1000 / 8
    for count in _counts(scores):
        assert abs(count - expected) <= expected * 0.20


def test_validate_pipeline_recovery_pass_case() -> None:
    result = validate_pipeline_recovery(
        "power_law",
        {"b": 1.0},
        [
            FitResult(
                model_name="m",
                distribution="power_law",
                parameters={"beta": 1.1},
                bic=1.0,
                aic=1.0,
                chi2_stat=1.0,
                chi2_pvalue=0.5,
                n_errors=100,
            )
        ],
        target_scores=[0.5, 1.0, 1.5, 2.0],
        final_scores=[0.5, 1.0, 1.5, 2.0],
    )
    assert result["verdict"] == "PASS"


def test_validate_pipeline_recovery_fail_on_wrong_family() -> None:
    result = validate_pipeline_recovery(
        "power_law",
        {"b": 1.0},
        [
            FitResult(
                model_name="m",
                distribution="exponential",
                parameters={"lambda": 0.5},
                bic=1.0,
                aic=1.0,
                chi2_stat=1.0,
                chi2_pvalue=0.5,
                n_errors=100,
            )
        ],
        target_scores=[0.5, 1.0, 1.5, 2.0],
        final_scores=[0.5, 1.0, 1.5, 2.0],
    )
    assert result["verdict"] == "FAIL"


def test_validate_pipeline_recovery_fail_on_low_judge_correlation() -> None:
    result = validate_pipeline_recovery(
        "power_law",
        {"b": 1.0},
        [
            FitResult(
                model_name="m",
                distribution="power_law",
                parameters={"beta": 1.0},
                bic=1.0,
                aic=1.0,
                chi2_stat=1.0,
                chi2_pvalue=0.5,
                n_errors=100,
            )
        ],
        target_scores=[0.5, 1.0, 1.5, 2.0],
        final_scores=[2.0, 1.5, 1.0, 0.5],
    )
    assert result["judge_calibration_pass"] is False
    assert result["verdict"] == "FAIL"
