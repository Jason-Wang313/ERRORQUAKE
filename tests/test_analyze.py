from __future__ import annotations

import numpy as np

from errorquake.analyze import (
    compute_prediction_metrics,
    estimate_b_value,
    fit_all_distributions,
    fit_discrete_power_law,
    fit_exponential,
    predict_catastrophic_rate,
    ratio_test,
    run_prediction_experiment,
    vuong_test,
)


def test_fit_discrete_power_law_recovers_beta() -> None:
    rng = np.random.default_rng(42)
    support = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    probs = support ** (-1.5)
    probs = probs / probs.sum()
    scores = rng.choice(support, size=1000, p=probs)
    fit = fit_discrete_power_law(scores)
    assert abs(fit.parameters["beta"] - 1.5) < 0.35


def test_fit_exponential_recovers_lambda() -> None:
    rng = np.random.default_rng(42)
    support = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    probs = np.exp(-0.8 * support)
    probs = probs / probs.sum()
    scores = rng.choice(support, size=1000, p=probs)
    fit = fit_exponential(scores)
    assert abs(fit.parameters["lambda"] - 0.8) < 0.25


def test_fit_all_distributions_sorted_by_bic() -> None:
    scores = np.array([0.5] * 50 + [1.0] * 30 + [1.5] * 20, dtype=float)
    fits = fit_all_distributions(scores, "model-x")
    assert fits == sorted(fits, key=lambda item: item.bic)


def test_estimate_b_value_ci_contains_true_value() -> None:
    rng = np.random.default_rng(42)
    support = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    probs = 10 ** (-1.0 * (support - 0.5))
    probs = probs / probs.sum()
    scores = rng.choice(support, size=1200, p=probs)
    b_value = estimate_b_value(scores, model_name="m", n_bootstrap=300)
    assert b_value.b_ci_lower <= b_value.b <= b_value.b_ci_upper
    assert b_value.m_min in support
    assert b_value.n_above_mmin > 0


def test_ratio_test_behaviour() -> None:
    power_like = np.array([0.5] * 100 + [1.0] * 50 + [1.5] * 25 + [2.0] * 12, dtype=float)
    irregular = np.array([0.5] * 120 + [1.0] * 10 + [1.5] * 90 + [2.0] * 5, dtype=float)
    assert ratio_test(power_like, [0.5, 1.0, 1.5, 2.0])["supports_power_law"] is True
    assert ratio_test(irregular, [0.5, 1.0, 1.5, 2.0])["supports_power_law"] is False


def test_prediction_helpers() -> None:
    predicted = predict_catastrophic_rate(1.0, 100, 0.5, target_magnitude=3.0)
    assert round(predicted, 2) == 0.32
    results = run_prediction_experiment(
        {"m": np.array([0.5] * 100 + [1.0] * 60 + [1.5] * 40)},
        {"m": np.array([3.0] * 10 + [2.5] * 20)},
    )
    metrics = compute_prediction_metrics(results)
    assert metrics["within_1_5x_fraction"] >= 0.0


def test_vuong_prefers_power_law_on_power_law_data() -> None:
    rng = np.random.default_rng(42)
    support = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    probs = support ** (-1.3)
    probs = probs / probs.sum()
    scores = rng.choice(support, size=1500, p=probs)
    power_fit = fit_discrete_power_law(scores)
    exp_fit = fit_exponential(scores)
    result = vuong_test(scores, power_fit, exp_fit)
    assert result["preferred"] == "power_law"
