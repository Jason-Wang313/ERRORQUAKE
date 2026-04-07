"""Distribution fitting, b-value estimation, and prediction pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
from scipy import optimize, stats

DEFAULT_SCALE_POINTS = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)


@dataclass
class FitResult:
    """Result of fitting one distribution to one model's error data."""

    model_name: str
    distribution: str
    parameters: dict[str, float]
    bic: float
    aic: float
    chi2_stat: float
    chi2_pvalue: float
    n_errors: int


@dataclass
class BValue:
    """b-value estimate with confidence interval."""

    model_name: str
    b: float
    b_ci_lower: float
    b_ci_upper: float
    m_min: float
    n_above_mmin: int
    method: str


@dataclass
class PredictionResult:
    """Prediction of catastrophic failure rate from micro-errors."""

    model_name: str
    b_easy: float
    predicted_catastrophic: float
    observed_catastrophic: float
    ratio: float
    within_1_5x: bool


def _quantize_to_grid(scores: np.ndarray) -> np.ndarray:
    """Snap scores to nearest grid point in DEFAULT_SCALE_POINTS.

    Final scores can be averages (e.g. 2.25 from primary=2.0, secondary=2.5).
    Distribution fitting requires scores to lie on the discrete support, so
    snap each score to the closest grid point. This fixes both the chi-square
    sum mismatch and the searchsorted out-of-bounds issues.
    """
    arr = np.asarray(scores, dtype=float)
    # For each score, find the nearest grid point
    diffs = np.abs(arr[:, None] - DEFAULT_SCALE_POINTS[None, :])
    return DEFAULT_SCALE_POINTS[np.argmin(diffs, axis=1)]


def _prepare_scores(scores: np.ndarray, m_min: float = 0.5) -> np.ndarray:
    array = np.asarray(scores, dtype=float)
    # Snap to grid before filtering so non-grid averages match grid points
    array = _quantize_to_grid(array)
    filtered = array[array >= m_min - 1e-9]
    if filtered.size == 0:
        raise ValueError("Need at least one score at or above m_min.")
    return filtered


def _support_for(scores: np.ndarray, m_min: float) -> np.ndarray:
    support = DEFAULT_SCALE_POINTS[DEFAULT_SCALE_POINTS >= m_min - 1e-9]
    observed_max = float(np.max(scores))
    support = support[support <= observed_max + 1e-9]
    if support.size == 0:
        raise ValueError("No valid support points for the requested m_min.")
    return support


def _normalised_pmf(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 1e-12, None)
    total = values.sum()
    if total <= 0 or not np.isfinite(total):
        # Degenerate case: return uniform
        return np.ones_like(values) / len(values)
    return values / total


def _counts(scores: np.ndarray, support: np.ndarray) -> np.ndarray:
    # Use isclose for float-safe equality (scores are pre-quantized)
    return np.array(
        [np.isclose(scores, point, atol=1e-6).sum() for point in support],
        dtype=float,
    )


def _safe_indices(support: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Map each score to its index in support, clipped to valid range."""
    idx = np.searchsorted(support, scores)
    return np.clip(idx, 0, len(support) - 1)


def _metrics(
    distribution: str,
    params: dict[str, float],
    pmf: np.ndarray,
    scores: np.ndarray,
    support: np.ndarray,
) -> FitResult:
    counts = _counts(scores, support)
    # Use safe indices and add eps to log
    idx = _safe_indices(support, scores)
    ll = float(np.sum(np.log(pmf[idx] + 1e-12)))
    k = len(params)
    n = len(scores)

    # Compute expected and renormalize so sum(expected) == sum(counts)
    # to satisfy scipy.stats.chisquare's tolerance check.
    counts_sum = float(counts.sum())
    if counts_sum > 0:
        expected = pmf * counts_sum
    else:
        expected = pmf * n
    expected = np.clip(expected, 1e-9, None)
    # Final renormalization to match counts.sum() exactly
    if expected.sum() > 0 and counts_sum > 0:
        expected = expected * (counts_sum / expected.sum())

    try:
        chi2_stat, chi2_pvalue = stats.chisquare(counts, f_exp=expected)
        chi2_stat = float(chi2_stat)
        chi2_pvalue = float(chi2_pvalue)
    except Exception:
        chi2_stat = float("nan")
        chi2_pvalue = float("nan")

    bic = float(k * math.log(n) - 2 * ll) if n > 0 and np.isfinite(ll) else float("inf")
    aic = float(2 * k - 2 * ll) if np.isfinite(ll) else float("inf")
    return FitResult(
        model_name="unknown",
        distribution=distribution,
        parameters=params,
        bic=bic,
        aic=aic,
        chi2_stat=chi2_stat,
        chi2_pvalue=chi2_pvalue,
        n_errors=int(n),
    )


def _fit_single_param(
    scores: np.ndarray,
    m_min: float,
    distribution: str,
    transform: Callable[[np.ndarray, float], np.ndarray],
    bounds: tuple[float, float],
    param_name: str,
) -> FitResult:
    prepared = _prepare_scores(scores, m_min=m_min)
    support = _support_for(prepared, m_min)
    idx = _safe_indices(support, prepared)

    def objective(value: float) -> float:
        pmf = _normalised_pmf(transform(support, value))
        return -float(np.sum(np.log(pmf[idx] + 1e-12)))

    result = optimize.minimize_scalar(objective, bounds=bounds, method="bounded")
    param = float(result.x)
    pmf = _normalised_pmf(transform(support, param))
    return _metrics(distribution, {param_name: param}, pmf, prepared, support)


def fit_discrete_power_law(scores: np.ndarray, m_min: float = 0.5) -> FitResult:
    """Discrete MLE fit: P(M=m) ∝ m^(-β). Uses exact discrete MLE."""
    return _fit_single_param(
        scores,
        m_min,
        "power_law",
        lambda support, beta: support ** (-beta),
        bounds=(0.05, 10.0),
        param_name="beta",
    )


def fit_exponential(scores: np.ndarray) -> FitResult:
    """MLE fit: P(M=m) ∝ exp(-λm)."""
    prepared = _prepare_scores(scores, m_min=float(np.min(scores[scores > 0])))
    return _fit_single_param(
        prepared,
        float(np.min(prepared)),
        "exponential",
        lambda support, lambda_: np.exp(-lambda_ * support),
        bounds=(0.01, 10.0),
        param_name="lambda",
    )


def _fit_two_param(
    scores: np.ndarray,
    m_min: float,
    distribution: str,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
    initial: np.ndarray,
    bounds: list[tuple[float, float]],
    names: tuple[str, str],
) -> FitResult:
    prepared = _prepare_scores(scores, m_min=m_min)
    support = _support_for(prepared, m_min)
    idx = _safe_indices(support, prepared)

    def objective(params: np.ndarray) -> float:
        pmf = _normalised_pmf(transform(support, params))
        return -float(np.sum(np.log(pmf[idx] + 1e-12)))

    result = optimize.minimize(objective, x0=initial, bounds=bounds, method="L-BFGS-B")
    params = np.asarray(result.x, dtype=float)
    pmf = _normalised_pmf(transform(support, params))
    return _metrics(
        distribution,
        {names[0]: float(params[0]), names[1]: float(params[1])},
        pmf,
        prepared,
        support,
    )


def fit_truncated_power_law(scores: np.ndarray, m_min: float = 0.5) -> FitResult:
    """P(M=m) ∝ m^(-β) * exp(-λm). Two-parameter fit."""
    return _fit_two_param(
        scores,
        m_min,
        "truncated_power_law",
        lambda support, params: (support ** (-params[0])) * np.exp(-params[1] * support),
        initial=np.array([1.5, 0.1]),
        bounds=[(0.05, 10.0), (0.001, 10.0)],
        names=("beta", "lambda"),
    )


def fit_lognormal(scores: np.ndarray) -> FitResult:
    """Lognormal fit via MLE."""
    prepared = _prepare_scores(scores, m_min=float(np.min(scores[scores > 0])))
    m_min = float(np.min(prepared))
    return _fit_two_param(
        prepared,
        m_min,
        "lognormal",
        lambda support, params: stats.lognorm.pdf(
            support,
            s=max(params[1], 1e-3),
            scale=np.exp(params[0]),
        ),
        initial=np.array([math.log(np.mean(prepared)), 0.5]),
        bounds=[(-5.0, 5.0), (0.05, 5.0)],
        names=("mu", "sigma"),
    )


def fit_stretched_exponential(scores: np.ndarray) -> FitResult:
    """P(M=m) ∝ exp(-λ * m^γ). Two-parameter fit."""
    prepared = _prepare_scores(scores, m_min=float(np.min(scores[scores > 0])))
    m_min = float(np.min(prepared))
    return _fit_two_param(
        prepared,
        m_min,
        "stretched_exp",
        lambda support, params: np.exp(-params[0] * np.power(support, params[1])),
        initial=np.array([0.5, 1.0]),
        bounds=[(0.001, 10.0), (0.1, 5.0)],
        names=("lambda", "gamma"),
    )


def _failed_fit(distribution: str, model_name: str, n: int, reason: str = "") -> FitResult:
    """Sentinel FitResult for fits that failed numerically."""
    return FitResult(
        model_name=model_name,
        distribution=distribution,
        parameters={"failed": 1.0, "reason_len": float(len(reason))},
        bic=float("inf"),
        aic=float("inf"),
        chi2_stat=float("nan"),
        chi2_pvalue=float("nan"),
        n_errors=int(n),
    )


def fit_all_distributions(
    scores: np.ndarray,
    model_name: str,
    m_min: float = 0.5,
) -> list[FitResult]:
    """
    Fit all 5 distributions. Return sorted by BIC (best first).

    Each fit is wrapped in try/except so a single numerical failure does not
    crash the whole pipeline. Failed fits return a sentinel with bic=inf.
    """
    try:
        prepared = _prepare_scores(scores, m_min=m_min)
    except ValueError:
        # No data above m_min
        return [
            _failed_fit(d, model_name, 0, "no_data")
            for d in ("power_law", "truncated_power_law", "lognormal",
                      "exponential", "stretched_exp")
        ]

    n = len(prepared)
    fits: list[FitResult] = []

    fitters = [
        ("power_law", lambda: fit_discrete_power_law(prepared, m_min=m_min)),
        ("truncated_power_law", lambda: fit_truncated_power_law(prepared, m_min=m_min)),
        ("lognormal", lambda: fit_lognormal(prepared)),
        ("exponential", lambda: fit_exponential(prepared)),
        ("stretched_exp", lambda: fit_stretched_exponential(prepared)),
    ]
    for name, fn in fitters:
        try:
            result = fn()
            fits.append(replace(result, model_name=model_name))
        except Exception as exc:
            fits.append(_failed_fit(name, model_name, n, str(exc)[:80]))

    return sorted(fits, key=lambda item: item.bic)


_MIN_N_FOR_B_FIT = 30  # require at least this many points above m_min for stable MLE


def _estimate_b(scores: np.ndarray, m_min: float) -> float:
    """Aki MLE for Gutenberg–Richter b-value.

    b = log10(e) / (M_bar - M_min - delta/2)

    Note: scores must already be quantized to grid points before calling this.
    The denominator subtracts delta/2 (the discretization correction) for
    binned data — without it, b is biased upward.
    """
    prepared = _prepare_scores(scores, m_min=m_min)
    if prepared.size == 0:
        return float("nan")
    m_bar = float(np.mean(prepared))
    # Discretization correction: for binned data with bin width delta=0.5,
    # the unbiased estimator subtracts delta/2 from the mean offset.
    # Without correction, b is overestimated when n is small.
    delta = 0.5
    denom = m_bar - m_min + delta / 2.0
    if denom < 0.05:  # protect against pathological denominators
        return float("nan")
    return math.log10(math.e) / denom


def _ks_distance(scores: np.ndarray, b: float, m_min: float) -> float:
    if not math.isfinite(b):
        return float("inf")
    prepared = np.sort(_prepare_scores(scores, m_min=m_min))
    if prepared.size == 0:
        return float("inf")
    support = _support_for(prepared, m_min)
    empirical = np.array([(prepared <= point).mean() for point in support])
    pmf = _normalised_pmf(10 ** (-b * (support - m_min)))
    model = np.cumsum(pmf)
    return float(np.max(np.abs(empirical - model)))


def estimate_b_value(
    scores: np.ndarray,
    model_name: str,
    method: str = "mle",
    n_bootstrap: int = 10000,
) -> BValue:
    """
    Estimate b-value with automatic m_min selection.

    FIX (2026-04-07): Three bugs caused multiple models to converge to b=1.737:
    1. m_min candidates came from raw scores including non-grid averaged values
       (e.g., 3.25 from averaging 3.0+3.5). After quantization these collapsed
       onto grid points, creating a fixed offset of 0.25 in the denominator.
    2. The KS-distance selector picked extreme m_min values (e.g., 3.75) with
       only 1-3 points above, which is statistically meaningless.
    3. The MLE formula lacked the discretization correction for binned data.

    The fix:
    - Quantize positive scores to grid points BEFORE selecting m_min
    - Restrict m_min candidates to grid points only
    - Require at least _MIN_N_FOR_B_FIT points above m_min
    - Apply the delta/2 discretization correction in the MLE formula
    """
    positive = np.asarray(scores, dtype=float)
    positive = positive[positive > 0]
    if positive.size < 3:
        raise ValueError("Need at least three positive scores to estimate b-value.")

    # CRITICAL FIX 1: Quantize FIRST, then select m_min from grid points only
    positive = _quantize_to_grid(positive)
    grid = DEFAULT_SCALE_POINTS  # [0.5, 1.0, ..., 4.0]
    observed_max = float(np.max(positive))

    # Candidate m_min values: grid points where at least _MIN_N_FOR_B_FIT
    # data points lie at or above
    candidate_mmins = []
    for gp in grid:
        if gp >= observed_max:
            continue
        n_above = int((positive >= gp - 1e-9).sum())
        if n_above >= _MIN_N_FOR_B_FIT:
            candidate_mmins.append(float(gp))

    if not candidate_mmins:
        # Fall back to lowest grid point with any data (last resort)
        candidate_mmins = [float(np.min(positive))]

    # Select m_min that minimizes KS distance to fitted exponential tail
    best_m_min = min(
        candidate_mmins,
        key=lambda m_min: _ks_distance(positive, _estimate_b(positive, m_min), m_min),
    )

    if method == "ratio":
        ratios = ratio_test(positive, list(_support_for(positive, best_m_min)))["ratios"]
        delta = 0.5
        valid_ratios = [ratio for ratio in ratios if ratio > 0]
        b = -float(np.mean(np.log10(valid_ratios))) / delta if valid_ratios else _estimate_b(positive, best_m_min)
    else:
        b = _estimate_b(positive, best_m_min)

    prepared = _prepare_scores(positive, m_min=best_m_min)
    bootstrap_draws = max(200, min(n_bootstrap, 2000))
    samples = np.random.default_rng(42).choice(
        prepared,
        size=(bootstrap_draws, prepared.size),
        replace=True,
    )
    bootstrap_bs = np.array([_estimate_b(sample, best_m_min) for sample in samples], dtype=float)
    bootstrap_bs = bootstrap_bs[np.isfinite(bootstrap_bs)]
    if bootstrap_bs.size > 0:
        lower, upper = np.percentile(bootstrap_bs, [2.5, 97.5])
    else:
        lower = upper = b
    return BValue(
        model_name=model_name,
        b=float(b),
        b_ci_lower=float(lower),
        b_ci_upper=float(upper),
        m_min=float(best_m_min),
        n_above_mmin=int(prepared.size),
        method=method,
    )


def ratio_test(scores: np.ndarray, scale_points: list[float]) -> dict[str, Any]:
    """
    Gutenberg-Richter ratio test.
    """
    prepared = np.asarray(scores, dtype=float)
    scale = sorted(scale_points)
    cumulative = [float((prepared >= point).sum()) for point in scale]
    ratios: list[float] = []
    for current, nxt in zip(cumulative, cumulative[1:]):
        if current <= 0:
            continue
        ratios.append(nxt / current)
    cv = float(np.std(ratios) / np.mean(ratios)) if ratios and np.mean(ratios) else float("inf")
    return {"ratios": ratios, "cv": cv, "supports_power_law": cv < 0.3}


def predict_catastrophic_rate(
    b_easy: float,
    n_easy_errors: int,
    m_min: float,
    target_magnitude: float = 3.0,
) -> float:
    """
    Predict N(M >= target) from a b-value estimated on easy queries.
    """
    return float(n_easy_errors * (10 ** (-b_easy * (target_magnitude - m_min))))


def run_prediction_experiment(
    easy_scores: dict[str, np.ndarray],
    hard_scores: dict[str, np.ndarray],
    target_magnitude: float = 3.0,
) -> list[PredictionResult]:
    """
    Full Experiment 3 pipeline across all models.
    """
    results: list[PredictionResult] = []
    for model_name in sorted(easy_scores.keys() & hard_scores.keys()):
        b_value = estimate_b_value(easy_scores[model_name], model_name=model_name)
        easy_positive = np.asarray(easy_scores[model_name], dtype=float)
        easy_positive = easy_positive[easy_positive >= b_value.m_min]
        observed = float(np.sum(np.asarray(hard_scores[model_name], dtype=float) >= target_magnitude))
        predicted = predict_catastrophic_rate(
            b_easy=b_value.b,
            n_easy_errors=int(easy_positive.size),
            m_min=b_value.m_min,
            target_magnitude=target_magnitude,
        )
        ratio = 1.0 if observed == 0 and predicted == 0 else (
            float("inf") if observed == 0 else predicted / observed
        )
        results.append(
            PredictionResult(
                model_name=model_name,
                b_easy=b_value.b,
                predicted_catastrophic=predicted,
                observed_catastrophic=observed,
                ratio=ratio,
                within_1_5x=abs(ratio - 1.0) < 0.5 if math.isfinite(ratio) else False,
            )
        )
    return results


def compute_prediction_metrics(results: list[PredictionResult]) -> dict[str, Any]:
    """
    Compute aggregate metrics for prediction results.
    """
    predicted = [result.predicted_catastrophic for result in results]
    observed = [result.observed_catastrophic for result in results]
    rho, _ = stats.spearmanr(predicted, observed) if len(results) > 1 else (1.0, 0.0)
    return {
        "spearman_rho": float(rho),
        "within_1_5x_fraction": (
            sum(result.within_1_5x for result in results) / len(results) if results else 0.0
        ),
        "calibration_data": [
            {
                "model_name": result.model_name,
                "predicted": result.predicted_catastrophic,
                "observed": result.observed_catastrophic,
            }
            for result in results
        ],
    }


def _pmf_for_fit(support: np.ndarray, fit: FitResult) -> np.ndarray:
    params = fit.parameters
    if fit.distribution == "power_law":
        raw = support ** (-params["beta"])
    elif fit.distribution == "truncated_power_law":
        raw = (support ** (-params["beta"])) * np.exp(-params["lambda"] * support)
    elif fit.distribution == "lognormal":
        raw = stats.lognorm.pdf(support, s=params["sigma"], scale=np.exp(params["mu"]))
    elif fit.distribution == "exponential":
        raw = np.exp(-params["lambda"] * support)
    elif fit.distribution == "stretched_exp":
        raw = np.exp(-params["lambda"] * np.power(support, params["gamma"]))
    else:
        raise ValueError(f"Unknown distribution: {fit.distribution}")
    return _normalised_pmf(raw)


def vuong_test(
    scores: np.ndarray,
    fit1: FitResult,
    fit2: FitResult,
) -> dict[str, Any]:
    """
    Vuong's likelihood ratio test between two non-nested models.
    """
    prepared = _prepare_scores(scores, m_min=float(np.min(scores[scores > 0])))
    support = _support_for(prepared, float(np.min(prepared)))
    pmf1 = _pmf_for_fit(support, fit1)
    pmf2 = _pmf_for_fit(support, fit2)
    ll_diff = np.log(pmf1[np.searchsorted(support, prepared)]) - np.log(
        pmf2[np.searchsorted(support, prepared)]
    )
    mean_diff = float(np.mean(ll_diff))
    std_diff = float(np.std(ll_diff, ddof=1)) if len(ll_diff) > 1 else 0.0
    if std_diff == 0:
        z_stat = 0.0
    else:
        z_stat = math.sqrt(len(ll_diff)) * mean_diff / std_diff
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    preferred = fit1.distribution if mean_diff >= 0 else fit2.distribution
    return {
        "z_statistic": float(z_stat),
        "p_value": p_value,
        "preferred": preferred,
        "significant": p_value < 0.05,
    }
