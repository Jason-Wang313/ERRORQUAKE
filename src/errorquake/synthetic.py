"""Experiment 0: synthetic pipeline validation."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from errorquake.analyze import BValue, FitResult, estimate_b_value, fit_all_distributions, ratio_test
from errorquake.generate import GenerationRequestError, RateLimiter, _call_nim_text
from errorquake.magnitude import SCALE_11, parse_judge_output, render_judge_prompt, resolve_scores
from errorquake.queries import Query, load_queries
from errorquake.report import plot_judge_confusion_matrix, plot_synthetic_validation
from errorquake.utils import ProjectConfig, now_iso, read_jsonl, setup_logger, write_jsonl

SCALE_POINTS = np.array([level.score for level in SCALE_11], dtype=float)
NONZERO_SCALE_POINTS = SCALE_POINTS[SCALE_POINTS > 0]
DEFAULT_DISTRIBUTIONS = ("power_law", "exponential", "uniform")
PRIMARY_JUDGE_MODEL = "meta/llama-3.1-405b-instruct"
SECONDARY_JUDGE_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
GENERATION_MODEL = "deepseek-ai/deepseek-v3.2"

SYNTHETIC_SPECS: dict[str, dict[str, Any]] = {
    "power_law": {"prefix": "PL", "params": {"b": 1.0}},
    "exponential": {"prefix": "EXP", "params": {"lambda_": 0.8}},
    "uniform": {"prefix": "UNI", "params": {}},
}

MODEL_FALLBACKS: dict[str, list[str]] = {
    "qwen/qwen3-next-80b": ["qwen/qwen3-next-80b-a3b-instruct"],
}


def _rewrite_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    write_jsonl(path, [record])


def _quantize_scalar(score: float, scale_points: np.ndarray | None = None) -> float:
    support = SCALE_POINTS if scale_points is None else np.asarray(scale_points, dtype=float)
    return float(support[np.argmin(np.abs(support - float(score)))])


def _quantize_scores(scores: Sequence[float], scale_points: np.ndarray | None = None) -> np.ndarray:
    return np.array([_quantize_scalar(score, scale_points=scale_points) for score in scores], dtype=float)


def _distribution_weights(distribution: str, support: np.ndarray, **params: Any) -> np.ndarray:
    if distribution == "power_law":
        exponent = float(params.get("b", 1.0))
        return support ** (-exponent)
    if distribution == "exponential":
        lambda_ = float(params.get("lambda_", 0.8))
        return np.exp(-lambda_ * support)
    if distribution == "uniform":
        return np.ones_like(support)
    raise ValueError(f"Unknown synthetic distribution: {distribution}")


def _score_counts(scores: Sequence[float], scale_points: np.ndarray | None = None) -> dict[str, int]:
    support = NONZERO_SCALE_POINTS if scale_points is None else np.asarray(scale_points, dtype=float)
    rounded = _quantize_scores(scores, scale_points=support)
    return {f"{point:.1f}": int(np.sum(np.isclose(rounded, point))) for point in support}


def generate_synthetic_scores(
    distribution: str,
    n: int = 500,
    scale_points: list[float] | None = None,
    rng: np.random.Generator | None = None,
    **params: Any,
) -> np.ndarray:
    """Generate discrete target severity scores from a known distribution."""
    generator = rng or np.random.default_rng(42)
    support = np.array(scale_points or NONZERO_SCALE_POINTS.tolist(), dtype=float)
    weights = _distribution_weights(distribution, support, **params)
    probabilities = weights / weights.sum()
    draws = generator.choice(support, size=n, replace=True, p=probabilities)
    return _quantize_scores(draws, scale_points=support)


def _target_scores_path(output_dir: Path, distribution: str) -> Path:
    return output_dir / f"target_scores_{distribution}.json"


def _responses_path(output_dir: Path, distribution: str) -> Path:
    return output_dir / f"responses_{distribution}.jsonl"


def _primary_scores_path(output_dir: Path, distribution: str) -> Path:
    return output_dir / f"scores_primary_{distribution}.jsonl"


def _secondary_scores_path(output_dir: Path, distribution: str) -> Path:
    return output_dir / f"scores_secondary_{distribution}.jsonl"


def _resolved_scores_path(output_dir: Path, distribution: str) -> Path:
    return output_dir / f"resolved_scores_{distribution}.jsonl"


def _load_target_scores(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get("scores", [])
    else:
        values = payload
    return np.array(values, dtype=float)


def _save_target_scores(
    path: Path,
    *,
    distribution: str,
    params: dict[str, Any],
    scores: Sequence[float],
) -> None:
    payload = {
        "distribution": distribution,
        "parameters": params,
        "scores": [float(score) for score in scores],
        "bin_counts": _score_counts(scores),
        "generated_at": now_iso(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _tiers_for_target_score(score: float) -> tuple[int, ...]:
    if score <= 1.0:
        return (1, 2)
    if score <= 2.5:
        return (2, 3)
    return (3, 4, 5)


def select_source_queries(
    target_scores: Sequence[float],
    queries: list[Query],
    rng: np.random.Generator | None = None,
) -> list[Query]:
    """Sample real source queries, stratified by the target severity band."""
    if not queries:
        raise ValueError("Need at least one source query for synthetic generation.")

    generator = rng or np.random.default_rng(42)
    by_tier: dict[int, list[Query]] = {tier: [] for tier in range(1, 6)}
    for query in queries:
        by_tier.setdefault(query.tier, []).append(query)

    chosen: list[Query] = []
    for score in target_scores:
        tiers = _tiers_for_target_score(float(score))
        pool = [query for tier in tiers for query in by_tier.get(tier, [])]
        if not pool:
            pool = queries
        index = int(generator.integers(0, len(pool)))
        chosen.append(pool[index])
    return chosen


def _condensed_rubric() -> str:
    lines = []
    for level in SCALE_11:
        lines.append(f"{level.score:.1f}: {level.label} — {level.definition}")
    return "\n".join(lines)


def _synthetic_response_prompt(query: Query, target_score: float) -> str:
    return (
        "You are generating a test response for a scoring calibration experiment.\n\n"
        f"Question: {query.question}\n"
        f"Correct answer: {query.ground_truth}\n\n"
        f"Generate a response to this question that contains a factual error at "
        f"severity level {target_score:.1f} on the following scale:\n\n"
        f"{_condensed_rubric()}\n\n"
        "Your response should:\n"
        "- Sound like a natural LLM response.\n"
        "- Contain exactly one primary error at the target severity.\n"
        "- For score 0.5: introduce a tiny imprecision.\n"
        "- For score 1.0-2.0: keep the response plausible and natural.\n"
        "- For score 3.0+: fabricate a fact, entity, or source.\n\n"
        "Respond with ONLY the simulated LLM response. No meta-commentary."
    )


def _compact_judge_prompt(
    *,
    question: str,
    ground_truth: str,
    model_response: str,
    judge_role: str,
) -> str:
    role_label = "Primary" if judge_role == "primary" else "Secondary"
    return (
        f"{role_label} ERRORQUAKE Judge\n\n"
        "Assess the factual error severity of the model response.\n"
        "Score the MOST severe factual error only.\n"
        "If any factual error exists, the minimum score is 0.5.\n\n"
        "Severity scale:\n"
        f"{_condensed_rubric()}\n\n"
        f"Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"Model response: {model_response}\n\n"
        "Return JSON only:\n"
        '{"chain_of_thought":"brief reasoning","identified_errors":["..."],'
        '"score":1.0,"confidence":"high"}'
    )


def _ordered_score_output_rule() -> str:
    return (
        "\n\nAdditional output rule: Return the JSON keys in this exact order: "
        "score, confidence, identified_errors, chain_of_thought. Keep "
        "chain_of_thought under 25 words and identified_errors to at most 2 items."
    )


def _candidate_models(model_id: str) -> list[str]:
    candidates: list[str] = []
    for candidate in [model_id, *MODEL_FALLBACKS.get(model_id, [])]:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


async def _generate_single_response(
    *,
    query: Query,
    target_score: float,
    model_id: str,
    rate_limiter: RateLimiter,
) -> str:
    prompt = _synthetic_response_prompt(query, target_score)
    last_error: Exception | None = None
    for candidate_model in _candidate_models(model_id):
        try:
            return (
                await _call_nim_text(
                    model_id=candidate_model,
                    prompt=prompt,
                    max_tokens=300,
                    rate_limiter=rate_limiter,
                    temperature=0.7,
                    timeout_s=90,
                )
            ).strip()
        except GenerationRequestError as exc:
            last_error = exc
            if "404" in str(exc) and candidate_model != _candidate_models(model_id)[-1]:
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Synthetic generation failed before any model call was attempted.")


def _judge_output_prefix(judge_role: str) -> str:
    if judge_role not in {"primary", "secondary"}:
        raise ValueError(f"Unknown judge role: {judge_role}")
    return judge_role


async def _score_single_response(
    *,
    judge_role: str,
    judge_model: str,
    question: str,
    ground_truth: str,
    model_response: str,
    rate_limiter: RateLimiter,
) -> dict[str, Any]:
    full_prompt = render_judge_prompt(
        scale=SCALE_11,
        question=question,
        ground_truth=ground_truth,
        model_response=model_response,
        judge_role=judge_role,
    )
    deepseek_primary_prompt = full_prompt + _ordered_score_output_rule()
    compact_prompt = _compact_judge_prompt(
        question=question,
        ground_truth=ground_truth,
        model_response=model_response,
        judge_role=judge_role,
    )
    candidates = _candidate_models(judge_model)
    if judge_role == "secondary":
        prompt_variants = [
            {"prompt": compact_prompt, "max_tokens": 350, "timeout_s": 60, "attempts": 2},
            {"prompt": compact_prompt, "max_tokens": 220, "timeout_s": 45, "attempts": 1},
            {"prompt": full_prompt, "max_tokens": 900, "timeout_s": 80, "attempts": 1},
        ]
    else:
        prompt_variants = [
            {"prompt": full_prompt, "max_tokens": 1400, "timeout_s": 120, "attempts": 3},
        ]
    for candidate_model in candidates:
        active_variants = prompt_variants
        if judge_role == "primary" and "deepseek-ai/deepseek-v3.2" in candidate_model:
            active_variants = [
                {
                    "prompt": deepseek_primary_prompt.replace(
                        _ordered_score_output_rule(),
                        (
                            "\n\nAdditional output rule: Return the JSON keys in this exact order: "
                            "score, confidence, identified_errors, chain_of_thought. Keep "
                            "chain_of_thought under 12 words and identified_errors to exactly 1 short item."
                        ),
                    ),
                    "max_tokens": 80,
                    "timeout_s": 25,
                    "attempts": 2,
                },
                {"prompt": deepseek_primary_prompt, "max_tokens": 140, "timeout_s": 35, "attempts": 1},
            ]
        for variant in active_variants:
            for _ in range(int(variant["attempts"])):
                try:
                    raw = await _call_nim_text(
                        model_id=candidate_model,
                        prompt=str(variant["prompt"]),
                        max_tokens=int(variant["max_tokens"]),
                        rate_limiter=rate_limiter,
                        temperature=0.0,
                        timeout_s=int(variant["timeout_s"]),
                    )
                except GenerationRequestError as exc:
                    if "404" in str(exc) and candidate_model != candidates[-1]:
                        break
                    if any(token in str(exc).lower() for token in ("timed out", "timeout", "rate limit")):
                        await asyncio.sleep(3)
                        continue
                    raise
                parsed = parse_judge_output(raw)
                score = parsed.get("score")
                if score is None:
                    await asyncio.sleep(1)
                    continue
                return {
                    "score": _quantize_scalar(float(score)),
                    "score_raw": float(score),
                    "confidence": str(parsed.get("confidence", "unknown")),
                    "chain_of_thought": str(parsed.get("chain_of_thought", "")),
                    "identified_errors": list(parsed.get("identified_errors", [])),
                    "model_id_used": candidate_model,
                }
    raise RuntimeError(f"{judge_role} judge failed to return a usable score.")


async def generate_synthetic_responses(
    *,
    distribution: str,
    target_scores: Sequence[float],
    queries: list[Query],
    output_path: Path,
    model_id: str,
    rpm: int,
    resume: bool,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    """Generate one synthetic response per target score and persist checkpoints."""
    spec = SYNTHETIC_SPECS[distribution]
    existing = read_jsonl(output_path) if resume else []
    existing_ids = {str(record.get("synthetic_id")) for record in existing}
    ordered_records = list(existing)
    rate_limiter = RateLimiter(rpm=rpm)

    for index, (target_score, query) in enumerate(zip(target_scores, queries), start=1):
        synthetic_id = f"SYN_{spec['prefix']}_{index:04d}"
        if synthetic_id in existing_ids:
            continue
        response_text = await _generate_single_response(
            query=query,
            target_score=float(target_score),
            model_id=model_id,
            rate_limiter=rate_limiter,
        )
        record = {
            "synthetic_id": synthetic_id,
            "distribution": distribution,
            "target_score": float(target_score),
            "source_query_id": query.id,
            "question": query.question,
            "ground_truth": query.ground_truth,
            "synthetic_response": response_text,
            "generation_model": model_id,
            "generated_at": now_iso(),
        }
        existing_ids.add(synthetic_id)
        ordered_records.append(record)
        _append_jsonl(output_path, record)
        if logger and (index % 50 == 0 or index == len(target_scores)):
            logger.info("[%s generate] %s/%s complete", distribution, index, len(target_scores))

    return ordered_records


async def score_synthetic_responses(
    *,
    distribution: str,
    responses: list[dict[str, Any]],
    output_path: Path,
    judge_role: str,
    judge_model: str,
    rpm: int,
    resume: bool,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    """Score synthetic responses with one judge model, checkpointing each item."""
    prefix = _judge_output_prefix(judge_role)
    existing = read_jsonl(output_path) if resume else []
    existing_ids = {str(record.get("synthetic_id")) for record in existing}
    ordered_records = list(existing)
    rate_limiter = RateLimiter(rpm=rpm)

    max_passes = 12
    for pass_index in range(1, max_passes + 1):
        pass_progress = 0
        remaining = 0
        for index, response in enumerate(responses, start=1):
            synthetic_id = str(response.get("synthetic_id"))
            if synthetic_id in existing_ids:
                continue
            remaining += 1
            try:
                scored = await _score_single_response(
                    judge_role=judge_role,
                    judge_model=judge_model,
                    question=str(response.get("question", "")),
                    ground_truth=str(response.get("ground_truth", "")),
                    model_response=str(response.get("synthetic_response", "")),
                    rate_limiter=rate_limiter,
                )
            except (GenerationRequestError, RuntimeError) as exc:
                if logger:
                    logger.warning(
                        "[%s %s] pass %s failed for %s: %s",
                        distribution,
                        judge_role,
                        pass_index,
                        synthetic_id,
                        exc,
                    )
                continue
            record = {
                "synthetic_id": synthetic_id,
                "distribution": distribution,
                "target_score": float(response.get("target_score", 0.0)),
                f"{prefix}_score": float(scored["score"]),
                f"{prefix}_score_raw": float(scored["score_raw"]),
                f"{prefix}_confidence": scored["confidence"],
                f"{prefix}_chain_of_thought": scored["chain_of_thought"],
                f"{prefix}_identified_errors": scored["identified_errors"],
                f"{prefix}_judge_model": str(scored.get("model_id_used", judge_model)),
                "scored_at": now_iso(),
            }
            existing_ids.add(synthetic_id)
            ordered_records.append(record)
            _append_jsonl(output_path, record)
            pass_progress += 1
            completed = len(existing_ids)
            if logger and (completed % 50 == 0 or completed == len(responses)):
                logger.info(
                    "[%s %s] %s/%s complete",
                    distribution,
                    judge_role,
                    completed,
                    len(responses),
                )
        if len(existing_ids) >= len(responses):
            return ordered_records
        if logger:
            logger.warning(
                "[%s %s] pass %s finished with %s remaining and %s new scores",
                distribution,
                judge_role,
                pass_index,
                len(responses) - len(existing_ids),
                pass_progress,
            )
        if pass_progress == 0 and pass_index < max_passes:
            await asyncio.sleep(10)

    missing = len(responses) - len(existing_ids)
    raise RuntimeError(
        f"{distribution} {judge_role} scoring incomplete after {max_passes} passes; {missing} items remain."
    )

    return ordered_records


def _resolve_distribution_scores(
    *,
    distribution: str,
    responses: list[dict[str, Any]],
    primary_scores: list[dict[str, Any]],
    secondary_scores: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    primary_by_id = {str(record.get("synthetic_id")): record for record in primary_scores}
    secondary_by_id = {str(record.get("synthetic_id")): record for record in secondary_scores}
    resolved_records: list[dict[str, Any]] = []

    for response in responses:
        synthetic_id = str(response.get("synthetic_id"))
        primary = primary_by_id[synthetic_id]
        secondary = secondary_by_id[synthetic_id]
        resolved_score, method = resolve_scores(
            float(primary["primary_score"]),
            float(secondary["secondary_score"]),
        )
        resolved_records.append(
            {
                "synthetic_id": synthetic_id,
                "distribution": distribution,
                "target_score": float(response["target_score"]),
                "primary_score": float(primary["primary_score"]),
                "secondary_score": float(secondary["secondary_score"]),
                "final_score": _quantize_scalar(float(resolved_score)),
                "resolution_method": method,
                "source_query_id": response["source_query_id"],
                "question": response["question"],
                "ground_truth": response["ground_truth"],
                "synthetic_response": response["synthetic_response"],
            }
        )

    _rewrite_jsonl(output_path, resolved_records)
    return resolved_records


def _uniform_family_recovered(fits: list[FitResult]) -> bool:
    if not fits:
        return False
    if all(fit.chi2_pvalue < 0.05 for fit in fits):
        return True
    if len(fits) == 1:
        return True
    return (fits[1].bic - fits[0].bic) <= 2.0


def validate_pipeline_recovery(
    true_distribution: str,
    true_params: dict[str, Any],
    recovered_fits: list[FitResult],
    *,
    final_scores: Sequence[float] | None = None,
    target_scores: Sequence[float] | None = None,
    b_estimate: BValue | None = None,
    tolerance_b: float = 0.3,
    calibration_threshold: float = 0.70,
) -> dict[str, Any]:
    """Check whether the pipeline recovers the known generating process."""
    best_fit_family = recovered_fits[0].distribution if recovered_fits else "unknown"
    if true_distribution == "power_law":
        family_recovered = best_fit_family in {"power_law", "truncated_power_law"}
    elif true_distribution == "uniform":
        family_recovered = _uniform_family_recovered(recovered_fits)
    else:
        family_recovered = best_fit_family == true_distribution

    recovered_b = b_estimate.b if b_estimate is not None else recovered_fits[0].parameters.get("beta")
    true_b = true_params.get("b")
    param_recovered = None
    if true_distribution == "power_law":
        param_recovered = (
            recovered_b is not None
            and true_b is not None
            and abs(float(recovered_b) - float(true_b)) <= tolerance_b
        )

    judge_correlation = 1.0
    judge_calibration_pass = True
    if final_scores is not None and target_scores is not None and len(final_scores) > 1:
        rho, _ = stats.spearmanr(np.asarray(target_scores, dtype=float), np.asarray(final_scores, dtype=float))
        judge_correlation = 0.0 if np.isnan(rho) else float(rho)
        judge_calibration_pass = judge_correlation >= calibration_threshold

    if not judge_calibration_pass:
        verdict = "FAIL"
    elif family_recovered and (param_recovered is None or param_recovered):
        verdict = "PASS"
    elif family_recovered:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    details = (
        f"Best fit: {best_fit_family}. "
        f"Judge-target Spearman rho={judge_correlation:.3f}. "
        f"Family recovered={family_recovered}."
    )
    if true_distribution == "power_law":
        details += f" Recovered b={recovered_b!r} vs true b={true_b!r}."

    return {
        "distribution": true_distribution,
        "family_recovered": family_recovered,
        "true_family": true_distribution,
        "best_fit_family": best_fit_family,
        "param_recovered": param_recovered,
        "true_b": true_b,
        "recovered_b": recovered_b,
        "judge_correlation": judge_correlation,
        "judge_calibration_pass": judge_calibration_pass,
        "verdict": verdict,
        "details": details,
    }


def _fit_ranking(fits: list[FitResult]) -> list[dict[str, Any]]:
    return [
        {
            "distribution": fit.distribution,
            "bic": float(fit.bic),
            "aic": float(fit.aic),
            "chi2_stat": float(fit.chi2_stat),
            "chi2_pvalue": float(fit.chi2_pvalue),
            "parameters": {key: float(value) for key, value in fit.parameters.items()},
        }
        for fit in fits
    ]


def _per_level_accuracy(target_scores: Sequence[float], final_scores: Sequence[float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    targets = np.asarray(target_scores, dtype=float)
    finals = np.asarray(final_scores, dtype=float)
    for point in NONZERO_SCALE_POINTS:
        mask = np.isclose(targets, point)
        if not np.any(mask):
            payload[f"{point:.1f}"] = 0.0
            continue
        payload[f"{point:.1f}"] = float(np.mean(np.abs(finals[mask] - targets[mask]) <= 0.5))
    return payload


def _confusion_matrix(
    target_scores: Sequence[float],
    judged_scores: Sequence[float],
    scale_points: np.ndarray | None = None,
) -> list[list[int]]:
    support = SCALE_POINTS if scale_points is None else np.asarray(scale_points, dtype=float)
    target_q = _quantize_scores(target_scores, scale_points=support)
    judged_q = _quantize_scores(judged_scores, scale_points=support)
    matrix: list[list[int]] = []
    for target in support:
        row = []
        target_mask = np.isclose(target_q, target)
        for judged in support:
            row.append(int(np.sum(target_mask & np.isclose(judged_q, judged))))
        matrix.append(row)
    return matrix


def _summarise_resolution_methods(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("resolution_method")) for record in records)
    total = max(len(records), 1)
    return {
        "primary_only": int(counts.get("primary", 0)),
        "averaged": int(counts.get("average", 0)),
        "human_required": int(counts.get("human_required", 0)),
        "human_required_fraction": float(counts.get("human_required", 0) / total),
    }


def _distribution_report(
    *,
    distribution: str,
    target_scores: np.ndarray,
    resolved_records: list[dict[str, Any]],
) -> dict[str, Any]:
    final_scores = np.array([record["final_score"] for record in resolved_records], dtype=float)
    primary_scores = np.array([record["primary_score"] for record in resolved_records], dtype=float)
    fits = fit_all_distributions(final_scores, model_name=f"synthetic-{distribution}")
    b_estimate = estimate_b_value(final_scores, model_name=f"synthetic-{distribution}")
    recovery = validate_pipeline_recovery(
        distribution,
        SYNTHETIC_SPECS[distribution]["params"],
        fits,
        final_scores=final_scores,
        target_scores=target_scores,
        b_estimate=b_estimate,
    )
    ratio = ratio_test(final_scores, NONZERO_SCALE_POINTS.tolist())
    rho_primary, _ = stats.spearmanr(target_scores, primary_scores)
    rho_primary = 0.0 if np.isnan(rho_primary) else float(rho_primary)

    return {
        "true_parameters": SYNTHETIC_SPECS[distribution]["params"],
        "true_b": SYNTHETIC_SPECS[distribution]["params"].get("b"),
        "recovered_b": float(b_estimate.b) if distribution == "power_law" else None,
        "best_fit_family": fits[0].distribution,
        "bic_ranking": _fit_ranking(fits),
        "ratio_test": ratio,
        "judge_target_correlation": float(recovery["judge_correlation"]),
        "primary_target_correlation": rho_primary,
        "per_level_accuracy": _per_level_accuracy(target_scores, final_scores),
        "systematic_bias": float(np.mean(final_scores - target_scores)),
        "resolution_stats": _summarise_resolution_methods(resolved_records),
        "target_bin_counts": _score_counts(target_scores),
        "final_bin_counts": _score_counts(final_scores),
        "recovery": recovery,
        "verdict": recovery["verdict"],
    }


def _analysis_dir(results_dir: Path) -> Path:
    return results_dir if results_dir.name == "analysis" else results_dir / "analysis"


def _overall_verdict(results: dict[str, dict[str, Any]]) -> str:
    verdicts = {payload["verdict"] for payload in results.values()}
    if "FAIL" in verdicts:
        return "NO-GO"
    if "MARGINAL" in verdicts:
        return "MARGINAL"
    return "GO"


def _ensure_queries(config: ProjectConfig) -> list[Query]:
    queries = load_queries(config.data_dir)
    if not queries:
        raise RuntimeError("No source queries found under data/queries.")
    return queries


async def run_experiment_0(
    config: ProjectConfig,
    *,
    n: int = 500,
    distributions: Sequence[str] | None = None,
    output_dir: Path | None = None,
    rpm: int = 35,
    skip_scoring: bool = False,
    score_only: bool = False,
    analyze_only: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the full synthetic validation pipeline for Phase 2A."""
    synthetic_dir = output_dir or (config.data_dir / "synthetic")
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = _analysis_dir(config.results_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("errorquake.synthetic", synthetic_dir / "logs")
    selected = list(distributions or DEFAULT_DISTRIBUTIONS)
    invalid = [name for name in selected if name not in SYNTHETIC_SPECS]
    if invalid:
        raise ValueError(f"Unknown synthetic distributions: {', '.join(invalid)}")

    queries = _ensure_queries(config)
    rng = np.random.default_rng(42)
    all_primary_scores: list[float] = []
    all_final_scores: list[float] = []
    all_target_scores: list[float] = []
    results: dict[str, dict[str, Any]] = {}
    all_resolved_records: list[dict[str, Any]] = []

    for distribution in selected:
        spec = SYNTHETIC_SPECS[distribution]
        target_path = _target_scores_path(synthetic_dir, distribution)
        responses_path = _responses_path(synthetic_dir, distribution)
        primary_path = _primary_scores_path(synthetic_dir, distribution)
        secondary_path = _secondary_scores_path(synthetic_dir, distribution)
        resolved_path = _resolved_scores_path(synthetic_dir, distribution)

        if target_path.exists() and (resume or score_only or analyze_only):
            target_scores = _load_target_scores(target_path)
        else:
            target_scores = generate_synthetic_scores(
                distribution,
                n=n,
                rng=rng,
                **spec["params"],
            )
            _save_target_scores(
                target_path,
                distribution=distribution,
                params=spec["params"],
                scores=target_scores,
            )
        logger.info("[%s] target bin counts: %s", distribution, _score_counts(target_scores))

        responses = read_jsonl(responses_path) if responses_path.exists() else []
        if not score_only and not analyze_only:
            source_queries = select_source_queries(target_scores, queries, rng=rng)
            responses = await generate_synthetic_responses(
                distribution=distribution,
                target_scores=target_scores,
                queries=source_queries,
                output_path=responses_path,
                model_id=config.generation_model,
                rpm=rpm,
                resume=resume,
                logger=logger,
            )

        if skip_scoring:
            continue

        if not analyze_only:
            if not responses:
                raise RuntimeError(f"No synthetic responses found for {distribution}.")
            primary_scores = await score_synthetic_responses(
                distribution=distribution,
                responses=responses,
                output_path=primary_path,
                judge_role="primary",
                judge_model=config.primary_judge,
                rpm=rpm,
                resume=resume or score_only,
                logger=logger,
            )
            secondary_scores = await score_synthetic_responses(
                distribution=distribution,
                responses=responses,
                output_path=secondary_path,
                judge_role="secondary",
                judge_model=config.secondary_judge,
                rpm=rpm,
                resume=resume or score_only,
                logger=logger,
            )
            resolved_records = _resolve_distribution_scores(
                distribution=distribution,
                responses=responses,
                primary_scores=primary_scores,
                secondary_scores=secondary_scores,
                output_path=resolved_path,
            )
        else:
            resolved_records = read_jsonl(resolved_path)
            if not resolved_records:
                responses = read_jsonl(responses_path)
                primary_scores = read_jsonl(primary_path)
                secondary_scores = read_jsonl(secondary_path)
                if not responses or not primary_scores or not secondary_scores:
                    raise RuntimeError(
                        f"Analyze-only requires existing responses and score files for {distribution}."
                    )
                resolved_records = _resolve_distribution_scores(
                    distribution=distribution,
                    responses=responses,
                    primary_scores=primary_scores,
                    secondary_scores=secondary_scores,
                    output_path=resolved_path,
                )

        results[distribution] = _distribution_report(
            distribution=distribution,
            target_scores=target_scores,
            resolved_records=resolved_records,
        )
        all_resolved_records.extend(resolved_records)
        all_primary_scores.extend(record["primary_score"] for record in resolved_records)
        all_final_scores.extend(record["final_score"] for record in resolved_records)
        all_target_scores.extend(record["target_score"] for record in resolved_records)

    if skip_scoring:
        return {
            "experiment": "Experiment 0: Synthetic Pipeline Validation",
            "date": now_iso().split("T", 1)[0],
            "status": "responses_generated",
            "n_per_distribution": n,
            "distributions": selected,
            "output_dir": str(synthetic_dir),
        }

    target_vs_primary = _confusion_matrix(all_target_scores, all_primary_scores)
    target_vs_final = _confusion_matrix(all_target_scores, all_final_scores)
    report = {
        "experiment": "Experiment 0: Synthetic Pipeline Validation",
        "date": now_iso().split("T", 1)[0],
        "n_per_distribution": n,
        "generation_model": config.generation_model,
        "primary_judge": config.primary_judge,
        "secondary_judge": config.secondary_judge,
        "scale": config.active_scale,
        "results": results,
        "overall_verdict": _overall_verdict(results),
        "score_disagreement_stats": _summarise_resolution_methods(all_resolved_records),
        "overall_per_level_accuracy": _per_level_accuracy(all_target_scores, all_final_scores),
        "overall_systematic_bias": float(
            np.mean(np.asarray(all_final_scores, dtype=float) - np.asarray(all_target_scores, dtype=float))
        ),
        "judge_confusion_matrix": {
            "scale_points": [float(point) for point in SCALE_POINTS],
            "target_vs_primary": target_vs_primary,
            "target_vs_final": target_vs_final,
        },
    }

    report_path = analysis_dir / "experiment_0_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_synthetic_validation(report, output_path=config.figures_dir / "fig11_synthetic_validation.pdf")
    plot_synthetic_validation(report, output_path=config.figures_dir / "fig11_synthetic_validation.png")
    plot_judge_confusion_matrix(report, output_path=config.figures_dir / "judge_confusion_matrix.png")
    return report


def default_synthetic_config() -> ProjectConfig:
    """Return the Phase 2A default model configuration."""
    return ProjectConfig(
        generation_model=GENERATION_MODEL,
        primary_judge=PRIMARY_JUDGE_MODEL,
        secondary_judge=SECONDARY_JUDGE_MODEL,
    )
