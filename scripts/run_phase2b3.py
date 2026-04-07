"""
Phase 2b+3: Mini Model Pilot + Human Rating Set

Evaluates 3 diverse models on 200 stratified pilot queries, scores all 600
responses with dual LLM judges, generates diagnostics, and prepares a
100-item blind human rating set.

Usage:
    python scripts/run_phase2b3.py                # Run all steps
    python scripts/run_phase2b3.py --step 1       # Run only step 1
    python scripts/run_phase2b3.py --step 2       # Run only step 2 (needs step 1)
    python scripts/run_phase2b3.py --step 3       # Run only step 3 (needs step 2)
    python scripts/run_phase2b3.py --step 4       # Run only steps 4-6 (needs step 3)
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -- project imports --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.generate import RateLimiter, _call_nim_text
from errorquake.magnitude import (
    SCALE_11,
    parse_judge_output,
    render_judge_prompt,
    resolve_scores,
)
from errorquake.queries import DOMAINS, TIERS, Query, load_queries
from errorquake.utils import now_iso, read_jsonl, write_jsonl

# ============================================================
# CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PILOT_DIR = DATA_DIR / "pilot"
RESULTS_DIR = PROJECT_ROOT / "results" / "analysis"
FIGURES_DIR = PROJECT_ROOT / "figures"

PILOT_QUERIES_PATH = PILOT_DIR / "pilot_queries.jsonl"
PILOT_SCORED_PATH = PILOT_DIR / "pilot_scored.jsonl"
HUMAN_RATING_SET_PATH = PILOT_DIR / "human_rating_set.jsonl"
HUMAN_RATING_FORM_PATH = PILOT_DIR / "human_rating_form.csv"
DIAGNOSTICS_PATH = RESULTS_DIR / "pilot_diagnostics.json"

RPM = 35  # NIM free-tier rate limit

# -- Pilot models --
PILOT_MODELS = {
    "gemma-3-4b": "google/gemma-3-4b-it",
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",
    "deepseek-v3.2": "deepseek-ai/deepseek-v3.2",
}

# -- Judge assignment (self-score swap for deepseek) --
# model_name -> (primary_judge_model_id, secondary_judge_model_id)
JUDGE_ASSIGNMENT: dict[str, tuple[str, str]] = {
    "gemma-3-4b": ("deepseek-ai/deepseek-v3.2", "qwen/qwen3-next-80b-a3b-instruct"),
    "llama-3.1-70b": ("deepseek-ai/deepseek-v3.2", "qwen/qwen3-next-80b-a3b-instruct"),
    "deepseek-v3.2": ("meta/llama-3.1-405b-instruct", "qwen/qwen3-next-80b-a3b-instruct"),
}

SYSTEM_PROMPT = "You are a helpful assistant. Answer the following question accurately and concisely."


# ============================================================
# STEP 1: SELECT PILOT QUERIES
# ============================================================

def step1_select_pilot_queries() -> list[dict]:
    """Sample 200 queries: 5 tiers × 8 domains × 5 per cell."""
    print("\n" + "=" * 60)
    print("STEP 1: Selecting 200 pilot queries")
    print("=" * 60)

    all_queries = load_queries(DATA_DIR)
    print(f"Loaded {len(all_queries)} total queries")

    # Group by (domain, tier)
    by_cell: dict[tuple[str, int], list[Query]] = defaultdict(list)
    for q in all_queries:
        by_cell[(q.domain, q.tier)].append(q)

    rng = random.Random(42)
    pilot: list[Query] = []
    for domain in DOMAINS:
        for tier in TIERS:
            cell = by_cell[(domain, tier)]
            if len(cell) < 5:
                print(f"  WARNING: {domain}_T{tier} has only {len(cell)} queries, taking all")
                pilot.extend(cell)
            else:
                pilot.extend(rng.sample(cell, 5))

    print(f"Selected {len(pilot)} pilot queries")

    # Verify stratification
    dist = Counter((q.domain, q.tier) for q in pilot)
    for domain in DOMAINS:
        for tier in TIERS:
            count = dist.get((domain, tier), 0)
            if count != 5:
                print(f"  WARNING: {domain}_T{tier} has {count} queries (expected 5)")

    # Save
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    records = [q.to_dict() for q in pilot]
    # Overwrite (not append) for idempotency
    PILOT_QUERIES_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"Saved to {PILOT_QUERIES_PATH}")
    return records


def load_pilot_queries() -> list[dict]:
    """Load pilot queries from JSONL."""
    return read_jsonl(PILOT_QUERIES_PATH)


# ============================================================
# STEP 2: EVALUATE 3 MODELS
# ============================================================

async def _evaluate_one_model(
    model_name: str,
    model_id: str,
    queries: list[dict],
    rate_limiter: RateLimiter,
    concurrency: int = 8,
) -> Path:
    """Evaluate a single model on all pilot queries with concurrency + checkpointing."""
    output_path = PILOT_DIR / f"responses_{model_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed query IDs (skip errors too)
    existing = read_jsonl(output_path)
    completed_ids = {r["query_id"] for r in existing if "query_id" in r and not r.get("error")}
    pending = [q for q in queries if q["id"] not in completed_ids]

    if not pending:
        print(f"  {model_name}: all {len(queries)} already done, skipping")
        return output_path

    print(f"  {model_name}: {len(completed_ids)} done, {len(pending)} remaining (concurrency={concurrency})")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counter = {"done": len(completed_ids)}

    async def _do_one(q: dict) -> None:
        async with semaphore:
            question = q["question"]
            start = time.perf_counter()
            try:
                response_text = await _call_nim_text(
                    model_id=model_id,
                    prompt=f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:",
                    max_tokens=500,
                    rate_limiter=rate_limiter,
                    temperature=0.0,
                    timeout_s=120,
                )
                error = None
            except Exception as exc:
                response_text = ""
                error = str(exc)
                print(f"    ERROR on {q['id']}: {error[:100]}")

            latency_ms = (time.perf_counter() - start) * 1000

            record = {
                "query_id": q["id"],
                "model_name": model_name,
                "model_id": model_id,
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "domain": q["domain"],
                "tier": q["tier"],
                "response_text": response_text,
                "latency_ms": round(latency_ms, 1),
                "timestamp": now_iso(),
                "error": error,
            }
            async with write_lock:
                write_jsonl(output_path, [record])
                counter["done"] += 1
                done = counter["done"]
                total = len(queries)
                if done % 20 == 0 or done == total:
                    print(f"    {model_name}: {done}/{total} ({done*100/total:.0f}%)")

    await asyncio.gather(*[_do_one(q) for q in pending])

    return output_path


async def step2_evaluate_models(queries: list[dict]) -> dict[str, Path]:
    """Evaluate all 3 pilot models sequentially."""
    print("\n" + "=" * 60)
    print("STEP 2: Evaluating 3 models (600 NIM calls)")
    print("=" * 60)

    rate_limiter = RateLimiter(rpm=RPM)
    results: dict[str, Path] = {}

    for model_name, model_id in PILOT_MODELS.items():
        print(f"\n  Starting {model_name} ({model_id})...")
        results[model_name] = await _evaluate_one_model(
            model_name, model_id, queries, rate_limiter
        )

    return results


# ============================================================
# STEP 3: SCORE ALL RESPONSES WITH DUAL JUDGES
# ============================================================

async def _score_single_response(
    response: dict,
    judge_model_id: str,
    judge_role: str,
    rate_limiter: RateLimiter,
) -> dict:
    """Call a judge model to score a single response."""
    scale = SCALE_11
    prompt = render_judge_prompt(
        scale=scale,
        question=response["question"],
        ground_truth=response["ground_truth"],
        model_response=response["response_text"],
        judge_role=judge_role,
    )

    try:
        raw = await _call_nim_text(
            model_id=judge_model_id,
            prompt=prompt,
            max_tokens=1024,
            rate_limiter=rate_limiter,
            temperature=0.0,
            timeout_s=120,
        )
        parsed = parse_judge_output(raw)
    except Exception as exc:
        parsed = {
            "score": None,
            "confidence": "error",
            "chain_of_thought": str(exc),
            "identified_errors": [],
        }

    return parsed


async def step3_score_responses() -> Path:
    """Score all 600 responses with dual judges."""
    print("\n" + "=" * 60)
    print("STEP 3: Scoring 600 responses with dual judges (1200 NIM calls)")
    print("=" * 60)

    # Load all responses
    all_responses: list[dict] = []
    for model_name in PILOT_MODELS:
        path = PILOT_DIR / f"responses_{model_name}.jsonl"
        responses = read_jsonl(path)
        # Filter out errors
        valid = [r for r in responses if not r.get("error")]
        print(f"  {model_name}: {len(valid)} valid responses (of {len(responses)})")
        all_responses.extend(valid)

    print(f"  Total responses to score: {len(all_responses)}")

    # Load already-scored
    existing = read_jsonl(PILOT_SCORED_PATH)
    scored_keys = {
        (r["query_id"], r["model_name"])
        for r in existing
        if "query_id" in r and "model_name" in r
    }
    pending = [
        r for r in all_responses
        if (r["query_id"], r["model_name"]) not in scored_keys
    ]

    if not pending:
        print("  All responses already scored, skipping")
        return PILOT_SCORED_PATH

    print(f"  Already scored: {len(scored_keys)}, remaining: {len(pending)}")

    rate_limiter = RateLimiter(rpm=RPM)
    semaphore = asyncio.Semaphore(4)  # 4 concurrent items = 8 judge calls in flight
    write_lock = asyncio.Lock()
    counter = {"done": len(scored_keys)}
    total = len(all_responses)

    async def _score_one(response: dict) -> None:
        async with semaphore:
            model_name = response["model_name"]
            primary_judge_id, secondary_judge_id = JUDGE_ASSIGNMENT[model_name]

            # Run both judges concurrently for the same item
            primary_result, secondary_result = await asyncio.gather(
                _score_single_response(response, primary_judge_id, "primary", rate_limiter),
                _score_single_response(response, secondary_judge_id, "secondary", rate_limiter),
            )

            # Resolve scores
            p_score = primary_result.get("score")
            s_score = secondary_result.get("score")

            if p_score is not None and s_score is not None:
                final_score, resolution_method = resolve_scores(float(p_score), float(s_score))
            elif p_score is not None:
                final_score, resolution_method = float(p_score), "primary_only"
            elif s_score is not None:
                final_score, resolution_method = float(s_score), "secondary_only"
            else:
                final_score, resolution_method = None, "both_failed"

            scored_record = {
                "query_id": response["query_id"],
                "model_name": model_name,
                "question": response["question"],
                "ground_truth": response["ground_truth"],
                "model_response": response["response_text"],
                "domain": response.get("domain", ""),
                "tier": response.get("tier", 0),
                "primary_judge": primary_judge_id,
                "primary_score": p_score,
                "primary_confidence": primary_result.get("confidence", "unknown"),
                "primary_chain_of_thought": primary_result.get("chain_of_thought", ""),
                "secondary_judge": secondary_judge_id,
                "secondary_score": s_score,
                "secondary_confidence": secondary_result.get("confidence", "unknown"),
                "secondary_chain_of_thought": secondary_result.get("chain_of_thought", ""),
                "final_score": final_score,
                "resolution_method": resolution_method,
            }
            async with write_lock:
                write_jsonl(PILOT_SCORED_PATH, [scored_record])
                counter["done"] += 1
                done = counter["done"]
                if done % 20 == 0 or done == total:
                    print(f"    Scored: {done}/{total} ({done*100/total:.0f}%)")

    await asyncio.gather(*[_score_one(r) for r in pending])

    return PILOT_SCORED_PATH


# ============================================================
# STEP 4: DIAGNOSTICS
# ============================================================

def _score_histogram_bar(count: int, max_count: int, width: int = 40) -> str:
    if max_count == 0:
        return ""
    bar_len = int(count / max_count * width)
    return "\u2588" * bar_len


def step4_diagnostics(scored: list[dict]) -> dict:
    """Compute and report pilot diagnostics."""
    print("\n" + "=" * 60)
    print("STEP 4: Generating Diagnostics")
    print("=" * 60)

    # Filter to records with valid final scores
    valid = [r for r in scored if r.get("final_score") is not None]
    print(f"  Valid scored records: {len(valid)}")

    diagnostics: dict = {}

    # --- 4.1 Per-Model Error Rates ---
    print("\n--- 4.1 Per-Model Error Rates ---")
    error_rates: dict[str, dict] = {}
    for model_name in PILOT_MODELS:
        model_records = [r for r in valid if r["model_name"] == model_name]
        errors = [r for r in model_records if float(r["final_score"]) > 0.0]
        total = len(model_records)
        rate = len(errors) / total * 100 if total > 0 else 0
        error_rates[model_name] = {
            "errors": len(errors),
            "total": total,
            "rate_pct": round(rate, 1),
        }
        print(f"  {model_name:>20s}: {len(errors)}/{total} errors ({rate:.1f}%)")

    diagnostics["per_model_error_rates"] = error_rates

    # Sanity check
    rates = {m: d["rate_pct"] for m, d in error_rates.items()}
    if rates.get("gemma-3-4b", 0) <= rates.get("deepseek-v3.2", 100):
        print("  *** WARNING: gemma error rate <= deepseek. Something may be wrong! ***")

    # --- 4.2 Score Distribution Per Model ---
    print("\n--- 4.2 Score Distributions ---")
    score_levels = [s.score for s in SCALE_11]
    distributions: dict[str, dict] = {}

    for model_name in PILOT_MODELS:
        model_records = [r for r in valid if r["model_name"] == model_name]
        scores = [float(r["final_score"]) for r in model_records]
        # Quantize to nearest scale point
        counts: dict[str, int] = {}
        for level in score_levels:
            count = sum(1 for s in scores if abs(s - level) < 0.25)
            counts[f"{level:.1f}"] = count
        distributions[model_name] = counts
        print(f"\n  {model_name} score distribution:")
        max_count = max(counts.values()) if counts else 1
        for level_str, count in counts.items():
            bar = _score_histogram_bar(count, max_count)
            print(f"    {level_str}: {bar} ({count})")

    diagnostics["score_distributions"] = distributions

    # Save figure
    _plot_score_distributions(distributions)

    # --- 4.3 Judge Disagreement Statistics ---
    print("\n--- 4.3 Judge Disagreement ---")
    both_valid = [
        r for r in valid
        if r.get("primary_score") is not None and r.get("secondary_score") is not None
    ]
    total_both = len(both_valid)

    agree = sum(1 for r in both_valid if abs(float(r["primary_score"]) - float(r["secondary_score"])) < 0.5)
    averaged = sum(
        1 for r in both_valid
        if 0.5 <= abs(float(r["primary_score"]) - float(r["secondary_score"])) < 1.5
    )
    human_req = sum(
        1 for r in both_valid
        if abs(float(r["primary_score"]) - float(r["secondary_score"])) >= 1.5
    )

    disagreement = {
        "total_dual_scored": total_both,
        "agreed_pct": round(agree / total_both * 100, 1) if total_both else 0,
        "averaged_pct": round(averaged / total_both * 100, 1) if total_both else 0,
        "human_required_pct": round(human_req / total_both * 100, 1) if total_both else 0,
        "agreed": agree,
        "averaged": averaged,
        "human_required": human_req,
    }
    diagnostics["judge_disagreement"] = disagreement

    print(f"  Total items: {total_both}")
    print(f"  Primary = Final (|diff| < 0.5):   {agree} ({disagreement['agreed_pct']}%)")
    print(f"  Averaged (0.5 ≤ |diff| < 1.5):    {averaged} ({disagreement['averaged_pct']}%)")
    print(f"  Human required (|diff| ≥ 1.5):     {human_req} ({disagreement['human_required_pct']}%)")

    if disagreement["human_required_pct"] >= 15:
        print("  *** WARNING: Human-required rate ≥ 15%. Judges disagree too much! ***")

    # --- 4.4 Judge Score Usage ---
    print("\n--- 4.4 Judge Score Usage ---")
    judge_usage: dict[str, dict] = {}
    for role in ("primary", "secondary"):
        key = f"{role}_score"
        scores_by_judge = [float(r[key]) for r in both_valid if r.get(key) is not None]
        usage: dict[str, int] = {}
        for level in score_levels:
            count = sum(1 for s in scores_by_judge if abs(s - level) < 0.25)
            usage[f"{level:.1f}"] = count
        judge_usage[role] = usage

        total_j = len(scores_by_judge)
        print(f"\n  {role.title()} judge score usage:")
        parts = [f"{lvl}: {cnt}" for lvl, cnt in usage.items()]
        print(f"    {' | '.join(parts)}")

        # Check for compression
        if total_j > 0:
            max_bin = max(usage.values())
            if max_bin > total_j * 0.5:
                dominant = [lvl for lvl, cnt in usage.items() if cnt == max_bin][0]
                print(f"    *** COMPRESSION WARNING: {role} puts {max_bin}/{total_j} "
                      f"({max_bin/total_j*100:.0f}%) into score {dominant} ***")

    diagnostics["judge_score_usage"] = judge_usage

    # --- 4.5 Per-Tier Error Rates ---
    print("\n--- 4.5 Per-Tier Error Rates ---")
    tier_rates: dict[str, dict] = {}
    for tier in TIERS:
        tier_records = [r for r in valid if int(r.get("tier", 0)) == tier]
        errors = [r for r in tier_records if float(r["final_score"]) > 0.0]
        total_t = len(tier_records)
        rate = len(errors) / total_t * 100 if total_t > 0 else 0
        tier_rates[f"T{tier}"] = {
            "errors": len(errors),
            "total": total_t,
            "rate_pct": round(rate, 1),
        }
        print(f"  T{tier}: {len(errors)}/{total_t} errors ({rate:.1f}%)")

    diagnostics["per_tier_error_rates"] = tier_rates

    # Check monotonicity
    tier_rate_values = [tier_rates[f"T{t}"]["rate_pct"] for t in TIERS]
    monotonic = all(a <= b for a, b in zip(tier_rate_values, tier_rate_values[1:]))
    diagnostics["tier_calibration_monotonic"] = monotonic
    if not monotonic:
        print("  *** WARNING: Tier error rates NOT monotonically increasing! ***")
    else:
        print("  Tier calibration: monotonically increasing ✓")

    # --- 4.6 Magnitude-Frequency Preview (gemma only) ---
    print("\n--- 4.6 Magnitude-Frequency Preview ---")
    gemma_records = [
        r for r in valid
        if r["model_name"] == "gemma-3-4b" and float(r["final_score"]) > 0.0
    ]
    _plot_magnitude_frequency(gemma_records)

    # Save diagnostics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_PATH.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Diagnostics saved to {DIAGNOSTICS_PATH}")

    return diagnostics


def _plot_score_distributions(distributions: dict[str, dict]) -> None:
    """3-panel histogram of score distributions per model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    score_labels = [f"{s.score:.1f}" for s in SCALE_11]
    x = np.arange(len(score_labels))

    for ax, (model_name, counts) in zip(axes, distributions.items()):
        values = [counts.get(label, 0) for label in score_labels]
        bars = ax.bar(x, values, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Error Severity Score")
        ax.set_xticks(x)
        ax.set_xticklabels(score_labels, rotation=45)
        ax.set_ylabel("Count" if ax == axes[0] else "")
        # Add count labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    fig.suptitle("Pilot Score Distributions by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "pilot_score_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def _plot_magnitude_frequency(gemma_errors: list[dict]) -> None:
    """Log N(M) vs M plot for gemma errors."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not gemma_errors:
        print("  No gemma errors to plot magnitude-frequency.")
        return

    scores = sorted([float(r["final_score"]) for r in gemma_errors])
    magnitudes = sorted(set(scores))

    # Cumulative count: N(M) = number of errors with score >= M
    n_m = []
    for m in magnitudes:
        count = sum(1 for s in scores if s >= m)
        if count > 0:
            n_m.append((m, count))

    if not n_m:
        print("  No data for magnitude-frequency plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ms = [p[0] for p in n_m]
    ns = [p[1] for p in n_m]

    ax.semilogy(ms, ns, "o-", color="darkred", markersize=8, linewidth=2)
    ax.set_xlabel("Error Magnitude (M)", fontsize=12)
    ax.set_ylabel("N(M) = count of errors ≥ M", fontsize=12)
    ax.set_title("Magnitude-Frequency Preview (gemma-3-4b)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add a reference line for rough power-law comparison
    if len(ms) >= 2:
        log_ns = np.log10(ns)
        coeffs = np.polyfit(ms, log_ns, 1)
        fit_line = 10 ** np.polyval(coeffs, ms)
        ax.semilogy(ms, fit_line, "--", color="gray", alpha=0.6,
                     label=f"Linear fit (slope={coeffs[0]:.2f})")
        ax.legend()

    out = FIGURES_DIR / "pilot_magnitude_frequency_preview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================
# STEP 5: PREPARE HUMAN RATING SET
# ============================================================

def step5_human_rating_set(scored: list[dict]) -> list[dict]:
    """Select 100 items for blind human rating."""
    print("\n" + "=" * 60)
    print("STEP 5: Preparing Human Rating Set (100 items)")
    print("=" * 60)

    valid = [r for r in scored if r.get("final_score") is not None]
    rng = random.Random(42)

    def _sample_from_bins(records: list[dict], total: int) -> list[dict]:
        """Sample from severity bins ensuring coverage."""
        perfect = [r for r in records if float(r["final_score"]) == 0.0]
        low = [r for r in records if 0.0 < float(r["final_score"]) <= 1.0]
        mid = [r for r in records if 1.5 <= float(r["final_score"]) <= 2.5]
        high = [r for r in records if float(r["final_score"]) >= 3.0]

        result: list[dict] = []
        # Ensure at least 3 from each non-empty bin
        for bin_records in [perfect, low, mid, high]:
            n = min(3, len(bin_records))
            result.extend(rng.sample(bin_records, n))

        # Fill remaining proportionally
        remaining_target = total - len(result)
        already_ids = {r["query_id"] for r in result}
        pool = [r for r in records if r["query_id"] not in already_ids]

        if pool and remaining_target > 0:
            n_extra = min(remaining_target, len(pool))
            result.extend(rng.sample(pool, n_extra))

        return result[:total]

    human_set: list[dict] = []
    for model_name in PILOT_MODELS:
        model_records = [r for r in valid if r["model_name"] == model_name]
        sampled = _sample_from_bins(model_records, 33)
        human_set.extend(sampled)
        print(f"  {model_name}: selected {len(sampled)} items")

    # Trim or pad to exactly 100
    if len(human_set) < 100:
        already_ids = {(r["query_id"], r["model_name"]) for r in human_set}
        remaining = [
            r for r in valid
            if (r["query_id"], r["model_name"]) not in already_ids
        ]
        needed = 100 - len(human_set)
        if remaining:
            human_set.extend(rng.sample(remaining, min(needed, len(remaining))))
    human_set = human_set[:100]

    # Shuffle to prevent model-clustering
    rng.shuffle(human_set)

    # Build blind records (no scores, no model identity, no tier)
    blind_records: list[dict] = []
    for idx, r in enumerate(human_set, start=1):
        blind_records.append({
            "rating_id": f"HR_{idx:03d}",
            "query_id": r["query_id"],
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "model_response": r["model_response"],
            "model_name": "REDACTED",
        })

    # Save JSONL
    HUMAN_RATING_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_RATING_SET_PATH.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in blind_records) + "\n",
        encoding="utf-8",
    )

    # Save CSV form
    with HUMAN_RATING_FORM_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rating_id", "question", "ground_truth", "model_response",
            "score_11point", "score_7point", "score_5level", "notes",
        ])
        for rec in blind_records:
            writer.writerow([
                rec["rating_id"],
                rec["question"],
                rec["ground_truth"],
                rec["model_response"],
                "",  # score_11point (human fills)
                "",  # score_7point (human fills)
                "",  # score_5level (human fills)
                "",  # notes
            ])

    # Coverage stats
    _internal_set = human_set  # has real scores
    n_perfect = sum(1 for r in _internal_set if float(r["final_score"]) == 0.0)
    n_low = sum(1 for r in _internal_set if 0.0 < float(r["final_score"]) <= 1.0)
    n_mid = sum(1 for r in _internal_set if 1.5 <= float(r["final_score"]) <= 2.5)
    n_high = sum(1 for r in _internal_set if float(r["final_score"]) >= 3.0)

    print(f"\n  Human rating set: {len(blind_records)} items")
    print(f"  Coverage: {n_perfect} perfect, {n_low} low, {n_mid} mid, {n_high} high severity")
    print(f"  JSONL: {HUMAN_RATING_SET_PATH}")
    print(f"  CSV:   {HUMAN_RATING_FORM_PATH}")

    # Also save the key (internal, not for raters) mapping rating_id -> true info
    key_path = PILOT_DIR / "human_rating_key.jsonl"
    key_records = []
    for idx, r in enumerate(human_set, start=1):
        key_records.append({
            "rating_id": f"HR_{idx:03d}",
            "query_id": r["query_id"],
            "model_name": r["model_name"],
            "tier": r.get("tier"),
            "domain": r.get("domain"),
            "final_score": r["final_score"],
            "primary_score": r.get("primary_score"),
            "secondary_score": r.get("secondary_score"),
        })
    key_path.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in key_records) + "\n",
        encoding="utf-8",
    )

    return blind_records


# ============================================================
# STEP 6: SUMMARY REPORT
# ============================================================

def step6_summary(diagnostics: dict, human_set_size: int) -> None:
    """Print the final summary report."""
    er = diagnostics.get("per_model_error_rates", {})
    da = diagnostics.get("judge_disagreement", {})
    ju = diagnostics.get("judge_score_usage", {})
    tr = diagnostics.get("per_tier_error_rates", {})
    mono = diagnostics.get("tier_calibration_monotonic", False)

    # Score usage analysis
    primary_usage = ju.get("primary", {})
    total_primary = sum(primary_usage.values())
    sorted_levels = sorted(primary_usage.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_levels[:3] if sorted_levels else []
    uses_all = sum(1 for v in primary_usage.values() if v > 0)
    compression = any(v > total_primary * 0.5 for v in primary_usage.values()) if total_primary > 0 else False

    # Count human set coverage from key file
    key_records = read_jsonl(PILOT_DIR / "human_rating_key.jsonl")
    n_perfect = sum(1 for r in key_records if r.get("final_score") is not None and float(r["final_score"]) == 0.0)
    n_low = sum(1 for r in key_records if r.get("final_score") is not None and 0.0 < float(r["final_score"]) <= 1.0)
    n_mid = sum(1 for r in key_records if r.get("final_score") is not None and 1.5 <= float(r["final_score"]) <= 2.5)
    n_high = sum(1 for r in key_records if r.get("final_score") is not None and float(r["final_score"]) >= 3.0)

    print("\n")
    print("=" * 56)
    print("PILOT COMPLETE — SUMMARY")
    print("=" * 56)
    print()
    print(f"Models evaluated: 3 ({', '.join(PILOT_MODELS.keys())})")
    print(f"Queries per model: 200")

    total_scored = sum(d.get("total", 0) for d in er.values())
    print(f"Total responses scored: {total_scored}")
    print()

    print("ERROR RATES:")
    for model_name, data in er.items():
        print(f"  {model_name:>20s}: {data['rate_pct']}%")
    print()

    print("JUDGE DISAGREEMENT:")
    print(f"  Agreed (|diff| < 0.5):     {da.get('agreed_pct', '?')}%")
    print(f"  Averaged (0.5-1.5):        {da.get('averaged_pct', '?')}%")
    print(f"  Human required (≥1.5):     {da.get('human_required_pct', '?')}%")
    print()

    print("SCORE USAGE (primary judge):")
    print(f"  Uses {uses_all} of {len(primary_usage)} levels: {'YES' if uses_all >= 7 else 'NO'}")
    if top3:
        parts = [f"{lvl} ({cnt}/{total_primary}, {cnt/total_primary*100:.0f}%)" for lvl, cnt in top3]
        print(f"  Top 3 levels: {', '.join(parts)}")
    print(f"  Compression warning: {'YES' if compression else 'NO'}")
    print()

    print("TIER CALIBRATION:")
    tier_vals = [tr.get(f"T{t}", {}).get("rate_pct", "?") for t in TIERS]
    print(f"  T1→T5 error rates: {' → '.join(str(v) for v in tier_vals)}%")
    print(f"  Monotonically increasing: {'YES' if mono else 'NO'}")
    print()

    print("HUMAN RATING SET:")
    print(f"  {human_set_size} items saved to {HUMAN_RATING_FORM_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Coverage: {n_perfect} perfect, {n_low} low, {n_mid} mid, {n_high} high severity")
    print("  Ready for distribution to 3 human raters.")
    print()

    print("NEXT STEP:")
    print("  Distribute human_rating_form.csv to 3 raters.")
    print("  Each rater scores all 100 items on ALL THREE scales")
    print("  (11-point, 7-point, 5-level) independently.")
    print("  Bring results back for Phase 2b analysis.")
    print("=" * 56)


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2b+3: Pilot + Human Rating Set")
    parser.add_argument("--step", type=int, default=0, help="Run only this step (1-4, 0=all)")
    args = parser.parse_args()

    step = args.step

    # Step 1: Select pilot queries
    if step in (0, 1):
        step1_select_pilot_queries()

    # Load pilot queries (needed for step 2)
    pilot_queries = load_pilot_queries()
    if not pilot_queries:
        print("ERROR: No pilot queries found. Run step 1 first.")
        return

    # Step 2: Evaluate models
    if step in (0, 2):
        await step2_evaluate_models(pilot_queries)

    # Step 3: Score responses
    if step in (0, 3):
        await step3_score_responses()

    # Steps 4-6: Diagnostics + Human set + Summary
    if step in (0, 4):
        scored = read_jsonl(PILOT_SCORED_PATH)
        if not scored:
            print("ERROR: No scored data found. Run step 3 first.")
            return

        diagnostics = step4_diagnostics(scored)
        human_set = step5_human_rating_set(scored)
        step6_summary(diagnostics, len(human_set))


if __name__ == "__main__":
    asyncio.run(main())
