"""
Automatic Verification Layer (Plan Item A)
==========================================
Identifies queries with unambiguously verifiable ground truth,
builds a rule-based severity scorer, and validates judge b-values
against automatically-derived b-values.

This gives ERRORQUAKE a dual-validation architecture matching ERBench's
automatic verification paradigm.
"""

import json
import math
import re
import string
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "results" / "evaluations_10k"
SCORES_DIR = REPO / "results" / "scores_10k"
ANALYSIS_DIR = REPO / "results" / "analysis"
OUTPUT_DIR = REPO / "results" / "analysis" / "oral_upgrade"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_numbers(text):
    """Extract all numbers from text."""
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', text)]


def extract_key_entities(text):
    """Extract capitalized multi-word entities (rough NER)."""
    # Find sequences of capitalized words
    entities = re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
    return [e.strip() for e in entities if len(e) > 3]


def classify_query_verifiability(query_id, question, ground_truth):
    """
    Classify whether a query has unambiguously verifiable ground truth.

    Returns (is_verifiable, verification_type, key_facts) or (False, None, None).

    Verification types:
    - 'numeric': answer contains a specific number
    - 'named_entity': answer is a specific named entity
    - 'binary': answer is yes/no or a specific fact
    - 'enumeration': answer is a specific count or list
    """
    if not ground_truth or len(ground_truth.strip()) < 10:
        return False, None, None

    gt = ground_truth.strip()
    q = question.strip().lower()

    # Extract numbers from ground truth
    gt_numbers = extract_numbers(gt)

    # Type 1: Numeric answer (specific number in GT)
    # Questions like "What is the weight of...", "How many...", "What percentage..."
    numeric_q = any(kw in q for kw in [
        'how many', 'how much', 'what is the', 'what was the', 'what percentage',
        'what is the number', 'what is the population', 'what is the weight',
        'what is the distance', 'what is the area', 'what is the rate',
        'what is the odds', 'what is the average', 'what is the precise',
        'how old', 'how long', 'how far', 'how high', 'how deep',
    ])
    if gt_numbers and numeric_q and len(gt_numbers) <= 3:
        return True, 'numeric', {'key_numbers': gt_numbers, 'gt_snippet': gt[:200]}

    # Type 2: Named entity answer
    # Questions like "Who...", "What is the name of...", "Which..."
    entity_q = any(kw in q for kw in [
        'who is', 'who was', 'who discovered', 'who invented', 'who created',
        'what is the name', 'what is the capital', 'which country', 'which city',
        'which organ', 'which treaty', 'which element', 'which planet',
        'what currency', 'what language', 'what is the term',
    ])
    gt_entities = extract_key_entities(gt)
    if entity_q and gt_entities:
        return True, 'named_entity', {'key_entities': gt_entities[:3], 'gt_snippet': gt[:200]}

    # Type 3: Specific factual answer with numbers
    if gt_numbers and len(gt) < 300:
        return True, 'numeric', {'key_numbers': gt_numbers, 'gt_snippet': gt[:200]}

    # Type 4: Short definitive answer
    if len(gt) < 150 and any(kw in q for kw in ['what is', 'what was', 'what are', 'define']):
        return True, 'definitional', {'gt_snippet': gt[:200]}

    return False, None, None


def auto_score_response(question, ground_truth, response, verification_type, key_facts):
    """
    Automatically score a response against ground truth.

    Returns auto_severity score (0-4):
    - 0: correct (key facts match)
    - 1: minor error (close numbers, right entity type)
    - 2: substitution (wrong entity of correct type, moderate numerical error)
    - 3: fabrication (invented number, wrong entity entirely)
    - 4: harmful fabrication (confident assertion of entirely wrong critical fact)
    """
    if not response or len(response.strip()) < 5:
        return None, "empty_response"

    gt_norm = normalize_text(ground_truth)
    resp_norm = normalize_text(response)

    if verification_type == 'numeric':
        gt_numbers = key_facts.get('key_numbers', [])
        resp_numbers = extract_numbers(response)

        if not gt_numbers:
            return None, "no_gt_numbers"

        # Check if the primary GT number appears in the response
        primary_gt = gt_numbers[0]

        if not resp_numbers:
            # Response has no numbers — likely a deflection or non-answer
            # Check if response mentions the right topic at all
            if any(w in resp_norm for w in gt_norm.split()[:5]):
                return 2.0, "topic_correct_no_number"
            return 3.0, "off_topic_no_number"

        # Find the closest response number to the GT number
        best_ratio = float('inf')
        for rn in resp_numbers:
            if primary_gt != 0:
                ratio = abs(rn - primary_gt) / abs(primary_gt)
            elif rn == 0:
                ratio = 0
            else:
                ratio = abs(rn)
            best_ratio = min(best_ratio, ratio)

        if best_ratio < 0.05:  # Within 5%
            return 0.0, "exact_match"
        elif best_ratio < 0.15:  # Within 15%
            return 0.5, "close_match"
        elif best_ratio < 0.30:  # Within 30%
            return 1.0, "moderate_deviation"
        elif best_ratio < 0.50:  # Within 50%
            return 1.5, "significant_deviation"
        elif best_ratio < 1.0:  # Within 100%
            return 2.0, "large_deviation"
        elif best_ratio < 5.0:  # Within 5x
            return 2.5, "order_of_magnitude_close"
        elif best_ratio < 100.0:  # Within 100x
            return 3.0, "order_of_magnitude_off"
        else:
            return 3.5, "extreme_deviation"

    elif verification_type == 'named_entity':
        gt_entities = key_facts.get('key_entities', [])

        # Check if any GT entity appears in the response
        for entity in gt_entities:
            if entity.lower() in resp_norm:
                return 0.0, "entity_found"

        # Check partial entity matches
        for entity in gt_entities:
            words = entity.lower().split()
            if any(w in resp_norm for w in words if len(w) > 3):
                return 1.0, "partial_entity_match"

        # Entity not found — check if response gives a different specific entity
        resp_entities = extract_key_entities(response)
        if resp_entities:
            return 2.5, "wrong_entity"

        return 3.0, "entity_not_found"

    elif verification_type == 'definitional':
        # Check keyword overlap
        gt_words = set(gt_norm.split())
        resp_words = set(resp_norm.split())
        # Remove stop words
        stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'of', 'in', 'to', 'and', 'for', 'it', 'that', 'this', 'with', 'on', 'at', 'by', 'from', 'as', 'or', 'be'}
        gt_content = gt_words - stop
        resp_content = resp_words - stop

        if not gt_content:
            return None, "no_content_words"

        overlap = len(gt_content & resp_content) / len(gt_content)

        if overlap >= 0.6:
            return 0.0, "high_overlap"
        elif overlap >= 0.4:
            return 1.0, "moderate_overlap"
        elif overlap >= 0.2:
            return 2.0, "low_overlap"
        else:
            return 3.0, "minimal_overlap"

    return None, "unclassified"


def main():
    print("=" * 70)
    print("AUTOMATIC VERIFICATION LAYER")
    print("=" * 70)

    # Step 1: Identify verifiable queries
    print("\n--- Step 1: Identifying verifiable queries ---")

    # Load one evaluation file to get all queries with ground truth
    all_queries = {}
    for ef in sorted(EVAL_DIR.glob("*.jsonl")):
        with open(ef) as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["query_id"]
                if qid not in all_queries:
                    all_queries[qid] = {
                        "question": rec.get("question", ""),
                        "ground_truth": rec.get("ground_truth", ""),
                        "domain": rec.get("domain", ""),
                        "tier": rec.get("tier", ""),
                    }
        break  # Only need one file for query metadata

    print(f"  Total unique queries: {len(all_queries)}")

    verifiable = {}
    by_type = defaultdict(int)
    for qid, qdata in all_queries.items():
        is_ver, vtype, kfacts = classify_query_verifiability(
            qid, qdata["question"], qdata["ground_truth"])
        if is_ver:
            verifiable[qid] = {
                **qdata,
                "verification_type": vtype,
                "key_facts": kfacts,
            }
            by_type[vtype] += 1

    print(f"  Verifiable queries: {len(verifiable)} ({len(verifiable)/len(all_queries)*100:.1f}%)")
    for vt, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {vt}: {count}")

    # Step 2: Auto-score all model responses on verifiable queries
    print("\n--- Step 2: Auto-scoring model responses ---")

    auto_scores = defaultdict(list)  # model -> list of (qid, auto_score, judge_score)
    total_scored = 0
    total_skipped = 0

    # Load judge scores
    judge_scores = {}  # (model, qid) -> final_score
    for sf in sorted(SCORES_DIR.glob("*.jsonl")):
        model = sf.stem
        with open(sf) as f:
            for line in f:
                rec = json.loads(line)
                fs = rec.get("final_score")
                if fs is not None:
                    judge_scores[(model, rec["query_id"])] = fs

    # Score responses
    for ef in sorted(EVAL_DIR.glob("*.jsonl")):
        model = ef.stem
        with open(ef) as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["query_id"]
                if qid not in verifiable:
                    continue

                vdata = verifiable[qid]
                auto_sev, reason = auto_score_response(
                    rec.get("question", ""),
                    rec.get("ground_truth", ""),
                    rec.get("response_text", ""),
                    vdata["verification_type"],
                    vdata["key_facts"],
                )

                j_score = judge_scores.get((model, qid))

                if auto_sev is not None and j_score is not None:
                    auto_scores[model].append({
                        "query_id": qid,
                        "auto_score": auto_sev,
                        "judge_score": j_score,
                        "reason": reason,
                        "domain": vdata["domain"],
                    })
                    total_scored += 1
                else:
                    total_skipped += 1

    print(f"  Total auto-scored: {total_scored}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Models with auto-scores: {len(auto_scores)}")

    # Step 3: Compute auto-b values per model
    print("\n--- Step 3: Computing auto-b values ---")

    from errorquake.analyze import estimate_b_value

    auto_b_values = {}
    judge_b_values = {}

    for model, items in sorted(auto_scores.items()):
        auto_arr = np.array([it["auto_score"] for it in items])
        judge_arr = np.array([it["judge_score"] for it in items])

        # Filter to errors only (score > 0)
        auto_errors = auto_arr[auto_arr > 0]
        judge_errors = judge_arr[judge_arr > 0]

        if len(auto_errors) >= 30:
            try:
                ab = estimate_b_value(auto_errors, model_name=model)
                auto_b_values[model] = ab.b
            except Exception:
                auto_b_values[model] = None

        if len(judge_errors) >= 30:
            try:
                jb = estimate_b_value(judge_errors, model_name=model)
                judge_b_values[model] = jb.b
            except Exception:
                judge_b_values[model] = None

        n_auto_err = len(auto_errors)
        n_judge_err = len(judge_errors)
        ab_str = f"{auto_b_values.get(model, 'N/A'):.3f}" if auto_b_values.get(model) else "N/A"
        jb_str = f"{judge_b_values.get(model, 'N/A'):.3f}" if judge_b_values.get(model) else "N/A"
        print(f"  {model:30s} n_items={len(items):5d} auto_err={n_auto_err:4d} auto_b={ab_str:>6s} judge_b={jb_str:>6s}")

    # Step 4: Correlate auto-b with judge-b
    print("\n--- Step 4: Auto-b vs Judge-b correlation ---")

    paired_models = []
    paired_auto_b = []
    paired_judge_b = []

    for model in auto_b_values:
        if auto_b_values[model] is not None and model in judge_b_values and judge_b_values[model] is not None:
            paired_models.append(model)
            paired_auto_b.append(auto_b_values[model])
            paired_judge_b.append(judge_b_values[model])

    if len(paired_models) >= 5:
        rho, p = stats.spearmanr(paired_auto_b, paired_judge_b)
        pearson_r, pearson_p = stats.pearsonr(paired_auto_b, paired_judge_b)

        print(f"  Paired models: {len(paired_models)}")
        print(f"  Spearman rho(auto_b, judge_b) = {rho:.4f} (p={p:.6f})")
        print(f"  Pearson  r(auto_b, judge_b)   = {pearson_r:.4f} (p={pearson_p:.6f})")
        print(f"  Target: rho >= 0.80 for strong validation")

        if rho >= 0.80:
            verdict = "STRONG_VALIDATION"
        elif rho >= 0.60:
            verdict = "MODERATE_VALIDATION"
        elif rho >= 0.40:
            verdict = "WEAK_VALIDATION"
        else:
            verdict = "VALIDATION_FAILED"

        print(f"  Verdict: {verdict}")
    else:
        rho, p = None, None
        pearson_r, pearson_p = None, None
        verdict = "INSUFFICIENT_DATA"
        print(f"  Only {len(paired_models)} paired models — need at least 5")

    # Step 5: Per-score agreement (auto vs judge)
    print("\n--- Step 5: Auto-judge agreement by score band ---")

    all_auto = []
    all_judge = []
    for model, items in auto_scores.items():
        for it in items:
            all_auto.append(it["auto_score"])
            all_judge.append(it["judge_score"])

    all_auto = np.array(all_auto)
    all_judge = np.array(all_judge)

    # Agreement within ±1.0
    agree_1 = np.mean(np.abs(all_auto - all_judge) <= 1.0)
    agree_05 = np.mean(np.abs(all_auto - all_judge) <= 0.5)
    item_rho, item_p = stats.spearmanr(all_auto, all_judge)

    print(f"  Total items: {len(all_auto)}")
    print(f"  Agreement within ±0.5: {agree_05*100:.1f}%")
    print(f"  Agreement within ±1.0: {agree_1*100:.1f}%")
    print(f"  Item-level Spearman rho: {item_rho:.4f} (p={item_p:.6f})")

    # Save results
    results = {
        "summary": {
            "total_queries": len(all_queries),
            "verifiable_queries": len(verifiable),
            "verifiable_fraction": round(len(verifiable) / len(all_queries), 4),
            "by_type": dict(by_type),
            "total_auto_scored": total_scored,
            "models_with_auto_scores": len(auto_scores),
        },
        "b_value_correlation": {
            "n_paired_models": len(paired_models),
            "spearman_rho": round(rho, 4) if rho is not None else None,
            "spearman_p": round(p, 6) if p is not None else None,
            "pearson_r": round(pearson_r, 4) if pearson_r is not None else None,
            "pearson_p": round(pearson_p, 6) if pearson_p is not None else None,
            "verdict": verdict,
            "target": "rho >= 0.80 for strong validation",
        },
        "item_level_agreement": {
            "n_items": len(all_auto),
            "agreement_within_0.5": round(float(agree_05), 4),
            "agreement_within_1.0": round(float(agree_1), 4),
            "item_spearman_rho": round(float(item_rho), 4),
            "item_spearman_p": round(float(item_p), 8),
        },
        "per_model": {
            model: {
                "n_items": len(items),
                "n_auto_errors": sum(1 for it in items if it["auto_score"] > 0),
                "auto_b": round(auto_b_values.get(model, 0), 4) if auto_b_values.get(model) else None,
                "judge_b_on_subset": round(judge_b_values.get(model, 0), 4) if judge_b_values.get(model) else None,
            }
            for model, items in sorted(auto_scores.items())
        },
        "paired_b_values": [
            {"model": m, "auto_b": round(ab, 4), "judge_b": round(jb, 4)}
            for m, ab, jb in zip(paired_models, paired_auto_b, paired_judge_b)
        ],
    }

    output_path = OUTPUT_DIR / "auto_verification.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
