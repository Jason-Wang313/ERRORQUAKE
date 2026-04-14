"""
Severity Mechanism Taxonomy Classifier (Plan Item B)
=====================================================
Classifies error responses into mechanism categories.

Phase 1 (this script, no API needed):
  - Sample 1,000 items across severity bands
  - Build the classification prompt
  - Prepare the items for classification
  - Run a rule-based pre-classifier for obvious cases

Phase 2 (needs API):
  - Send items to Claude/GPT for LLM classification
  - Aggregate results
  - Build mechanism × model × severity heatmaps

Usage:
  python scripts/run_taxonomy_classifier.py --prepare    # Sample and prepare items
  python scripts/run_taxonomy_classifier.py --classify   # Run LLM classification (needs API)
  python scripts/run_taxonomy_classifier.py --analyze    # Analyze classified items
"""

import json
import re
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "results" / "evaluations_10k"
SCORES_DIR = REPO / "results" / "scores_10k"
OUTPUT_DIR = REPO / "results" / "analysis" / "oral_upgrade" / "taxonomy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAXONOMY = {
    "A_RETRIEVAL": {
        "description": "Correct concept retrieved, wrong specific fact",
        "subcategories": {
            "A1_entity_substitution": "Wrong real entity of the same semantic type (e.g., wrong battle, wrong person, wrong chemical)",
            "A2_temporal_misattribution": "Right event or entity, wrong date/year/time period",
            "A3_geographic_misattribution": "Right event or entity, wrong place/location",
            "A4_numerical_distortion": "Right entity, wrong number (magnitude off by >15%)",
        }
    },
    "B_REASONING": {
        "description": "Correct facts combined incorrectly",
        "subcategories": {
            "B1_causal_inversion": "Reverses cause and effect or gets the direction of a relationship wrong",
            "B2_scope_overgeneralization": "Applies a specific fact too broadly or conflates related but distinct concepts",
            "B3_logical_error": "Invalid deduction from correct premises",
        }
    },
    "C_GENERATION": {
        "description": "Fabricated content stated as fact",
        "subcategories": {
            "C1_entity_fabrication": "Invents a nonexistent entity (person, organization, place, law, paper)",
            "C2_citation_fabrication": "Invents a paper, statistic, study, or source that doesn't exist",
            "C3_detail_confabulation": "Fabricates specific details about a real topic (fake article count, fake ruling structure)",
            "C4_false_precision": "Adds fabricated numbers, dates, or statistics to increase apparent credibility",
        }
    },
    "D_METACOGNITIVE": {
        "description": "Wrong relationship to own knowledge state",
        "subcategories": {
            "D1_denial_deflection": "Claims the asked-about fact doesn't exist or isn't relevant when it does",
            "D2_overconfident_assertion": "States highly uncertain claim with false certainty and no hedging",
        }
    },
    "E_AMPLIFICATION": {
        "description": "Kernel of truth distorted through elaboration",
        "subcategories": {
            "E1_partial_truth_inflated": "Correct starting point but wrong elaboration or conclusion",
            "E2_analogical_overshoot": "Applies pattern from a similar domain incorrectly",
        }
    },
    "F_FORMAT": {
        "description": "Response structure issues rather than factual errors",
        "subcategories": {
            "F1_incomplete_response": "Correct start but cuts off before answering the question",
            "F2_irrelevant_response": "Responds to a different question than asked",
        }
    },
}


def build_classification_prompt(question, ground_truth, response, severity_score):
    """Build the prompt for LLM-based mechanism classification."""
    taxonomy_text = ""
    for cat_id, cat in TAXONOMY.items():
        taxonomy_text += f"\n{cat_id}: {cat['description']}\n"
        for sub_id, sub_desc in cat['subcategories'].items():
            taxonomy_text += f"  {sub_id}: {sub_desc}\n"

    prompt = f"""You are classifying the error mechanism in an LLM response that was scored {severity_score}/4.0 on a severity scale.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

MODEL RESPONSE: {response}

SEVERITY SCORE: {severity_score} (0=correct, 0.5=trivial, 1.0=minor, 2.0=significant, 3.0=fabrication, 4.0=dangerous fabrication)

TAXONOMY:
{taxonomy_text}

Classify the PRIMARY error mechanism. If the response is correct (severity 0.0), output "CORRECT".

Output ONLY a JSON object:
{{"primary_category": "X_CATEGORY", "primary_subcategory": "X#_subcategory", "secondary_category": null or "Y_CATEGORY", "confidence": "high/medium/low", "explanation": "one sentence explaining the classification"}}
"""
    return prompt


def rule_based_preclassify(question, ground_truth, response, severity_score):
    """
    Rule-based pre-classification for obvious cases.
    Returns (category, subcategory, confidence) or (None, None, None) if uncertain.
    """
    if severity_score <= 0.25:
        return "CORRECT", "CORRECT", "high"

    gt = ground_truth.lower() if ground_truth else ""
    resp = response.lower() if response else ""

    # Check for numerical errors
    gt_nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', ground_truth or "")]
    resp_nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', response or "")]

    if gt_nums and resp_nums:
        primary = gt_nums[0]
        closest_ratio = min(
            abs(rn - primary) / abs(primary) if primary != 0 else abs(rn)
            for rn in resp_nums
        ) if resp_nums else float('inf')

        if 0.15 < closest_ratio <= 1.0:
            return "A_RETRIEVAL", "A4_numerical_distortion", "medium"
        if closest_ratio > 1.0:
            return "C_GENERATION", "C4_false_precision", "medium"

    # Check for empty/truncated response
    if not response or len(response.strip()) < 20:
        return "F_FORMAT", "F1_incomplete_response", "high"

    # Check for denial patterns
    denial_patterns = [
        "not a subject", "no historical record", "there is no",
        "does not exist", "is not", "was not", "no such",
        "i cannot find", "i'm not aware", "i don't have information",
    ]
    if any(p in resp for p in denial_patterns) and severity_score >= 2.0:
        return "D_METACOGNITIVE", "D1_denial_deflection", "medium"

    return None, None, None


def prepare_items():
    """Sample 1,000 items stratified by severity band and model."""
    print("--- Preparing 1,000 items for taxonomy classification ---")

    # Load all scores and evaluations
    items = []
    for ef in sorted(EVAL_DIR.glob("*.jsonl")):
        model = ef.stem
        # Load scores for this model
        sf = SCORES_DIR / f"{model}.jsonl"
        if not sf.exists():
            continue

        model_scores = {}
        with open(sf) as f:
            for line in f:
                rec = json.loads(line)
                fs = rec.get("final_score")
                if fs is not None:
                    model_scores[rec["query_id"]] = fs

        # Load evaluations
        with open(ef) as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["query_id"]
                if qid in model_scores and model_scores[qid] > 0.25:
                    items.append({
                        "model": model,
                        "query_id": qid,
                        "domain": rec.get("domain", ""),
                        "tier": rec.get("tier", ""),
                        "question": rec.get("question", ""),
                        "ground_truth": rec.get("ground_truth", ""),
                        "response": rec.get("response_text", ""),
                        "severity_score": model_scores[qid],
                    })

    print(f"  Total error items: {len(items)}")

    # Stratified sample: 200 per severity band
    bands = {
        "0.5-1.0": (0.25, 1.01),
        "1.0-1.5": (1.01, 1.51),
        "1.5-2.0": (1.51, 2.01),
        "2.0-3.0": (2.01, 3.01),
        "3.0-4.0": (3.01, 4.01),
    }

    sampled = []
    random.seed(42)

    for band_name, (lo, hi) in bands.items():
        band_items = [it for it in items if lo <= it["severity_score"] < hi]
        n_sample = min(200, len(band_items))
        # Stratify within band by model
        by_model = defaultdict(list)
        for it in band_items:
            by_model[it["model"]].append(it)

        # Round-robin sample from models
        band_sample = []
        model_list = list(by_model.keys())
        random.shuffle(model_list)
        idx = 0
        while len(band_sample) < n_sample and idx < 10000:
            model = model_list[idx % len(model_list)]
            if by_model[model]:
                item = random.choice(by_model[model])
                by_model[model].remove(item)
                band_sample.append(item)
            idx += 1

        sampled.extend(band_sample)
        print(f"  Band {band_name}: {len(band_sample)} items from {len(band_items)} available")

    print(f"  Total sampled: {len(sampled)}")

    # Run rule-based pre-classification
    preclassified = 0
    for item in sampled:
        cat, sub, conf = rule_based_preclassify(
            item["question"], item["ground_truth"],
            item["response"], item["severity_score"])
        item["rule_category"] = cat
        item["rule_subcategory"] = sub
        item["rule_confidence"] = conf
        if cat:
            preclassified += 1

    print(f"  Rule-based pre-classified: {preclassified}/{len(sampled)}")

    # Build LLM classification prompts
    for item in sampled:
        item["classification_prompt"] = build_classification_prompt(
            item["question"], item["ground_truth"],
            item["response"], item["severity_score"])

    # Save prepared items
    output_path = OUTPUT_DIR / "taxonomy_items_prepared.jsonl"
    with open(output_path, "w") as f:
        for item in sampled:
            # Don't save the full prompt in the JSONL (too large)
            save_item = {k: v for k, v in item.items() if k != "classification_prompt"}
            f.write(json.dumps(save_item, ensure_ascii=False) + "\n")

    # Save prompts separately
    prompts_path = OUTPUT_DIR / "taxonomy_classification_prompts.jsonl"
    with open(prompts_path, "w") as f:
        for item in sampled:
            f.write(json.dumps({
                "id": f"{item['model']}__{item['query_id']}",
                "prompt": item["classification_prompt"],
            }, ensure_ascii=False) + "\n")

    # Save taxonomy definition
    tax_path = OUTPUT_DIR / "taxonomy_definition.json"
    with open(tax_path, "w") as f:
        json.dump(TAXONOMY, f, indent=2)

    # Summary stats
    rule_counts = Counter()
    for item in sampled:
        if item["rule_category"]:
            rule_counts[item["rule_category"]] += 1

    summary = {
        "total_sampled": len(sampled),
        "by_severity_band": {
            band: sum(1 for it in sampled if lo <= it["severity_score"] < hi)
            for band, (lo, hi) in bands.items()
        },
        "by_model": dict(Counter(it["model"] for it in sampled).most_common()),
        "by_domain": dict(Counter(it["domain"] for it in sampled).most_common()),
        "rule_preclassified": preclassified,
        "rule_category_counts": dict(rule_counts.most_common()),
    }

    summary_path = OUTPUT_DIR / "taxonomy_preparation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved:")
    print(f"    Items: {output_path}")
    print(f"    Prompts: {prompts_path}")
    print(f"    Taxonomy: {tax_path}")
    print(f"    Summary: {summary_path}")

    return sampled


def analyze_results():
    """Analyze classified items and build mechanism heatmaps."""
    items_path = OUTPUT_DIR / "taxonomy_items_classified.jsonl"
    if not items_path.exists():
        print(f"No classified items found at {items_path}")
        print("Run --classify first, or manually classify using the prompts in taxonomy_classification_prompts.jsonl")
        return

    items = []
    with open(items_path) as f:
        for line in f:
            items.append(json.loads(line))

    print(f"Analyzing {len(items)} classified items")

    # Build heatmaps
    # 1. Mechanism × severity band
    mech_severity = defaultdict(lambda: defaultdict(int))
    # 2. Mechanism × model
    mech_model = defaultdict(lambda: defaultdict(int))
    # 3. Mechanism × model size (small/medium/large)

    MODEL_SIZE = {
        "llama-3.2-3b-instruct": "small", "phi-3.5-mini": "small", "gemma-3-4b": "small",
        "qwen2.5-7b": "small", "llama-3.1-8b-instruct": "small", "eurollm-9b": "small",
        "solar-10.7b": "medium", "gemma-3-12b": "medium", "ministral-14b": "medium",
        "gpt-oss-20b": "medium", "mistral-small-24b": "medium",
        "gemma-2-27b": "large", "gemma-3-27b": "large", "seed-oss-36b": "large",
        "mistral-small-4-119b": "large", "mistral-medium-3": "large",
        "kimi-k2-instruct": "large", "llama-4-maverick": "large",
        "deepseek-v3.1": "large", "deepseek-v3.2": "large", "qwen3-next-80b": "large",
    }

    mech_size = defaultdict(lambda: defaultdict(int))

    for item in items:
        cat = item.get("primary_category") or item.get("rule_category")
        if not cat or cat == "CORRECT":
            continue

        sev = item["severity_score"]
        model = item["model"]
        size = MODEL_SIZE.get(model, "unknown")

        if sev <= 1.0:
            band = "low"
        elif sev <= 2.0:
            band = "medium"
        else:
            band = "high"

        mech_severity[cat][band] += 1
        mech_model[cat][model] += 1
        mech_size[cat][size] += 1

    # Print heatmaps
    print("\n=== Mechanism × Severity Band ===")
    cats = sorted(mech_severity.keys())
    print(f"{'Category':25s} {'low':>6s} {'medium':>6s} {'high':>6s} {'total':>6s}")
    for cat in cats:
        lo = mech_severity[cat]["low"]
        med = mech_severity[cat]["medium"]
        hi = mech_severity[cat]["high"]
        print(f"{cat:25s} {lo:6d} {med:6d} {hi:6d} {lo+med+hi:6d}")

    print("\n=== Mechanism × Model Size ===")
    print(f"{'Category':25s} {'small':>6s} {'medium':>6s} {'large':>6s}")
    for cat in cats:
        sm = mech_size[cat]["small"]
        md = mech_size[cat]["medium"]
        lg = mech_size[cat]["large"]
        print(f"{cat:25s} {sm:6d} {md:6d} {lg:6d}")

    # Save analysis
    analysis = {
        "mechanism_severity": {cat: dict(bands) for cat, bands in mech_severity.items()},
        "mechanism_model": {cat: dict(models) for cat, models in mech_model.items()},
        "mechanism_size": {cat: dict(sizes) for cat, sizes in mech_size.items()},
    }

    output_path = OUTPUT_DIR / "taxonomy_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Sample and prepare items")
    parser.add_argument("--classify", action="store_true", help="Run LLM classification (needs API)")
    parser.add_argument("--analyze", action="store_true", help="Analyze classified items")
    args = parser.parse_args()

    if not any([args.prepare, args.classify, args.analyze]):
        args.prepare = True  # Default to prepare

    if args.prepare:
        prepare_items()

    if args.classify:
        print("LLM classification requires API access.")
        print(f"Prompts are ready at: {OUTPUT_DIR / 'taxonomy_classification_prompts.jsonl'}")
        print("Run each prompt through Claude Opus 4 or GPT-4.1 and save results to:")
        print(f"  {OUTPUT_DIR / 'taxonomy_items_classified.jsonl'}")

    if args.analyze:
        analyze_results()


if __name__ == "__main__":
    main()
