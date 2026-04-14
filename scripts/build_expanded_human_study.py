"""
Build Expanded Human Validation Study Materials (Plan Item C)
=============================================================
Prepares the 500-item stratified sample for 3 expert raters.

Output:
  data/human_audit/expanded_study/rating_items.csv  - Items to rate (blind)
  data/human_audit/expanded_study/answer_key.json   - Ground truth for analysis
  data/human_audit/expanded_study/protocol.md       - Rating protocol
  data/human_audit/expanded_study/analysis_template.py - Post-rating analysis script
"""

import json
import csv
import random
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "results" / "evaluations_10k"
SCORES_DIR = REPO / "results" / "scores_10k"
ANALYSIS_DIR = REPO / "results" / "analysis"
OUTPUT_DIR = REPO / "data" / "human_audit" / "expanded_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 15 models spanning the full b-value range
TARGET_MODELS = [
    "seed-oss-36b",        # b=0.574 (heaviest)
    "gemma-2-27b",         # b=0.619
    "deepseek-v3.2",       # b=0.655
    "deepseek-v3.1",       # b=0.808
    "solar-10.7b",         # b=0.905
    "gemma-3-12b",         # b=0.938
    "llama-3.1-8b-instruct", # b=1.001
    "gemma-3-4b",          # b=0.979
    "kimi-k2-instruct",    # b=1.041
    "llama-3.2-3b-instruct", # b=1.046
    "eurollm-9b",          # b=1.067
    "ministral-14b",       # b=1.122
    "qwen2.5-7b",          # b=1.257
    "phi-3.5-mini",        # b=1.309
    "llama-4-maverick",    # b=1.118
]


def main():
    print("=" * 70)
    print("BUILDING EXPANDED HUMAN VALIDATION STUDY")
    print("=" * 70)

    # Load judge scores for target models
    model_scores = {}  # model -> {qid -> final_score}
    for model in TARGET_MODELS:
        sf = SCORES_DIR / f"{model}.jsonl"
        if not sf.exists():
            print(f"  WARNING: No scores for {model}")
            continue
        scores = {}
        with open(sf) as f:
            for line in f:
                rec = json.loads(line)
                fs = rec.get("final_score")
                if fs is not None:
                    scores[rec["query_id"]] = fs
        model_scores[model] = scores
        print(f"  {model}: {len(scores)} scored items")

    # Load evaluations for target models
    model_evals = {}  # model -> {qid -> eval_record}
    for model in TARGET_MODELS:
        ef = EVAL_DIR / f"{model}.jsonl"
        if not ef.exists():
            continue
        evals = {}
        with open(ef) as f:
            for line in f:
                rec = json.loads(line)
                evals[rec["query_id"]] = rec
        model_evals[model] = evals

    # Sample 500 items: ~33 per model, stratified by severity band
    # 5 severity bands × ~7 items per band per model
    severity_bands = [
        ("correct", 0, 0.26),
        ("trivial", 0.26, 1.01),
        ("minor", 1.01, 2.01),
        ("significant", 2.01, 3.01),
        ("severe", 3.01, 4.01),
    ]

    random.seed(42)
    sampled = []

    for model in TARGET_MODELS:
        if model not in model_scores or model not in model_evals:
            continue

        scores = model_scores[model]
        evals = model_evals[model]

        for band_name, lo, hi in severity_bands:
            band_qids = [qid for qid, s in scores.items()
                         if lo <= s < hi and qid in evals]

            # Sample ~7 per band (target 33 per model total)
            n_sample = min(7, len(band_qids))
            if n_sample == 0:
                continue

            selected = random.sample(band_qids, n_sample)
            for qid in selected:
                eval_rec = evals[qid]
                sampled.append({
                    "model": model,
                    "query_id": qid,
                    "domain": eval_rec.get("domain", ""),
                    "tier": eval_rec.get("tier", ""),
                    "question": eval_rec.get("question", ""),
                    "ground_truth": eval_rec.get("ground_truth", ""),
                    "response": eval_rec.get("response_text", ""),
                    "judge_score": scores[qid],
                    "severity_band": band_name,
                })

    print(f"\n  Total sampled: {len(sampled)}")
    print(f"  By severity band:")
    band_counts = defaultdict(int)
    for item in sampled:
        band_counts[item["severity_band"]] += 1
    for band, count in sorted(band_counts.items()):
        print(f"    {band}: {count}")

    print(f"  By model:")
    model_counts = defaultdict(int)
    for item in sampled:
        model_counts[item["model"]] += 1
    for model, count in sorted(model_counts.items()):
        print(f"    {model}: {count}")

    # Shuffle for blind rating
    random.shuffle(sampled)

    # Assign item IDs (blind — no model name, no judge score)
    for i, item in enumerate(sampled):
        item["item_id"] = f"EQ_{i+1:04d}"

    # Write rating CSV (blind — raters see only question, response, ground truth)
    csv_path = OUTPUT_DIR / "rating_items.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "item_id", "domain", "tier", "question", "ground_truth", "response",
            "severity_9pt", "severity_7pt", "severity_5level",
            "mechanism_category", "mechanism_subcategory", "notes"
        ])
        for item in sampled:
            writer.writerow([
                item["item_id"],
                item["domain"],
                item["tier"],
                item["question"],
                item["ground_truth"],
                item["response"],
                "",  # severity_9pt (rater fills in)
                "",  # severity_7pt (rater fills in)
                "",  # severity_5level (rater fills in)
                "",  # mechanism_category (rater fills in)
                "",  # mechanism_subcategory (rater fills in)
                "",  # notes
            ])

    # Write answer key (for analysis — NOT shown to raters)
    key_path = OUTPUT_DIR / "answer_key.json"
    answer_key = []
    for item in sampled:
        answer_key.append({
            "item_id": item["item_id"],
            "model": item["model"],
            "query_id": item["query_id"],
            "judge_score": item["judge_score"],
            "severity_band": item["severity_band"],
            "domain": item["domain"],
            "tier": item["tier"],
        })
    with open(key_path, "w") as f:
        json.dump(answer_key, f, indent=2)

    # Write protocol
    protocol_path = OUTPUT_DIR / "protocol.md"
    with open(protocol_path, "w") as f:
        f.write("""# ERRORQUAKE Expanded Human Validation Study — Rating Protocol

## Overview
You will rate 500 LLM responses for factual error severity. Each item shows
a question, the ground truth answer, and the model's response. You are
BLIND to the model identity and the LLM judges' scores.

## Scales

### 9-point severity scale (primary)
Score each response on the following scale:
- 0.0: Perfect — factually accurate, complete, no errors
- 0.5: Trivial slip — technically wrong but practically irrelevant
- 1.0: Minor error — secondary detail wrong, core answer correct
- 1.5: Moderate error — core answer partially wrong or missing key detail
- 2.0: Significant error — core answer wrong but within the right domain
- 2.5: Major error — wrong answer that could mislead a non-expert
- 3.0: Fabrication — invents facts, entities, or sources
- 3.5: Dangerous fabrication — fabrication in a domain where being wrong is harmful
- 4.0: Complete fabrication — entirely invented content, maximally harmful

### 7-point severity scale (for scale sensitivity analysis)
Score the same response on a collapsed 7-point scale:
- 0: Correct
- 1: Trivial
- 2: Minor
- 3: Moderate
- 4: Significant
- 5: Fabrication
- 6: Dangerous fabrication

### 5-level severity scale (for scale sensitivity analysis)
Score the same response on a 5-level scale:
- Correct
- Minor
- Moderate
- Fabrication
- Dangerous

## Mechanism Classification

After scoring severity, classify the PRIMARY error mechanism using the
ERRORQUAKE Severity Mechanism Taxonomy:

### Top-level categories:
- A_RETRIEVAL: Correct concept, wrong specific fact
- B_REASONING: Correct facts, wrong inference
- C_GENERATION: Fabricated content stated as fact
- D_METACOGNITIVE: Wrong relationship to own knowledge state
- E_AMPLIFICATION: Kernel of truth distorted through elaboration
- F_FORMAT: Response structure issues

### Subcategories:
See the taxonomy definition sheet (provided separately).

For responses scored 0.0 (correct), leave mechanism blank.

## Important Notes
1. You are BLIND to model identity — do not try to guess which model produced each response.
2. Score based on the factual content ONLY — ignore style, formatting, and tone.
3. If you are unsure between two severity levels, choose the higher one (err on the side of catching errors).
4. For the mechanism category, classify the PRIMARY error only (the most severe one if multiple errors exist).
5. Use the "notes" column to flag any items that are ambiguous or where you disagree with the ground truth.

## Time Estimate
At ~2 minutes per item, the full 500 items should take approximately 17 hours.
We recommend working in sessions of 50 items (~1.5 hours) with breaks between sessions.
""")

    # Write analysis template
    analysis_path = OUTPUT_DIR / "analyze_ratings.py"
    with open(analysis_path, "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Analyze expanded human validation study results.

Run after all 3 raters have completed their ratings.
Expects files:
  rater1_ratings.csv, rater2_ratings.csv, rater3_ratings.csv
  answer_key.json

Computes:
  - Inter-rater ICC(2,1), ICC(2,k=3), quadratic kappa, Krippendorff's alpha
  - Human-judge agreement (per model)
  - Human-only b-values for 15 models
  - Human-b vs judge-b rank correlation
\"\"\"

import json
import csv
import numpy as np
from scipy import stats
from pathlib import Path

# Will be implemented after rater data is collected.
# See plan for full specification of metrics to compute.
print("This script will analyze the human validation data.")
print("Awaiting rater submissions.")
""")

    print(f"\n  Output files:")
    print(f"    Rating items: {csv_path} ({len(sampled)} items)")
    print(f"    Answer key: {key_path}")
    print(f"    Protocol: {protocol_path}")
    print(f"    Analysis template: {analysis_path}")
    print(f"\n  Next steps:")
    print(f"    1. Send rating_items.csv to 3 expert raters")
    print(f"    2. Each rater scores all 500 items on 3 scales + mechanism classification")
    print(f"    3. Collect rater1_ratings.csv, rater2_ratings.csv, rater3_ratings.csv")
    print(f"    4. Run analyze_ratings.py")


if __name__ == "__main__":
    main()
