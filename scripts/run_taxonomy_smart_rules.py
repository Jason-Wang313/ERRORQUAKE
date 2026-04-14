"""
Smart Rule-Based Taxonomy Classifier
======================================
Classifies all 1,000 items locally using content analysis.
No API calls needed. Examines response content vs ground truth
to determine error mechanism.

Strategy:
1. Check for fabricated entities (proper nouns in response not in GT)
2. Check for numerical errors (numbers differ from GT)
3. Check for denial/deflection patterns
4. Check for partial overlap (starts right, goes wrong)
5. Use severity score as a prior (high severity = more likely fabrication)
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as scipy_stats

REPO = Path(__file__).resolve().parent.parent
TAXONOMY_DIR = REPO / "results" / "analysis" / "oral_upgrade" / "taxonomy"
INPUT_PATH = TAXONOMY_DIR / "taxonomy_items_prepared.jsonl"
OUTPUT_PATH = TAXONOMY_DIR / "taxonomy_items_classified.jsonl"
ANALYSIS_DIR = REPO / "results" / "analysis"

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

MODEL_PARAMS = {
    "llama-3.2-3b-instruct": 3.21, "phi-3.5-mini": 3.82, "gemma-3-4b": 4.3,
    "qwen2.5-7b": 7.62, "llama-3.1-8b-instruct": 8.03, "eurollm-9b": 9.16,
    "solar-10.7b": 10.7, "gemma-3-12b": 12.2, "ministral-14b": 14.0,
    "gpt-oss-20b": 20.0, "mistral-small-24b": 24.0,
    "gemma-2-27b": 27.0, "gemma-3-27b": 27.2, "seed-oss-36b": 36.0,
    "mistral-small-4-119b": 22.0, "mistral-medium-3": 37.0,
    "kimi-k2-instruct": 32.0, "llama-4-maverick": 17.0,
    "deepseek-v3.1": 37.0, "deepseek-v3.2": 37.0, "qwen3-next-80b": 80.0,
}


def extract_proper_nouns(text):
    """Extract capitalized multi-word sequences (rough NER)."""
    if not text:
        return set()
    nouns = set()
    for match in re.finditer(r'[A-Z][a-z]+(?:\s+(?:of|the|and|in|for|de|van|von|v\.)?\s*[A-Z][a-z]+)*', text):
        noun = match.group().strip()
        if len(noun) > 3 and noun not in ('The', 'This', 'That', 'These', 'However', 'Based', 'According'):
            nouns.add(noun.lower())
    return nouns


def extract_numbers(text):
    """Extract numbers from text."""
    if not text:
        return []
    return [float(x) for x in re.findall(r'(?<!\w)(\d+(?:,\d{3})*(?:\.\d+)?)(?!\w)', text.replace(',', ''))]


def word_overlap(text1, text2):
    """Compute content word overlap between two texts."""
    stop = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'of', 'in', 'to', 'and',
            'for', 'it', 'that', 'this', 'with', 'on', 'at', 'by', 'from', 'as', 'or',
            'be', 'not', 'no', 'but', 'its', 'has', 'had', 'have', 'been', 'can', 'will',
            'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might', 'shall',
            'also', 'than', 'then', 'just', 'more', 'most', 'such', 'other'}
    w1 = set(text1.lower().split()) - stop
    w2 = set(text2.lower().split()) - stop
    if not w1:
        return 0.0
    return len(w1 & w2) / len(w1)


def classify_item(item):
    """
    Smart rule-based classification using content analysis.
    Returns (category, subcategory, confidence, explanation).
    """
    sev = item["severity_score"]
    q = item.get("question", "") or ""
    gt = item.get("ground_truth", "") or ""
    resp = item.get("response", "") or ""

    # Trivial cases
    if sev <= 0.25:
        return "CORRECT", "CORRECT", "high", "Score indicates correct response"

    if not resp or len(resp.strip()) < 20:
        return "F_FORMAT", "F1_incomplete_response", "high", "Response too short or empty"

    # === Content analysis ===
    gt_nouns = extract_proper_nouns(gt)
    resp_nouns = extract_proper_nouns(resp)
    gt_nums = extract_numbers(gt)
    resp_nums = extract_numbers(resp)
    overlap = word_overlap(gt, resp)

    # Fabricated entities: proper nouns in response NOT in ground truth
    fabricated_nouns = resp_nouns - gt_nouns
    # Remove common false positives
    fabricated_nouns = {n for n in fabricated_nouns
                       if not any(w in n for w in ['the', 'united states', 'world war'])}

    # Number comparison
    has_number_error = False
    number_ratio = None
    if gt_nums and resp_nums:
        primary_gt = gt_nums[0]
        if primary_gt != 0:
            best = min(resp_nums, key=lambda rn: abs(rn - primary_gt) / max(abs(primary_gt), 1e-10))
            number_ratio = abs(best - primary_gt) / abs(primary_gt)
            has_number_error = number_ratio > 0.15

    # Denial patterns
    denial_patterns = [
        r"(?:there is no|does not exist|no (?:historical )?record|not a subject|"
        r"is not (?:typically|commonly)|did not (?:involve|have)|no such|"
        r"i(?:'m| am) not aware|cannot (?:find|confirm)|no (?:specific|exact))"
    ]
    has_denial = any(re.search(p, resp.lower()) for p in denial_patterns)

    # Confidence markers in response
    confidence_markers = [
        r'\*\*\d', r'approximately \d', r'exactly \d', r'precisely \d',
        r'the answer is', r'specifically', r'in fact',
    ]
    has_confident_assertion = sum(1 for p in confidence_markers if re.search(p, resp.lower()))

    # Citation/source fabrication markers
    citation_markers = [
        r'(?:according to|as reported by|based on|cited by|published in|the study by)',
        r'(?:\(\d{4}\)|\[\d+\])',  # year citations or numbered refs
    ]
    has_citation = any(re.search(p, resp.lower()) for p in citation_markers)
    gt_has_citation = any(re.search(p, gt.lower()) for p in citation_markers)

    # === Classification logic ===

    # F_FORMAT: irrelevant or incomplete
    if overlap < 0.05 and sev >= 2.0 and len(resp) > 50:
        return "F_FORMAT", "F2_irrelevant_response", "medium", "Very low content overlap with ground truth"

    # D_METACOGNITIVE: denial/deflection
    if has_denial and sev >= 2.0:
        return "D_METACOGNITIVE", "D1_denial_deflection", "high", f"Denial pattern detected in response"

    # C_GENERATION: fabrication (high severity + fabricated content)
    if sev >= 3.0:
        if has_citation and not gt_has_citation:
            return "C_GENERATION", "C2_citation_fabrication", "high", "Cites sources not in ground truth at high severity"
        if len(fabricated_nouns) >= 2:
            return "C_GENERATION", "C1_entity_fabrication", "high", f"Multiple fabricated entities: {list(fabricated_nouns)[:3]}"
        if has_number_error and number_ratio and number_ratio > 5.0:
            return "C_GENERATION", "C4_false_precision", "high", f"Number off by {number_ratio:.0f}x with confident assertion"
        if has_confident_assertion >= 2 and overlap < 0.3:
            return "C_GENERATION", "C3_detail_confabulation", "medium", "Confident assertion with low ground truth overlap"
        # Default high severity = confabulation
        return "C_GENERATION", "C3_detail_confabulation", "medium", "High severity with divergent content"

    # C_GENERATION at moderate-high severity
    if sev >= 2.5:
        if len(fabricated_nouns) >= 2:
            return "C_GENERATION", "C1_entity_fabrication", "medium", f"Fabricated entities at severity {sev}"
        if has_citation and not gt_has_citation:
            return "C_GENERATION", "C2_citation_fabrication", "medium", "Fabricated citation"
        if has_number_error and number_ratio and number_ratio > 2.0:
            return "C_GENERATION", "C4_false_precision", "medium", f"Number off by {number_ratio:.1f}x"

    # A_RETRIEVAL: numerical distortion
    if has_number_error and number_ratio:
        if number_ratio <= 1.0:
            return "A_RETRIEVAL", "A4_numerical_distortion", "high", f"Number off by {number_ratio*100:.0f}%"
        elif number_ratio <= 5.0:
            return "A_RETRIEVAL", "A4_numerical_distortion", "medium", f"Number off by {number_ratio:.1f}x"

    # A_RETRIEVAL: entity substitution
    if fabricated_nouns and gt_nouns and overlap >= 0.2:
        return "A_RETRIEVAL", "A1_entity_substitution", "medium", f"Different entities but topic overlap exists"

    # A_RETRIEVAL: temporal
    gt_years = set(re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', gt))
    resp_years = set(re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', resp))
    if gt_years and resp_years and gt_years != resp_years and overlap >= 0.3:
        return "A_RETRIEVAL", "A2_temporal_misattribution", "medium", f"Different years: GT={gt_years}, Resp={resp_years}"

    # E_AMPLIFICATION: partial truth inflated
    if 1.0 <= sev <= 2.5 and overlap >= 0.3:
        return "E_AMPLIFICATION", "E1_partial_truth_inflated", "medium", "Moderate severity with partial content overlap"

    # B_REASONING: logical error
    logic_markers = ['therefore', 'because', 'since', 'thus', 'hence', 'consequently', 'implies']
    if any(m in resp.lower() for m in logic_markers) and 1.5 <= sev <= 2.5 and overlap >= 0.2:
        return "B_REASONING", "B3_logical_error", "low", "Contains reasoning connectors with moderate error"

    # B_REASONING: scope overgeneralization
    if overlap >= 0.4 and sev <= 2.0:
        general_markers = ['all', 'every', 'always', 'never', 'any', 'most']
        if any(m in resp.lower().split() for m in general_markers):
            return "B_REASONING", "B2_scope_overgeneralization", "low", "Overgeneralization markers present"

    # Default by severity
    if sev >= 2.5:
        return "C_GENERATION", "C3_detail_confabulation", "low", f"High severity default (sev={sev})"
    if sev >= 1.5:
        return "E_AMPLIFICATION", "E1_partial_truth_inflated", "low", f"Mid severity default (sev={sev})"
    return "A_RETRIEVAL", "A1_entity_substitution", "low", f"Low severity default (sev={sev})"


def main():
    print("=" * 70)
    print("SMART RULE-BASED TAXONOMY CLASSIFICATION")
    print("=" * 70)

    # Load items
    items = []
    with open(INPUT_PATH) as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items)} items")

    # Classify all
    classified = []
    for item in items:
        cat, sub, conf, expl = classify_item(item)
        item["primary_category"] = cat
        item["primary_subcategory"] = sub
        item["classification_confidence"] = conf
        item["classification_explanation"] = expl
        item["classifier_model"] = "smart_rules_v1"
        classified.append(item)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for it in classified:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # === ANALYSIS ===
    print(f"\n--- Category Distribution ---")
    cats = Counter(it["primary_category"] for it in classified if it["primary_category"] != "CORRECT")
    non_correct = [it for it in classified if it["primary_category"] != "CORRECT"]
    total = len(non_correct)
    for c, n in cats.most_common():
        print(f"  {c}: {n} ({n/total*100:.1f}%)")

    print(f"\n--- Subcategory Distribution ---")
    subs = Counter(it["primary_subcategory"] for it in non_correct)
    for s, n in subs.most_common(12):
        print(f"  {s}: {n} ({n/total*100:.1f}%)")

    # === KEY ANALYSIS: Mechanism profile by model size ===
    print(f"\n--- Mechanism Profile by Model Size ---")
    size_mech = defaultdict(lambda: defaultdict(int))
    for it in non_correct:
        size = MODEL_SIZE.get(it["model"], "unknown")
        size_mech[size][it["primary_category"]] += 1

    all_cats = sorted(cats.keys())
    print(f"{'Size':>8s}", end="")
    for c in all_cats:
        print(f"  {c[:8]:>8s}", end="")
    print(f"  {'total':>6s}")

    for size in ["small", "medium", "large"]:
        total_s = sum(size_mech[size].values())
        print(f"{size:>8s}", end="")
        for c in all_cats:
            n = size_mech[size][c]
            pct = n / total_s * 100 if total_s > 0 else 0
            print(f"  {pct:7.1f}%", end="")
        print(f"  {total_s:6d}")

    # Chi-squared test: mechanism profile differs by size
    if len(size_mech) >= 2:
        # Build contingency table
        sizes = ["small", "medium", "large"]
        table = []
        for size in sizes:
            row = [size_mech[size].get(c, 0) for c in all_cats]
            if sum(row) > 0:
                table.append(row)

        if len(table) >= 2:
            table_arr = np.array(table)
            chi2, p, dof, expected = scipy_stats.chi2_contingency(table_arr)
            print(f"\n  Chi-squared test (mechanism × size): chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
            print(f"  {'SIGNIFICANT (p < 0.05): mechanism profiles differ by model size!' if p < 0.05 else 'Not significant'}")

    # === KEY ANALYSIS: Fabrication rate vs b-value ===
    print(f"\n--- Fabrication Rate vs b-value ---")
    with open(ANALYSIS_DIR / "full_21model_analysis.json") as f:
        full_analysis = json.load(f)

    model_fab_rate = {}
    for model in set(it["model"] for it in non_correct):
        model_items = [it for it in non_correct if it["model"] == model]
        n_total = len(model_items)
        n_fab = sum(1 for it in model_items if it["primary_category"] == "C_GENERATION")
        if n_total >= 10:
            model_fab_rate[model] = n_fab / n_total

    # Correlate with b-value
    paired_models = []
    fab_rates = []
    b_values = []
    for model, fab_rate in model_fab_rate.items():
        if model in full_analysis:
            paired_models.append(model)
            fab_rates.append(fab_rate)
            b_values.append(full_analysis[model]["b_value"]["b"])

    if len(paired_models) >= 5:
        fab_arr = np.array(fab_rates)
        b_arr = np.array(b_values)
        rho, p = scipy_stats.spearmanr(fab_arr, 1.0 / b_arr)  # fab rate vs 1/b (heavier tail)
        rho_direct, p_direct = scipy_stats.spearmanr(fab_arr, b_arr)

        print(f"  n_models: {len(paired_models)}")
        print(f"  rho(fabrication_rate, 1/b): {rho:.4f} (p={p:.4f})")
        print(f"  rho(fabrication_rate, b):   {rho_direct:.4f} (p={p_direct:.4f})")
        print(f"  {'IDEAL: fabrication correlates with heavier tails!' if rho > 0.3 and p < 0.1 else 'Correlation not strong enough'}")

        for m, fr, bv in sorted(zip(paired_models, fab_rates, b_values), key=lambda x: x[2]):
            size = MODEL_SIZE.get(m, "?")
            print(f"    {m:30s} b={bv:.3f} fab={fr:.3f} size={size}")

    # === KEY ANALYSIS: Mechanism by severity band ===
    print(f"\n--- Mechanism by Severity Band ---")
    band_mech = defaultdict(lambda: defaultdict(int))
    for it in non_correct:
        s = it["severity_score"]
        if s <= 1.0: band = "low (0.5-1.0)"
        elif s <= 2.0: band = "mid (1.0-2.0)"
        else: band = "high (2.0-4.0)"
        band_mech[band][it["primary_category"]] += 1

    for band in ["low (0.5-1.0)", "mid (1.0-2.0)", "high (2.0-4.0)"]:
        total_b = sum(band_mech[band].values())
        print(f"  {band}: ", end="")
        for c in all_cats:
            n = band_mech[band][c]
            pct = n / total_b * 100 if total_b > 0 else 0
            print(f"{c[:6]}={pct:.0f}% ", end="")
        print()

    # Save full analysis
    analysis_output = {
        "category_distribution": dict(cats),
        "subcategory_distribution": dict(subs),
        "mechanism_by_size": {s: dict(d) for s, d in size_mech.items()},
        "mechanism_by_severity": {b: dict(d) for b, d in band_mech.items()},
        "fabrication_vs_b": {
            "rho_fab_inv_b": round(float(rho), 4) if 'rho' in dir() else None,
            "p_fab_inv_b": round(float(p), 4) if 'p' in dir() else None,
            "per_model": {m: {"fab_rate": round(fr, 4), "b": round(bv, 4), "size": MODEL_SIZE.get(m)}
                          for m, fr, bv in zip(paired_models, fab_rates, b_values)}
        } if paired_models else {},
    }

    out_path = TAXONOMY_DIR / "taxonomy_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis_output, f, indent=2, default=float)

    print(f"\n  Analysis saved to {out_path}")
    print(f"  Classification saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
