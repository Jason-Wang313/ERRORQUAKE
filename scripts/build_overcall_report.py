"""Build the overcall diagnostic report from manual classifications.

The classifications below were produced by reading each item's question,
ground truth, and response and comparing them locally (no API calls).
"""

import json
from pathlib import Path
from collections import Counter

# Classification key:
# G = GENUINE_ERROR (response factually wrong in a meaningful way)
# O = OVERCALL (response correct or trivially imprecise; judge too harsh)
# A = AMBIGUOUS (hard to tell without domain expertise)

CLASSIFICATIONS = {
    "eurollm-9b": "G A A O G G O G G O O G G O A G O A A G".split(),
    "gemma-2-27b": "G O O G G O G O G G A G A O O G G G G O".split(),
    "gemma-3-12b": "G O G G A O G G G G G G A G G G G O A G".split(),
    "gemma-3-27b": "A O G G O G O O G G G G A O A O G O G O".split(),
    "gemma-3-4b":  "O A G G G G G G G G O G G O A G O O G G".split(),
    "gpt-oss-20b": "O A G A O G G A G G G O G G G G G G G O".split(),
    "llama-3.1-8b-instruct": "G A G O A O G G O G G O O G O O O O G O".split(),
    "llama-3.2-3b-instruct": "O G A G G G G O O O A G O O G O G A G O".split(),
    "ministral-14b": "O A O O O G O A A O O O O G O A G G O G".split(),
    "mistral-small-24b": "G O A A O G O O O O O A G O O A G G O O".split(),
    "mistral-small-4-119b": "O O G O O O A O A A G O G O G A O O O O".split(),
    "phi-3.5-mini": "G G G G G O O G G G G A G A O G G O G A".split(),
    "phi-4-mini-flash-reasoning": "G G G O O A G G G G G G G A G G G G G G".split(),
    "qwen2.5-7b":  "O A G G G O G A G G A G A O G A A A G A".split(),
    "qwen3-next-80b": "A G A G G A O O O O O A O G G A A O A O".split(),
    "seed-oss-36b": "O O G G G G O O A O G G O G A O G G G G".split(),
    "solar-10.7b": "O G O O A A G A G G A A O A G G G G G O".split(),
}

# Verify each list has 20
for m, c in CLASSIFICATIONS.items():
    assert len(c) == 20, f"{m} has {len(c)} not 20"

# 5 clearest overcall examples (cherry-picked from classifications above)
# Each is (model_name, item_index_1based, query_id, brief_explanation)
CLEAREST_OVERCALLS = [
    {
        "model": "seed-oss-36b",
        "item_index": 1,
        "query_id": "CULT_T5_0035",
        "question": "The composer Richard Wagner's opera 'Der Ring des Nibelungen' consists of how many operas?",
        "ground_truth": "Richard Wagner's 'Der Ring des Nibelungen' is a tetralogy, consisting of four operas: Das Rheingold, Die Walkure, Siegfried, and Gotterdammerung.",
        "model_response": "4",
        "explanation": "Response is exactly correct. Both judges scored 2.0 for what is literally the right number.",
    },
    {
        "model": "seed-oss-36b",
        "item_index": 2,
        "query_id": "TECH_T4_0130",
        "question": "Which machine learning technique is used to prevent overfitting by randomly dropping out units during training?",
        "ground_truth": "Dropout is a regularization technique used in neural networks to prevent overfitting by randomly dropping out units during training.",
        "model_response": "Dropout",
        "explanation": "Response is the exact right one-word answer. Judge penalized brevity.",
    },
    {
        "model": "mistral-small-4-119b",
        "item_index": 5,
        "query_id": "GEO_T1_0220",
        "question": "What is the term for the process of adjusting a map to fit the curvature of the Earth?",
        "ground_truth": "The process is related to map projection, as different projections adjust for the Earth's curvature in various ways.",
        "model_response": "The term for the process of adjusting a map to fit the curvature of the Earth is 'projection.'",
        "explanation": "Exact term match ('projection'). Judge scored 2.0 despite perfect alignment.",
    },
    {
        "model": "ministral-14b",
        "item_index": 1,
        "query_id": "HIST_T2_0112",
        "question": "The Battle of Plassey was fought in which year and marked a significant victory for the British East India Company?",
        "ground_truth": "The Battle of Plassey was fought in 1757 and marked a significant victory for the British East India Company.",
        "model_response": "The Battle of Plassey was fought on 23 June 1757 and marked a decisive victory for the British East India Company over the Nawab of Bengal, Siraj-ud-Daulah.",
        "explanation": "Response is exactly correct AND adds correct supplementary detail (date, opponent). Judge penalized the extra (correct) information.",
    },
    {
        "model": "qwen3-next-80b",
        "item_index": 11,
        "query_id": "BIO_T5_0181",
        "question": "What was the median overall survival benefit reported for the combination of encorafenib and binimetinib versus vemurafenib in the COLUMBUS trial for BRAF V600E melanoma?",
        "ground_truth": "The COLUMBUS trial reported a median overall survival of 33.6 months (95% CI 24.4-39.2) for encorafenib plus binimetinib versus 16.9 months (95% CI 10.9-24.4) for vemurafenib, representing a significant survival benefit.",
        "model_response": "The median overall survival benefit reported for the combination of encorafenib and binimetinib versus vemurafenib in the COLUMBUS trial for BRAF V600E melanoma was 33.6 months versus 16.9 months, representing a 16.7-month improvement.",
        "explanation": "Response gives the exact same numbers as the ground truth (33.6 vs 16.9 months) and correctly computes the difference. Judge scored 2.0 despite perfect numerical match.",
    },
]


def main():
    # Per-model stats
    per_model = {}
    total_g = total_o = total_a = 0
    for model, classes in CLASSIFICATIONS.items():
        c = Counter(classes)
        n = len(classes)
        per_model[model] = {
            "n_sampled": n,
            "n_genuine": c["G"],
            "n_overcall": c["O"],
            "n_ambiguous": c["A"],
            "overcall_rate": round(c["O"] / n, 3),
            "genuine_rate": round(c["G"] / n, 3),
            "ambiguous_rate": round(c["A"] / n, 3),
        }
        total_g += c["G"]
        total_o += c["O"]
        total_a += c["A"]

    total = total_g + total_o + total_a

    overall = {
        "n_models": len(CLASSIFICATIONS),
        "n_total_sampled": total,
        "n_total_genuine": total_g,
        "n_total_overcall": total_o,
        "n_total_ambiguous": total_a,
        "overall_overcall_rate": round(total_o / total, 3),
        "overall_genuine_rate": round(total_g / total, 3),
        "overall_ambiguous_rate": round(total_a / total, 3),
    }

    report = {
        "score_range": "1.75 to 2.25 (rounds to 2.0)",
        "classification_method": "Manual reading of question/ground_truth/response by agent (no API calls)",
        "overall": overall,
        "per_model": per_model,
        "clearest_overcalls_for_paper": CLEAREST_OVERCALLS,
    }

    out = Path("results/analysis/overcall_diagnostic.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    print("=" * 70)
    print("OVERCALL DIAGNOSTIC AT SCORE 2.0")
    print("=" * 70)
    print(f"Sampled {total} items across {len(CLASSIFICATIONS)} models "
          f"(20 per model, score in [1.75, 2.25])")
    print()
    print(f"OVERALL:")
    print(f"  Genuine errors:    {total_g:3d} ({total_g*100/total:.1f}%)")
    print(f"  Overcalls:         {total_o:3d} ({total_o*100/total:.1f}%)")
    print(f"  Ambiguous:         {total_a:3d} ({total_a*100/total:.1f}%)")
    print()
    print("PER-MODEL OVERCALL RATES (sorted high to low):")
    sorted_models = sorted(per_model.items(), key=lambda x: -x[1]["overcall_rate"])
    print(f"  {'Model':35s} {'n':>4} {'gen':>5} {'over':>5} {'amb':>5} {'over%':>7}")
    for m, s in sorted_models:
        print(f"  {m:35s} {s['n_sampled']:4d} {s['n_genuine']:5d} "
              f"{s['n_overcall']:5d} {s['n_ambiguous']:5d} "
              f"{s['overcall_rate']*100:6.1f}%")
    print()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
