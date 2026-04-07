"""Experiment 7: Training pipeline effects (matched pairs).

Compare b-values between matched model pairs to isolate scale,
generation, and architecture effects. Bootstrap test for significance
(do the b-value distributions overlap on resampling?).

Output: results/analysis/exp7_training.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.analyze import _estimate_b, _quantize_to_grid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
ANALYSIS = PROJECT_ROOT / "results" / "analysis" / "full_21model_analysis.json"
OUT_PATH = PROJECT_ROOT / "results" / "analysis" / "exp7_training.json"

PAIRS = [
    # name, model_a, model_b, hypothesis
    ("Gemma-3 4B vs 12B",  "gemma-3-4b",  "gemma-3-12b",  "scale within Gemma-3"),
    ("Gemma-3 12B vs 27B", "gemma-3-12b", "gemma-3-27b",  "scale within Gemma-3"),
    ("Gemma-3 4B vs 27B",  "gemma-3-4b",  "gemma-3-27b",  "scale within Gemma-3"),
    ("Gemma-2 vs Gemma-3 27B", "gemma-2-27b", "gemma-3-27b", "generation upgrade"),
    ("Llama-3.2 3B vs 3.1 8B", "llama-3.2-3b-instruct", "llama-3.1-8b-instruct", "scale within Llama"),
    ("Qwen2.5 7B vs Llama 3.1 8B", "qwen2.5-7b", "llama-3.1-8b-instruct", "family at ~8B"),
    ("Gemma-3 12B vs Ministral 14B", "gemma-3-12b", "ministral-14b", "family at ~13B"),
    ("Llama-4 Maverick (MoE) vs Ministral 14B (dense)", "llama-4-maverick", "ministral-14b", "MoE vs dense at ~14-17B"),
    ("DeepSeek v3.1 vs v3.2", "deepseek-v3.1", "deepseek-v3.2", "minor version bump"),
    ("Mistral small 24B vs medium 3", "mistral-small-24b", "mistral-medium-3", "Mistral tier"),
    ("Mistral 24B vs Gemma-3 27B", "mistral-small-24b", "gemma-3-27b", "family at ~25B"),
]


def load_positive_scores(stem: str) -> np.ndarray:
    out = []
    path = SCORES_DIR / f"{stem}.jsonl"
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = r.get("final_score")
        if s is not None and s > 0:
            out.append(float(s))
    arr = np.asarray(out, dtype=float)
    return _quantize_to_grid(arr)


def bootstrap_b_diff(a: np.ndarray, b: np.ndarray, m_min: float, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(42)
    a_above = a[a >= m_min]
    b_above = b[b >= m_min]
    if a_above.size < 30 or b_above.size < 30:
        return {"error": "too_few_above_mmin",
                "n_a": int(a_above.size), "n_b": int(b_above.size)}
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a_above, size=a_above.size, replace=True)
        sb = rng.choice(b_above, size=b_above.size, replace=True)
        diffs.append(_estimate_b(sa, m_min) - _estimate_b(sb, m_min))
    diffs = np.array(diffs)
    p2 = float(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_two_sided": p2,
        "n_a_above_mmin": int(a_above.size),
        "n_b_above_mmin": int(b_above.size),
    }


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 7: Training pipeline pair comparisons")
    print("=" * 70)

    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    results = []

    print()
    print(f"{'Pair':<48} {'b_A':>7} {'b_B':>7} {'Δb':>8} {'95% CI':>20} {'p':>7}")
    print("-" * 100)

    for label, a_name, b_name, hyp in PAIRS:
        if a_name not in analysis or b_name not in analysis:
            print(f"{label:<48} -- missing")
            continue
        b_a = float(analysis[a_name]["b_value"]["b"])
        b_b = float(analysis[b_name]["b_value"]["b"])
        m_a = float(analysis[a_name]["b_value"]["m_min"])
        m_b = float(analysis[b_name]["b_value"]["m_min"])
        m_min = max(m_a, m_b)  # use the larger m_min for fair comparison

        a_scores = load_positive_scores(a_name)
        b_scores = load_positive_scores(b_name)
        boot = bootstrap_b_diff(a_scores, b_scores, m_min)

        if "error" in boot:
            print(f"{label:<48} {b_a:>7.3f} {b_b:>7.3f} {b_a-b_b:>+8.3f} {'--':>20} {'--':>7}")
            results.append({"label": label, "model_a": a_name, "model_b": b_name,
                            "hypothesis": hyp, "b_a": b_a, "b_b": b_b,
                            "shared_m_min": m_min, "bootstrap": boot})
            continue

        ci_str = f"[{boot['ci_lower']:+.3f}, {boot['ci_upper']:+.3f}]"
        print(f"{label:<48} {b_a:>7.3f} {b_b:>7.3f} "
              f"{boot['mean_diff']:>+8.3f} {ci_str:>20} {boot['p_two_sided']:>7.4f}")
        results.append({
            "label": label,
            "hypothesis": hyp,
            "model_a": a_name,
            "model_b": b_name,
            "b_a_full": b_a,
            "b_b_full": b_b,
            "shared_m_min": m_min,
            "bootstrap": boot,
            "significant": boot["p_two_sided"] < 0.05,
        })

    sig = sum(1 for r in results if r.get("significant"))
    print()
    print(f"Significant pairs (p<0.05): {sig}/{len(results)}")

    OUT_PATH.write_text(json.dumps({"pairs": results}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
