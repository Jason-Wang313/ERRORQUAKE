"""Experiment 3 diagnostic: easy-hard b-value divergence.

For each of 21 models, fit b separately on tier 1-2 errors and tier 4-5
errors. If b_easy > b_hard consistently, the easy tail is steeper than
the hard tail, which means GR extrapolation from easy underestimates the
catastrophic count -> systematic magnitude bias in Experiment 3.

Output: results/analysis/exp3_diagnostic.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.analyze import estimate_b_value

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
OUT_PATH = PROJECT_ROOT / "results" / "analysis" / "exp3_diagnostic.json"

EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_split(path: Path):
    easy, hard = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        score = r.get("final_score")
        tier = r.get("tier")
        if score is None:
            continue
        if tier in (1, 2):
            easy.append(float(score))
        elif tier in (4, 5):
            hard.append(float(score))
    return np.asarray(easy, dtype=float), np.asarray(hard, dtype=float)


def fit_b(scores: np.ndarray, name: str, label: str) -> dict:
    pos = scores[scores > 0]
    if pos.size < 30:
        return {"label": label, "n_errors": int(pos.size), "error": "too_few"}
    try:
        bv = estimate_b_value(pos, model_name=f"{name}_{label}")
        return {
            "label": label,
            "n_errors": int(pos.size),
            "b": float(bv.b),
            "ci_lower": float(bv.b_ci_lower),
            "ci_upper": float(bv.b_ci_upper),
            "m_min": float(bv.m_min),
            "n_above_mmin": int(bv.n_above_mmin),
        }
    except Exception as exc:
        return {"label": label, "n_errors": int(pos.size), "error": str(exc)[:120]}


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 3 DIAGNOSTIC: Easy vs Hard b-value divergence")
    print("=" * 70)

    files = sorted(f for f in SCORES_DIR.glob("*.jsonl") if f.stem not in EXCLUDED)

    results = {}
    print(f"\n{'Model':<28} {'b_easy':>8} {'b_hard':>8} {'Δ b':>8} {'m_easy':>7} {'m_hard':>7}")
    print("-" * 70)
    for f in files:
        easy, hard = load_split(f)
        easy_fit = fit_b(easy, f.stem, "easy")
        hard_fit = fit_b(hard, f.stem, "hard")
        results[f.stem] = {"easy": easy_fit, "hard": hard_fit}

        b_e = easy_fit.get("b")
        b_h = hard_fit.get("b")
        m_e = easy_fit.get("m_min")
        m_h = hard_fit.get("m_min")
        if b_e is not None and b_h is not None:
            delta = b_e - b_h
            print(f"{f.stem:<28} {b_e:>8.3f} {b_h:>8.3f} {delta:>+8.3f} {m_e:>7.2f} {m_h:>7.2f}")
        else:
            print(f"{f.stem:<28} {'--':>8} {'--':>8} {'--':>8} {'--':>7} {'--':>7}")

    # Aggregate stats
    pairs = [
        (v["easy"]["b"], v["hard"]["b"])
        for v in results.values()
        if "b" in v["easy"] and "b" in v["hard"]
    ]
    deltas = [e - h for e, h in pairs]
    n_easy_steeper = sum(1 for d in deltas if d > 0)

    summary = {
        "n_models": len(pairs),
        "mean_b_easy": float(np.mean([p[0] for p in pairs])),
        "mean_b_hard": float(np.mean([p[1] for p in pairs])),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "std_delta": float(np.std(deltas)),
        "min_delta": float(min(deltas)),
        "max_delta": float(max(deltas)),
        "n_easy_steeper": n_easy_steeper,
        "n_hard_steeper": len(pairs) - n_easy_steeper,
    }

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  n models with both fits: {summary['n_models']}")
    print(f"  mean b_easy = {summary['mean_b_easy']:.3f}")
    print(f"  mean b_hard = {summary['mean_b_hard']:.3f}")
    print(f"  mean Δb (easy-hard) = {summary['mean_delta']:+.3f}")
    print(f"  median Δb = {summary['median_delta']:+.3f}")
    print(f"  std Δb   = {summary['std_delta']:.3f}")
    print(f"  range    = [{summary['min_delta']:+.3f}, {summary['max_delta']:+.3f}]")
    print(f"  easy steeper (b_easy > b_hard): {summary['n_easy_steeper']}/{summary['n_models']}")
    print()
    if summary["mean_delta"] > 0.1:
        print("  → Easy tails are systematically STEEPER than hard tails.")
        print("    GR extrapolation from easy UNDER-predicts catastrophes (b too high).")
    elif summary["mean_delta"] < -0.1:
        print("  → Easy tails are systematically FLATTER than hard tails.")
        print("    GR extrapolation from easy OVER-predicts catastrophes (b too low).")
    else:
        print("  → No systematic divergence; magnitude bias has another cause.")

    out = {"summary": summary, "models": results}
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
