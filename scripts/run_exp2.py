"""Experiment 2: b-value as model discriminator.

For all C(21,2) = 210 model pairs, find pairs where:
  - error rate gap < 0.05
  - b-value gap > 0.15 AND CIs do not overlap

Pre-registered criterion: at least 3 such pairs => b-value adds
information beyond the error rate.

Output: results/analysis/exp2_discriminator.json
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "results" / "analysis" / "full_21model_analysis.json"
OUT = ROOT / "results" / "analysis" / "exp2_discriminator.json"


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 2: b-value as model discriminator")
    print("=" * 70)

    data = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    models = []
    for name, d in data.items():
        bv = d.get("b_value", {})
        models.append({
            "name": name,
            "b": float(bv.get("b", float("nan"))),
            "ci_lo": float(bv.get("ci_lower", float("nan"))),
            "ci_hi": float(bv.get("ci_upper", float("nan"))),
            "err": float(d.get("error_rate", float("nan"))),
        })
    print(f"Loaded {len(models)} models")

    pairs = []
    for m1, m2 in combinations(models, 2):
        err_diff = abs(m1["err"] - m2["err"])
        b_diff = abs(m1["b"] - m2["b"])
        # CIs don't overlap if max(lo) > min(hi)
        ci_gap = max(m1["ci_lo"], m2["ci_lo"]) > min(m1["ci_hi"], m2["ci_hi"])
        if err_diff < 0.05 and b_diff > 0.15:
            pairs.append({
                "model_1": m1["name"],
                "model_2": m2["name"],
                "err_1": m1["err"],
                "err_2": m2["err"],
                "err_diff": err_diff,
                "b_1": m1["b"],
                "b_2": m2["b"],
                "b_diff": b_diff,
                "ci_1": [m1["ci_lo"], m1["ci_hi"]],
                "ci_2": [m2["ci_lo"], m2["ci_hi"]],
                "ci_disjoint": ci_gap,
            })

    pairs.sort(key=lambda p: -p["b_diff"])
    print(f"\nFound {len(pairs)} pairs with err_diff<0.05 AND b_diff>0.15")
    print(f"Of those, {sum(1 for p in pairs if p['ci_disjoint'])} have disjoint CIs")
    print()

    print(f"{'Model 1':<24} {'Model 2':<24} {'Δerr':>7} {'Δb':>7} {'CIs disjoint':>13}")
    print("-" * 80)
    for p in pairs[:30]:
        print(f"{p['model_1']:<24} {p['model_2']:<24} "
              f"{p['err_diff']:>7.3f} {p['b_diff']:>7.3f} "
              f"{('YES' if p['ci_disjoint'] else 'no'):>13}")

    n_strong = sum(1 for p in pairs if p["ci_disjoint"])

    summary = {
        "n_models": len(models),
        "n_pairs_total": len(list(combinations(models, 2))),
        "n_pairs_qualifying": len(pairs),
        "n_pairs_disjoint_CIs": n_strong,
        "criterion_passed": n_strong >= 3,
        "criterion_threshold": "b-value pairs with disjoint CIs at err diff < 5%",
    }

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total pairs: {summary['n_pairs_total']}")
    print(f"Qualifying (err<0.05, b>0.15): {summary['n_pairs_qualifying']}")
    print(f"Of those with disjoint CIs:    {summary['n_pairs_disjoint_CIs']}")
    print(f"Pre-registered criterion (≥3): {'PASS' if summary['criterion_passed'] else 'FAIL'}")

    OUT.write_text(json.dumps({"summary": summary, "pairs": pairs}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
