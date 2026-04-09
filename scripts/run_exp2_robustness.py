"""Exp 2 (b-discriminator) robustness under alternative scoring schemes.

For each aggregation rule (final_score, primary_only, secondary_only,
max, min) and for the dual-coverage subset: recompute per-model
b-values with bootstrap CIs, then count the pairs where

    |Δε| < 0.05  AND  |Δb| > 0.15  AND  95% b CIs are disjoint

This is now the headline claim under Option B. The count should be
stable across aggregation rules for Exp 2 to carry the paper.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value

SCORES = ROOT / "results" / "scores"
OUT = ROOT / "results" / "analysis" / "exp2_robustness.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_records(stem: str) -> list[dict]:
    return [json.loads(l) for l in open(SCORES / f"{stem}.jsonl",
            encoding="utf-8") if l.strip()]


def fit_b_with_ci(scores: list[float], name: str):
    arr = np.array([s for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return None
    pos = arr[arr > 0]
    eps = float((arr > 0).mean())
    if pos.size < 30:
        return {"b": None, "ci_lo": None, "ci_hi": None, "eps": eps}
    try:
        bv = estimate_b_value(pos, model_name=name)
        return {
            "b": float(bv.b),
            "ci_lo": float(bv.b_ci_lower),
            "ci_hi": float(bv.b_ci_upper),
            "eps": eps,
        }
    except Exception:
        return {"b": None, "ci_lo": None, "ci_hi": None, "eps": eps}


def count_pairs(per_model: dict[str, dict], label: str) -> dict:
    models = [m for m, v in per_model.items() if v.get("b") is not None]
    n_total = len(list(combinations(models, 2)))
    qualifying = 0
    disjoint = 0
    examples = []
    for a, b in combinations(models, 2):
        va, vb = per_model[a], per_model[b]
        eps_diff = abs(va["eps"] - vb["eps"])
        b_diff = abs(va["b"] - vb["b"])
        if eps_diff < 0.05 and b_diff > 0.15:
            qualifying += 1
            ci_disjoint = (max(va["ci_lo"], vb["ci_lo"])
                           > min(va["ci_hi"], vb["ci_hi"]))
            if ci_disjoint:
                disjoint += 1
                examples.append({
                    "model_1": a, "model_2": b,
                    "eps_1": va["eps"], "eps_2": vb["eps"],
                    "b_1": va["b"], "b_2": vb["b"], "b_diff": b_diff,
                })
    return {
        "label": label,
        "n_models_with_valid_b": len(models),
        "n_total_pairs": n_total,
        "n_qualifying": qualifying,
        "n_disjoint_CIs": disjoint,
        "top_pairs": sorted(examples, key=lambda e: -e["b_diff"])[:10],
    }


def aggregate(r: dict, scheme: str) -> float | None:
    p = r.get("primary_score")
    s = r.get("secondary_score")
    f = r.get("final_score")
    if scheme == "final_score":
        return f
    if scheme == "primary_only":
        return float(p) if p is not None else None
    if scheme == "secondary_only":
        return float(s) if s is not None else None
    if scheme == "max":
        if p is not None and s is not None:
            return max(float(p), float(s))
        return float(p) if p is not None else (float(s) if s is not None else None)
    if scheme == "min":
        if p is not None and s is not None:
            return min(float(p), float(s))
        return float(p) if p is not None else (float(s) if s is not None else None)
    return None


def main() -> None:
    print("=" * 70)
    print("EXP 2 DISCRIMINATOR ROBUSTNESS (Option B headline check)")
    print("=" * 70)

    all_models = [f.stem for f in SCORES.glob("*.jsonl") if f.stem not in EXCLUDED]
    print(f"Loading {len(all_models)} models...")
    model_recs = {m: load_records(m) for m in all_models}

    def fit_all(scheme: str, subset: list[str] | None = None):
        models = subset if subset is not None else all_models
        return {m: fit_b_with_ci([aggregate(r, scheme) for r in model_recs[m]],
                                 f"{m}_{scheme}")
                for m in models}

    results = {}
    print()
    for scheme in ("final_score", "primary_only", "secondary_only", "max", "min"):
        pm = fit_all(scheme)
        r = count_pairs(pm, f"all21_{scheme}")
        results[f"all21_{scheme}"] = r
        print(f"  {scheme:<16} n_models={r['n_models_with_valid_b']:>3}  "
              f"qualifying={r['n_qualifying']:>3}  "
              f"disjoint_CIs={r['n_disjoint_CIs']:>3}")

    # Dual-coverage subset (80%+ dual)
    dual = []
    for m in all_models:
        recs = model_recs[m]
        both = sum(1 for r in recs if r.get("primary_score") is not None
                   and r.get("secondary_score") is not None)
        if both / max(len(recs), 1) >= 0.80:
            dual.append(m)
    print(f"\nDual-coverage subset (>=80% both-judges): {len(dual)} models")

    for scheme in ("final_score", "primary_only", "secondary_only", "max", "min"):
        pm = fit_all(scheme, subset=dual)
        r = count_pairs(pm, f"dual_{scheme}")
        results[f"dual_{scheme}"] = r
        print(f"  {scheme:<16} n_models={r['n_models_with_valid_b']:>3}  "
              f"qualifying={r['n_qualifying']:>3}  "
              f"disjoint_CIs={r['n_disjoint_CIs']:>3}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY: disjoint-CI pair counts across robustness checks")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:<30} -> {v['n_disjoint_CIs']:>3} pairs")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
