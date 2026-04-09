"""Task 2.3: cross-domain jackknife.

For each of 8 domains: remove its 500 queries from all models, refit b,
and recount the Exp 2 discriminator pairs. Under Option B the question
is: does the headline 30-pair count survive removing any single domain.
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
OUT = ROOT / "results" / "analysis" / "cross_domain_jackknife.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_records(stem: str) -> list[dict]:
    return [json.loads(l) for l in open(SCORES / f"{stem}.jsonl",
            encoding="utf-8") if l.strip()]


def fit_b_ci_eps(records: list[dict]):
    scores = [r.get("final_score") for r in records if r.get("final_score") is not None]
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return None
    pos = arr[arr > 0]
    eps = float((arr > 0).mean())
    if pos.size < 30:
        return {"b": None, "ci_lo": None, "ci_hi": None, "eps": eps}
    try:
        bv = estimate_b_value(pos, model_name="jk")
        return {"b": float(bv.b), "ci_lo": float(bv.b_ci_lower),
                "ci_hi": float(bv.b_ci_upper), "eps": eps}
    except Exception:
        return {"b": None, "ci_lo": None, "ci_hi": None, "eps": eps}


def count_disjoint(per_model: dict[str, dict]) -> int:
    models = [m for m, v in per_model.items() if v.get("b") is not None]
    count = 0
    for a, b in combinations(models, 2):
        va, vb = per_model[a], per_model[b]
        if abs(va["eps"] - vb["eps"]) >= 0.05:
            continue
        if abs(va["b"] - vb["b"]) <= 0.15:
            continue
        if max(va["ci_lo"], vb["ci_lo"]) > min(va["ci_hi"], vb["ci_hi"]):
            count += 1
    return count


def main() -> None:
    print("=" * 70)
    print("CROSS-DOMAIN JACKKNIFE (Task 2.3)")
    print("=" * 70)

    domains = ["BIO", "CULT", "FIN", "GEO", "HIST", "LAW", "SCI", "TECH"]
    models = [f.stem for f in SCORES.glob("*.jsonl") if f.stem not in EXCLUDED]
    recs_by_model = {m: load_records(m) for m in models}
    print(f"Loaded {len(models)} models")

    # Baseline (no drop)
    baseline_pm = {m: fit_b_ci_eps(recs_by_model[m]) for m in models}
    baseline_pairs = count_disjoint(baseline_pm)
    print(f"\nBaseline (all 8 domains): {baseline_pairs} disjoint-CI pairs")

    # Jackknife per domain
    print()
    print(f"{'drop domain':<14} {'n_pairs':>10} {'rel':>7}")
    print("-" * 35)
    rows = []
    for dom in domains:
        pm = {}
        for m in models:
            filtered = [r for r in recs_by_model[m] if r.get("domain") != dom]
            pm[m] = fit_b_ci_eps(filtered)
        n_pairs = count_disjoint(pm)
        rel = n_pairs / max(baseline_pairs, 1)
        rows.append({"dropped_domain": dom, "n_pairs": n_pairs, "rel_to_baseline": rel})
        print(f"  {dom:<12} {n_pairs:>10} {rel:>7.2f}")

    min_pairs = min(r["n_pairs"] for r in rows)
    max_pairs = max(r["n_pairs"] for r in rows)
    print(f"\nJackknife range: [{min_pairs}, {max_pairs}] vs baseline {baseline_pairs}")
    print(f"All 8 drops exceed pre-registered criterion (>=3): "
          f"{all(r['n_pairs'] >= 3 for r in rows)}")

    out = {
        "baseline_pairs": baseline_pairs,
        "per_domain_drop": rows,
        "min_pairs_across_drops": min_pairs,
        "max_pairs_across_drops": max_pairs,
        "all_exceed_prereg_3": all(r["n_pairs"] >= 3 for r in rows),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
