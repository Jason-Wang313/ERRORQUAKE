"""Patch 3: per-tier scaling correlation.

For each tier T1..T5, fit b on within-tier positive scores per model
and compute Spearman(log10(active_params), b_tier) on the 14 dense
models. Detects Simpson's paradox: if no individual tier shows the
correlation but the aggregated does, the headline is fragile.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value

SCORES = ROOT / "results" / "scores"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "per_tier_scaling.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_per_tier(stem: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for line in open(SCORES / f"{stem}.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = r.get("tier")
        s = r.get("final_score")
        if t in out and s is not None:
            out[t].append(float(s))
    return out


def main() -> None:
    print("=" * 70)
    print("PER-TIER SCALING (Patch 3)")
    print("=" * 70)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = {p["name"]: p for p in exp5["points"]}
    dense = [p for p in exp5["points"] if p["architecture"] == "dense"]
    print(f"Dense models: {len(dense)}")

    # Per (model, tier): fit b on positive scores; allow None if too few
    per_model_tier = {}
    print()
    print(f"{'model':<28} " + " ".join(f"T{t}".rjust(7) for t in range(1, 6)))
    print("-" * 75)
    for d in dense:
        name = d["name"]
        if name in EXCLUDED:
            continue
        per_tier = load_per_tier(name)
        bs = {}
        for t in range(1, 6):
            scores = np.asarray(per_tier[t], dtype=float)
            errors = scores[scores > 0]
            if errors.size < 30:
                bs[t] = None
            else:
                try:
                    bv = estimate_b_value(errors, model_name=f"{name}_T{t}")
                    bs[t] = float(bv.b)
                except Exception:
                    bs[t] = None
        per_model_tier[name] = {"log_params": d["log_params"], "b_per_tier": bs}
        line = " ".join((f"{bs[t]:.3f}" if bs[t] is not None else "  --   ").rjust(7)
                        for t in range(1, 6))
        print(f"{name:<28} {line}")

    # Correlations per tier
    print()
    print("=" * 70)
    print("PER-TIER SCALING CORRELATIONS (dense, n=14)")
    print("=" * 70)
    print(f"{'tier':<6} {'n':>4} {'rho':>8} {'p':>8}  sign")
    print("-" * 40)
    per_tier_results = {}
    for t in range(1, 6):
        valid = [(per_model_tier[m]["log_params"], per_model_tier[m]["b_per_tier"][t])
                 for m in per_model_tier
                 if per_model_tier[m]["b_per_tier"][t] is not None]
        if len(valid) < 5:
            per_tier_results[t] = {"n": len(valid), "rho": None, "p": None}
            continue
        xs = np.array([v[0] for v in valid])
        ys = np.array([v[1] for v in valid])
        rho, p = stats.spearmanr(xs, ys)
        sign = "neg" if rho < 0 else "pos"
        per_tier_results[t] = {"n": len(valid), "rho": float(rho), "p": float(p), "sign": sign}
        print(f"T{t:<5} {len(valid):>4} {rho:>+8.3f} {p:>8.4f}  {sign}")

    out = {
        "per_model_tier": per_model_tier,
        "per_tier_correlations": per_tier_results,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
