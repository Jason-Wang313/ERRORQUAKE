"""Task 3.2: per-model threshold count table for deployment.

For each of the 21 models, count errors at M >= {1.5, 2.0, 2.5, 3.0,
3.5, 4.0}. Report absolute count, rate, and expected events per
million queries under i.i.d. deployment with 95% binomial CIs
(Wilson score interval).

Output: results/analysis/deployment_table.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "results" / "scores"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "deployment_table.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}

THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def load_scores(stem: str) -> np.ndarray:
    out = []
    for line in open(SCORES / f"{stem}.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = r.get("final_score")
        if s is not None:
            out.append(float(s))
    return np.asarray(out, dtype=float)


def main() -> None:
    print("=" * 80)
    print("DEPLOYMENT TABLE (Task 3.2)")
    print("=" * 80)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = {p["name"]: p for p in exp5["points"]}

    rows = []
    for name, p in sorted(points.items(), key=lambda kv: kv[1]["log_params"]):
        if name in EXCLUDED:
            continue
        scores = load_scores(name)
        n_total = int(scores.size)
        if n_total == 0:
            continue
        row = {
            "name": name,
            "architecture": p["architecture"],
            "active_params_b": p["active_params_b"],
            "error_rate": p["error_rate"],
            "b_value": p["b_value"],
            "n_queries": n_total,
        }
        for t in THRESHOLDS:
            k = int((scores >= t - 1e-9).sum())
            rate = k / n_total
            per_million = rate * 1e6
            lo, hi = wilson_ci(k, n_total)
            row[f"n_ge_{t}"] = k
            row[f"rate_ge_{t}"] = rate
            row[f"per_million_ge_{t}"] = per_million
            row[f"ci_lo_per_million_ge_{t}"] = lo * 1e6
            row[f"ci_hi_per_million_ge_{t}"] = hi * 1e6
        rows.append(row)

    # Print compact table (counts)
    print()
    header = f"{'model':<28} {'ε':>6} " + "".join(f"{f'≥{t}':>6}" for t in THRESHOLDS)
    print(header)
    print("-" * len(header))
    for r in rows:
        counts = "".join(f"{r[f'n_ge_{t}']:>6}" for t in THRESHOLDS)
        print(f"{r['name']:<28} {r['error_rate']:>6.3f} {counts}")

    # Print per-million
    print()
    print("Expected events per 1,000,000 queries:")
    header = f"{'model':<28} " + "".join(f"{f'M≥{t}':>8}" for t in THRESHOLDS)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = "".join(f"{r[f'per_million_ge_{t}']:>8.0f}" for t in THRESHOLDS)
        print(f"{r['name']:<28} {vals}")

    out = {
        "thresholds": THRESHOLDS,
        "per_model": rows,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

