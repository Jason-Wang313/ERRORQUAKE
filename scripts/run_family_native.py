"""Task 2.1: Family-native tail parameters + empirical tail ratio.

For each of 21 models, extract the BIC-best family's native shape
parameter and the empirical tail ratio
    tail_ratio = P(M >= 3.0 | M > 0) / P(M >= 1.0 | M > 0)

Both are family-independent summaries of tail heaviness. Compare
ranking vs the GR b-value.

For Option B: the question is "does the discriminator result
replicate with family-native summaries?" — i.e. do pairs with
matched accuracy still have distinguishable tail-shape signatures
under the family-native metric.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
SCORES = ROOT / "results" / "scores"
ANALYSIS = ROOT / "results" / "analysis" / "full_21model_analysis.json"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "family_native.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_positive(stem: str) -> np.ndarray:
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
        if s is not None and s > 0:
            out.append(float(s))
    return np.asarray(out, dtype=float)


def family_native_metric(dist: str, params: dict) -> float | None:
    """Map each family's shape param to a unified heavier-is-smaller scale.

    Returns a number where SMALLER = heavier tail (same as GR b).
    """
    if dist == "exponential":
        return float(params.get("lambda"))
    if dist == "stretched_exp":
        lam = params.get("lambda")
        gamma = params.get("gamma")
        if lam is None or gamma is None:
            return None
        # Effective decay rate for stretched exp in log space
        return float(lam ** (1.0 / max(gamma, 1e-6)))
    if dist == "truncated_power_law":
        # lambda term dominates for bounded discrete support
        return float(params.get("lambda", params.get("beta")))
    if dist == "lognormal":
        sigma = params.get("sigma")
        if sigma is None:
            return None
        return float(1.0 / max(sigma, 1e-6))
    if dist == "power_law":
        return float(params.get("beta"))
    return None


def main() -> None:
    print("=" * 70)
    print("TASK 2.1 — FAMILY-NATIVE TAIL PARAMETERS")
    print("=" * 70)

    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = {p["name"]: p for p in exp5["points"]}

    rows = []
    for name, d in analysis.items():
        if name in EXCLUDED:
            continue
        bf = d.get("best_fit", {})
        dist = bf.get("distribution")
        params = bf.get("parameters", {})
        native = family_native_metric(dist, params)
        # Empirical tail ratio (no fit, bounded-grid)
        pos = load_positive(name)
        n_pos = int(pos.size)
        if n_pos == 0:
            continue
        n_ge1 = int((pos >= 1.0 - 1e-9).sum())
        n_ge3 = int((pos >= 3.0 - 1e-9).sum())
        tail_ratio = n_ge3 / max(n_ge1, 1)
        # Reference GR b
        gr_b = d.get("b_value", {}).get("b")
        p5 = points.get(name, {})
        rows.append({
            "name": name,
            "architecture": p5.get("architecture"),
            "log_params": p5.get("log_params"),
            "error_rate": d.get("error_rate"),
            "best_fit": dist,
            "native_metric": native,
            "n_pos": n_pos,
            "n_ge_1": n_ge1,
            "n_ge_3": n_ge3,
            "tail_ratio": tail_ratio,
            "gr_b": gr_b,
        })

    print(f"\n{'model':<28} {'family':<20} {'native':>8} {'tail_ratio':>10} {'gr_b':>7}")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: x.get("gr_b") or 0):
        fm = (r["best_fit"] or "??")[:18]
        nm = f"{r['native_metric']:.3f}" if r["native_metric"] is not None else "  --  "
        print(f"{r['name']:<28} {fm:<20} {nm:>8} {r['tail_ratio']:>10.4f} "
              f"{(r['gr_b'] or 0):>7.3f}")

    # Consistency: rank correlation between GR b and native metric / tail ratio
    def corr(key):
        valid = [(r["gr_b"], r[key]) for r in rows
                 if r["gr_b"] is not None and r[key] is not None]
        if len(valid) < 5:
            return None
        a = np.array([v[0] for v in valid])
        b = np.array([v[1] for v in valid])
        rho, p = stats.spearmanr(a, b)
        return {"n": len(valid), "rho": float(rho), "p": float(p)}

    rho_native = corr("native_metric")
    rho_tail = corr("tail_ratio")
    print()
    print("Cross-check: does GR b track the family-native metric?")
    if rho_native:
        print(f"  Spearman(gr_b, family_native) = {rho_native['rho']:+.3f} "
              f"(p={rho_native['p']:.4f}, n={rho_native['n']})")
    if rho_tail:
        print(f"  Spearman(gr_b, tail_ratio)    = {rho_tail['rho']:+.3f} "
              f"(p={rho_tail['p']:.4f}, n={rho_tail['n']})")
    print("  (Higher rho = GR b is a good universal proxy)")

    # Discriminator replication: count |Δeps|<0.05 pairs with |Δtail_ratio| > threshold
    def count_disjoint_tail_pairs(threshold: float) -> dict:
        valid = [r for r in rows if r.get("error_rate") is not None]
        qual = 0
        for a, b in combinations(valid, 2):
            if abs(a["error_rate"] - b["error_rate"]) < 0.05:
                if abs(a["tail_ratio"] - b["tail_ratio"]) > threshold:
                    qual += 1
        return {"threshold": threshold, "n_qualifying": qual}

    print()
    print("Discriminator replication via empirical tail_ratio")
    print("  (|Δε|<0.05 pairs with |Δtail_ratio| > threshold):")
    tail_pair_counts = {}
    for t in (0.005, 0.010, 0.015, 0.020):
        c = count_disjoint_tail_pairs(t)
        tail_pair_counts[str(t)] = c["n_qualifying"]
        print(f"  threshold={t:.3f}: {c['n_qualifying']} pairs")

    out = {
        "per_model": rows,
        "gr_b_vs_native_metric": rho_native,
        "gr_b_vs_tail_ratio": rho_tail,
        "discriminator_via_tail_ratio": tail_pair_counts,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
