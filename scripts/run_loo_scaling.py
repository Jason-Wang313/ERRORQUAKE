"""Leave-one-out robustness for the Exp 5 scaling correlation.

For each of the 14 dense models, drop it and recompute the Spearman
correlation between log10(active params) and b-value on the remaining
13. Report the min/median/max rho and whether the p-value stays below
0.05 in every case.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ANALYSIS = Path("C:/projects/errorquake/results/analysis/exp5_scaling.json")
OUT = Path("C:/projects/errorquake/results/analysis/exp5_loo.json")

data = json.loads(ANALYSIS.read_text(encoding="utf-8"))
points = data["points"]
dense = [p for p in points if p["architecture"] == "dense"]
print(f"Dense models: {len(dense)}")

log_p = np.array([p["log_params"] for p in dense])
bs = np.array([p["b_value"] for p in dense])
baseline_rho, baseline_p = stats.spearmanr(log_p, bs)
print(f"Baseline Spearman rho = {baseline_rho:+.3f} (p = {baseline_p:.4f})")

loo = []
for i, p in enumerate(dense):
    mask = np.ones(len(dense), dtype=bool)
    mask[i] = False
    rho, pval = stats.spearmanr(log_p[mask], bs[mask])
    loo.append({
        "dropped": p["name"],
        "rho": float(rho),
        "p": float(pval),
        "still_significant": bool(float(pval) < 0.05),
        "sign_preserved": bool(float(rho) < 0),
    })

loo.sort(key=lambda r: r["rho"])
print()
print(f"{'dropped model':<28} {'rho':>8} {'p':>8}  sig? sign?")
print("-" * 60)
for r in loo:
    sig = "YES" if r["still_significant"] else "NO "
    sgn = "neg" if r["sign_preserved"] else "POS"
    print(f"{r['dropped']:<28} {r['rho']:>+8.3f} {r['p']:>8.4f}  {sig}  {sgn}")

rhos = np.array([r["rho"] for r in loo])
ps = np.array([r["p"] for r in loo])
n_sig = sum(r["still_significant"] for r in loo)
n_sign = sum(r["sign_preserved"] for r in loo)

print()
print("=" * 60)
print(f"LOO min rho = {rhos.min():+.3f}")
print(f"LOO median  = {np.median(rhos):+.3f}")
print(f"LOO max     = {rhos.max():+.3f}")
print(f"Significant (p<0.05) after drop: {n_sig}/{len(loo)}")
print(f"Sign preserved (rho<0) after drop: {n_sign}/{len(loo)}")

summary = {
    "baseline_rho": float(baseline_rho),
    "baseline_p": float(baseline_p),
    "n_dense": len(dense),
    "loo_min_rho": float(rhos.min()),
    "loo_median_rho": float(np.median(rhos)),
    "loo_max_rho": float(rhos.max()),
    "loo_max_p": float(ps.max()),
    "n_significant_after_drop": int(n_sig),
    "n_sign_preserved": int(n_sign),
    "per_drop": loo,
}
OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\nSaved -> {OUT}")
