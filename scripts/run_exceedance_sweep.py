"""Task 3.1: Exceedance threshold sweep (answers reviewer Q4).

For each minimum-exceedance threshold T in {30, 50, 75, 100, 150, 200}:
  1. Exclude models with fewer than T observations above their
     selected m_min (insufficient tail support).
  2. Recompute the Exp 2 disjoint-CI discriminator count on the
     surviving models.
  3. Recompute the Exp 5 scaling correlation on the surviving
     dense models, plus the partial correlation given epsilon.

Output: results/analysis/exceedance_sweep.json
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value

SCORES = ROOT / "results" / "scores"
ANALYSIS = ROOT / "results" / "analysis" / "full_21model_analysis.json"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "exceedance_sweep.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
THRESHOLDS = [30, 50, 75, 100, 150, 200]


def residualise(y, x):
    Xm = np.column_stack([np.ones(len(x)), x])
    c, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return y - Xm @ c


def load_records(stem: str) -> list[dict]:
    return [json.loads(l) for l in open(SCORES / f"{stem}.jsonl",
            encoding="utf-8") if l.strip()]


def main() -> None:
    print("=" * 70)
    print("EXCEEDANCE THRESHOLD SWEEP (Task 3.1, answers Q4)")
    print("=" * 70)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = {p["name"]: p for p in exp5["points"]}

    # Fresh fit per model from stored final_score (matches exp2_robustness
    # baseline so T=30 reproduces the 30-pair headline exactly).
    per_model = {}
    for name in points:
        if name in EXCLUDED:
            continue
        recs = load_records(name)
        scores = [r.get("final_score") for r in recs if r.get("final_score") is not None]
        arr = np.asarray(scores, dtype=float)
        pos = arr[arr > 0]
        eps = float((arr > 0).mean()) if arr.size > 0 else float("nan")
        if pos.size < 30:
            continue
        try:
            bv = estimate_b_value(pos, model_name=name)
        except Exception:
            continue
        per_model[name] = {
            "name": name,
            "architecture": points[name]["architecture"],
            "log_params": points[name]["log_params"],
            "b": float(bv.b),
            "ci_lo": float(bv.b_ci_lower),
            "ci_hi": float(bv.b_ci_upper),
            "eps": eps,
            "n_above": int(bv.n_above_mmin),
        }

    print(f"\n{'threshold':>10} {'n_surv':>7} {'n_dense':>8} "
          f"{'n_pairs':>8} {'rho':>8} {'p':>8} {'partial':>9} {'part_p':>8}")
    print("-" * 75)

    rows = []
    for T in THRESHOLDS:
        surviving = {k: v for k, v in per_model.items()
                     if v["n_above"] is not None and v["n_above"] >= T}
        # Discriminator pair count
        n_pairs = 0
        mod_list = list(surviving.keys())
        for a, b in combinations(mod_list, 2):
            va, vb = surviving[a], surviving[b]
            if (va["b"] is None or vb["b"] is None
                    or va["eps"] is None or vb["eps"] is None):
                continue
            if abs(va["eps"] - vb["eps"]) >= 0.05:
                continue
            if abs(va["b"] - vb["b"]) <= 0.15:
                continue
            # Keep only if CIs are disjoint: max(lo) > min(hi)
            if max(va["ci_lo"], vb["ci_lo"]) <= min(va["ci_hi"], vb["ci_hi"]):
                continue
            n_pairs += 1

        # Scaling on dense
        dense = [v for v in surviving.values() if v["architecture"] == "dense"
                 and v["b"] is not None]
        if len(dense) >= 5:
            lp = np.array([v["log_params"] for v in dense])
            bs = np.array([v["b"] for v in dense])
            eps = np.array([v["eps"] for v in dense])
            rho, p = stats.spearmanr(lp, bs)
            lp_r = residualise(lp, eps)
            b_r = residualise(bs, eps)
            p_rho, p_p = stats.spearmanr(lp_r, b_r)
        else:
            rho, p, p_rho, p_p = float("nan"), float("nan"), float("nan"), float("nan")

        rows.append({
            "threshold": T,
            "n_surviving": len(surviving),
            "n_dense_surviving": len(dense),
            "n_disjoint_pairs": int(n_pairs),
            "rho_scaling": float(rho) if not np.isnan(rho) else None,
            "p_scaling": float(p) if not np.isnan(p) else None,
            "partial_rho_given_eps": float(p_rho) if not np.isnan(p_rho) else None,
            "partial_p_given_eps": float(p_p) if not np.isnan(p_p) else None,
        })
        print(f"{T:>10} {len(surviving):>7} {len(dense):>8} "
              f"{n_pairs:>8} {rho:>+8.3f} {p:>8.4f} "
              f"{p_rho:>+9.3f} {p_p:>8.4f}")

    out = {"rows": rows, "thresholds_tested": THRESHOLDS}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

