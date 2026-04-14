"""Inter-judge agreement (weighted Cohen's kappa) on the dual-judge
score pairs across all 21 evaluated models. Reviewer Fix 7.

Output: results/analysis/judge_agreement.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "results" / "scores"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "judge_agreement.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}

# Round scores to nearest 0.5 grid for kappa categorisation.
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def quantize(x):
    """Snap to nearest grid point and return the integer index 0..8."""
    return int(np.argmin(np.abs(GRID - float(x))))


def main() -> None:
    print("=" * 70)
    print("INTER-JUDGE AGREEMENT (Reviewer Fix 7)")
    print("=" * 70)

    files = sorted(f for f in SCORES.glob("*.jsonl") if f.stem not in EXCLUDED)
    rows = []
    all_pri = []
    all_sec = []

    for f in files:
        pri = []
        sec = []
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("primary_score")
            s = r.get("secondary_score")
            if p is None or s is None:
                continue
            pri.append(quantize(p))
            sec.append(quantize(s))
        if not pri:
            continue
        pri_arr = np.array(pri)
        sec_arr = np.array(sec)
        # Linear-weighted Cohen's kappa
        kw = float(cohen_kappa_score(pri_arr, sec_arr, weights="linear"))
        kq = float(cohen_kappa_score(pri_arr, sec_arr, weights="quadratic"))
        # Spearman as a sanity check
        rho, _ = stats.spearmanr(pri_arr, sec_arr)
        # Mean absolute disagreement
        mad = float(np.mean(np.abs(pri_arr - sec_arr)))
        # Fraction with |diff| > 1.0
        big_disagree = float(np.mean(np.abs(pri_arr - sec_arr) > 1.0))
        rows.append({
            "model": f.stem,
            "n": int(len(pri)),
            "kappa_linear": kw,
            "kappa_quadratic": kq,
            "spearman": float(rho),
            "mean_abs_diff": mad,
            "frac_diff_gt_1": big_disagree,
        })
        all_pri.extend(pri)
        all_sec.extend(sec)

    rows.sort(key=lambda r: -r["kappa_linear"])
    print()
    print(f"{'model':<28} {'n':>5} {'k_lin':>7} {'k_quad':>7} {'rho':>6} {'MAD':>6} {'>1':>6}")
    print("-" * 70)
    for r in rows:
        print(f"{r['model']:<28} {r['n']:>5} {r['kappa_linear']:>7.3f} "
              f"{r['kappa_quadratic']:>7.3f} {r['spearman']:>6.3f} "
              f"{r['mean_abs_diff']:>6.3f} {r['frac_diff_gt_1']:>6.3f}")

    # Pooled kappa across all records
    all_pri = np.array(all_pri)
    all_sec = np.array(all_sec)
    pooled_lin = float(cohen_kappa_score(all_pri, all_sec, weights="linear"))
    pooled_quad = float(cohen_kappa_score(all_pri, all_sec, weights="quadratic"))
    pooled_rho, _ = stats.spearmanr(all_pri, all_sec)
    pooled_mad = float(np.mean(np.abs(all_pri - all_sec)))

    print()
    print("=" * 70)
    print(f"Pooled across {len(all_pri)} records:")
    print(f"  Linear-weighted Cohen's kappa  = {pooled_lin:.3f}")
    print(f"  Quadratic-weighted kappa       = {pooled_quad:.3f}")
    print(f"  Spearman                       = {float(pooled_rho):.3f}")
    print(f"  Mean abs diff                  = {pooled_mad:.3f}")

    # Q: does disagreement correlate with model size?
    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = {p["name"]: p for p in exp5["points"]}
    matched = [(r, points[r["model"]]) for r in rows if r["model"] in points]
    if matched:
        log_p = np.array([p["log_params"] for _, p in matched])
        mads = np.array([r["mean_abs_diff"] for r, _ in matched])
        kappas = np.array([r["kappa_linear"] for r, _ in matched])
        rho_size_mad, p_size_mad = stats.spearmanr(log_p, mads)
        rho_size_k, p_size_k = stats.spearmanr(log_p, kappas)
        print(f"\nDisagreement vs.\\ size (n={len(matched)}):")
        print(f"  Spearman(log_params, mean_abs_diff) = {rho_size_mad:+.3f} (p={p_size_mad:.4f})")
        print(f"  Spearman(log_params, kappa_linear)  = {rho_size_k:+.3f} (p={p_size_k:.4f})")

    out = {
        "per_model": rows,
        "pooled": {
            "n_records": int(len(all_pri)),
            "kappa_linear": pooled_lin,
            "kappa_quadratic": pooled_quad,
            "spearman": float(pooled_rho),
            "mean_abs_diff": pooled_mad,
        },
        "size_correlation": {
            "spearman_log_params_mad": float(rho_size_mad) if matched else None,
            "p_log_params_mad": float(p_size_mad) if matched else None,
            "spearman_log_params_kappa": float(rho_size_k) if matched else None,
            "p_log_params_kappa": float(p_size_k) if matched else None,
            "n": len(matched),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
