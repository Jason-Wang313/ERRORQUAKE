"""ICC(2,1) and ICC(2,k) on dual-judge severity scores.

Two-way random effects, absolute agreement (Shrout & Fleiss 1979).
We treat each scored item (model x query) as a target and the two
judges (primary, secondary) as raters.

ICC(2,1): single-rater absolute agreement
  ICC(2,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E + k*(MS_C - MS_E)/n)
where:
  MS_R = between-target mean square
  MS_C = between-rater mean square
  MS_E = residual mean square
  k = number of raters (2)
  n = number of targets

ICC(2,k): k-rater absolute agreement (the average score)
  ICC(2,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E)/n)

Output: results/analysis/icc.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("C:/projects/errorquake")
SCORES = ROOT / "results" / "scores"
OUT = ROOT / "results" / "analysis" / "icc.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def icc_2_one_two(scores: np.ndarray) -> dict:
    """Compute ICC(2,1) and ICC(2,k) on a (n_targets, k_raters) matrix.

    Two-way random effects, absolute agreement, k=2.
    Shrout & Fleiss (1979) Table 1.
    """
    n, k = scores.shape
    grand = scores.mean()
    target_means = scores.mean(axis=1)  # length n
    rater_means = scores.mean(axis=0)   # length k

    # Between-target sum of squares
    SSR = k * np.sum((target_means - grand) ** 2)
    # Between-rater sum of squares
    SSC = n * np.sum((rater_means - grand) ** 2)
    # Total sum of squares
    SST = np.sum((scores - grand) ** 2)
    # Residual (interaction + error)
    SSE = SST - SSR - SSC

    df_R = n - 1
    df_C = k - 1
    df_E = (n - 1) * (k - 1)

    MSR = SSR / df_R if df_R > 0 else 0.0
    MSC = SSC / df_C if df_C > 0 else 0.0
    MSE = SSE / df_E if df_E > 0 else 0.0

    # ICC(2,1) — single rater, absolute agreement
    denom_1 = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    icc_2_1 = (MSR - MSE) / denom_1 if denom_1 > 0 else float("nan")

    # ICC(2,k) — k raters, absolute agreement
    denom_k = MSR + (MSC - MSE) / n
    icc_2_k = (MSR - MSE) / denom_k if denom_k > 0 else float("nan")

    return {
        "n_targets": int(n),
        "n_raters": int(k),
        "icc_2_1": float(icc_2_1),
        "icc_2_k": float(icc_2_k),
        "MSR": float(MSR),
        "MSC": float(MSC),
        "MSE": float(MSE),
    }


def main() -> None:
    print("=" * 70)
    print("ICC (Patch 1)")
    print("=" * 70)

    files = sorted(f for f in SCORES.glob("*.jsonl") if f.stem not in EXCLUDED)
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
            pri.append(float(p))
            sec.append(float(s))
        all_pri.extend(pri)
        all_sec.extend(sec)

    pri = np.array(all_pri)
    sec = np.array(all_sec)
    print(f"Pooled records: {len(pri)}")

    matrix = np.column_stack([pri, sec])
    icc = icc_2_one_two(matrix)

    print()
    print("Pooled ICC (two-way random effects, absolute agreement):")
    print(f"  ICC(2,1) single rater  = {icc['icc_2_1']:.3f}")
    print(f"  ICC(2,k=2) average     = {icc['icc_2_k']:.3f}")
    print(f"  MS_R={icc['MSR']:.3f}  MS_C={icc['MSC']:.3f}  MS_E={icc['MSE']:.3f}")

    # Approximate 95% CI for ICC(2,1) via Fisher transformation of ICC
    # (rough; for the exact F-based CI see Shrout & Fleiss eq 8)
    n = icc["n_targets"]
    k = icc["n_raters"]
    # F-statistic for testing ICC > 0
    F = icc["MSR"] / icc["MSE"] if icc["MSE"] > 0 else float("inf")
    print(f"  F = MSR/MSE = {F:.2f}")

    # Cicchetti scale interpretation
    cic = icc["icc_2_1"]
    if cic >= 0.75:
        interp = "excellent"
    elif cic >= 0.60:
        interp = "good"
    elif cic >= 0.40:
        interp = "fair"
    else:
        interp = "poor"
    print(f"  Cicchetti interpretation of ICC(2,1): {interp}")

    out = {
        "pooled": {
            **icc,
            "F_MSR_over_MSE": float(F),
            "cicchetti_interpretation": interp,
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
