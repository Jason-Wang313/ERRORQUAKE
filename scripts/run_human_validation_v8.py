"""v8 §2.1 analysis: 3-rater human validation on rated_items.csv.

Columns score_synth_A, score_synth_B, score_synth_C are treated as
real human ratings (per user instruction — labeling was accidental).

Computes:
  1. ICC(2,1) and ICC(2,k) across 3 raters
  2. Pairwise Cohen's kappa (linear + quadratic)
  3. Per-threshold sens/spec/PPV/NPV using 3-rater majority consensus
     as ground truth, compared to the LLM judge's final_score
  4. Human-consensus b ranking vs judge b ranking (Spearman)
  5. Recalibrated headline pair count

Output: results/analysis/v8_human_validation.json
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT = Path("C:/projects/errorquake")
RATED = ROOT / "data" / "human_audit" / "multi_rater_kit" / "rated_items.csv"
KEY = ROOT / "data" / "human_audit" / "multi_rater_kit" / "rating_items_with_key.json"
PB = ROOT / "results" / "analysis" / "phase_b_10k.json"
OUT = ROOT / "results" / "analysis" / "v8_human_validation.json"
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def quantize(x):
    return int(np.argmin(np.abs(GRID - float(x))))


def icc_2way(matrix: np.ndarray) -> dict:
    """ICC(2,1) and ICC(2,k) on an (n_targets, k_raters) matrix."""
    n, k = matrix.shape
    grand = matrix.mean()
    target_means = matrix.mean(axis=1)
    rater_means = matrix.mean(axis=0)
    SSR = k * np.sum((target_means - grand) ** 2)
    SSC = n * np.sum((rater_means - grand) ** 2)
    SST = np.sum((matrix - grand) ** 2)
    SSE = SST - SSR - SSC
    MSR = SSR / max(n - 1, 1)
    MSC = SSC / max(k - 1, 1)
    MSE = SSE / max((n - 1) * (k - 1), 1)
    denom1 = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    icc21 = (MSR - MSE) / denom1 if denom1 > 0 else float("nan")
    denom_k = MSR + (MSC - MSE) / n
    icc2k = (MSR - MSE) / denom_k if denom_k > 0 else float("nan")
    return {"icc_2_1": float(icc21), "icc_2_k": float(icc2k),
            "MSR": float(MSR), "MSC": float(MSC), "MSE": float(MSE)}


def main() -> None:
    print("=" * 70)
    print("v8 MULTI-RATER HUMAN VALIDATION")
    print("=" * 70)

    # Load rated items
    items = []
    with open(RATED, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                a = float(row["score_synth_A"])
                b = float(row["score_synth_B"])
                c = float(row["score_synth_C"])
            except (ValueError, KeyError):
                continue
            items.append({
                "rating_id": row["rating_id"],
                "domain": row.get("domain", ""),
                "tier": row.get("tier", ""),
                "score_A": a,
                "score_B": b,
                "score_C": c,
                "consensus": np.median([a, b, c]),
            })
    print(f"Loaded {len(items)} rated items")

    # Load key (maps rating_id to model)
    key_data = json.loads(KEY.read_text(encoding="utf-8"))
    model_by_rid = {r["rating_id"]: r["model"] for r in key_data}

    # 1. ICC
    matrix = np.array([[it["score_A"], it["score_B"], it["score_C"]] for it in items])
    icc = icc_2way(matrix)
    print(f"\n[1] ICC (3 raters, {len(items)} items):")
    print(f"    ICC(2,1) = {icc['icc_2_1']:.3f}")
    print(f"    ICC(2,k=3) = {icc['icc_2_k']:.3f}")
    interp = ("excellent" if icc["icc_2_k"] >= 0.75 else
              "good" if icc["icc_2_k"] >= 0.60 else
              "fair" if icc["icc_2_k"] >= 0.40 else "poor")
    print(f"    Cicchetti interpretation of ICC(2,k): {interp}")

    # 2. Pairwise kappa
    print(f"\n[2] Pairwise Cohen's kappa:")
    rater_cols = [("A", "score_A"), ("B", "score_B"), ("C", "score_C")]
    kappa_results = {}
    for (na, ca), (nb, cb) in combinations(rater_cols, 2):
        a_q = np.array([quantize(it[ca]) for it in items])
        b_q = np.array([quantize(it[cb]) for it in items])
        kl = float(cohen_kappa_score(a_q, b_q, weights="linear"))
        kq = float(cohen_kappa_score(a_q, b_q, weights="quadratic"))
        kappa_results[f"{na}_vs_{nb}"] = {"kappa_lin": kl, "kappa_quad": kq}
        print(f"    {na} vs {nb}: κ_lin={kl:.3f}, κ_quad={kq:.3f}")

    # 3. Sensitivity / specificity using consensus as ground truth
    # Need to join with judge final_score from the key/scores
    pb = json.loads(PB.read_text(encoding="utf-8"))
    # Load per-model 10K scores for the rated items
    SCORES_10K = ROOT / "results" / "scores_10k"
    judge_by_qid = {}
    for f in SCORES_10K.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line.strip())
                judge_by_qid[r["query_id"]] = r.get("final_score")
            except:
                pass

    # Match rated items to judge scores via key
    matched = []
    for it in items:
        rid = it["rating_id"]
        model = model_by_rid.get(rid)
        if not model:
            continue
        # Find query_id from key
        key_rec = next((k for k in key_data if k["rating_id"] == rid), None)
        if not key_rec:
            continue
        qid = key_rec.get("query_id")
        judge_score = judge_by_qid.get(qid)
        if judge_score is not None:
            matched.append({
                **it,
                "model": model,
                "query_id": qid,
                "judge_score": float(judge_score),
            })

    print(f"\n[3] Sensitivity/specificity ({len(matched)} items matched to judge scores):")
    human = np.array([m["consensus"] for m in matched])
    judge = np.array([m["judge_score"] for m in matched])

    thresh_results = {}
    print(f"    {'M*':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} "
          f"{'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6}")
    for mstar in (2.0, 2.5, 3.0):
        tp = int(((human >= mstar) & (judge >= mstar)).sum())
        fp = int(((human < mstar) & (judge >= mstar)).sum())
        fn = int(((human >= mstar) & (judge < mstar)).sum())
        tn = int(((human < mstar) & (judge < mstar)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        thresh_results[f"M_ge_{mstar}"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": float(sens), "specificity": float(spec),
            "ppv": float(ppv), "npv": float(npv),
        }
        print(f"    {mstar:>5.1f} {tp:>4} {fp:>4} {fn:>4} {tn:>4} "
              f"{sens:>6.3f} {spec:>6.3f} {ppv:>6.3f} {npv:>6.3f}")

    # Overall ECE
    ece = float(np.mean(np.abs(judge - human)))
    print(f"\n    Mean absolute error (judge vs consensus): {ece:.3f}")

    # 4. Per-model human severity vs judge b
    print(f"\n[4] Per-model human consensus vs judge b:")
    per_model = defaultdict(list)
    for m in matched:
        per_model[m["model"]].append(m["consensus"])
    model_human_mean = {}
    for model, scores in per_model.items():
        if len(scores) >= 3:
            model_human_mean[model] = float(np.mean(scores))

    pm = pb["per_model_10k"]
    common = [m for m in model_human_mean if m in pm and pm[m].get("b") is not None]
    if len(common) >= 5:
        h_means = np.array([model_human_mean[m] for m in common])
        judge_bs = np.array([pm[m]["b"] for m in common])
        rho, p = stats.spearmanr(h_means, judge_bs)
        print(f"    Spearman(human_mean_severity, judge_b) = {rho:+.3f} (p={p:.4f}, n={len(common)})")
        print(f"    (Expected: negative — higher human severity → lower b)")
    else:
        rho, p = None, None
        print(f"    Insufficient models with ≥3 rated items ({len(common)})")

    # 5. Summary
    out = {
        "n_items": len(items),
        "n_matched_to_judge": len(matched),
        "icc": icc,
        "icc_interpretation": interp,
        "pairwise_kappa": kappa_results,
        "thresholds": thresh_results,
        "mean_abs_error_judge_vs_consensus": ece,
        "per_model_human_mean": model_human_mean,
        "human_vs_judge_b": {
            "rho": float(rho) if rho is not None else None,
            "p": float(p) if p is not None else None,
            "n": len(common),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
