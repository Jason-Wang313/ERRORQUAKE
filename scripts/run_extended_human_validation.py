"""Task 1.4: extended human validation on the 100-item pilot set.

Joins:
  data/pilot/human_ratings_claude_rater.jsonl  (rating_id, score_11point)
  data/pilot/human_rating_key.jsonl            (rating_id, query_id, model, final_score, ...)

Computes:
  (a) ECE of judge final_score against human score
  (a') ECE restricted to the tail (judge or human M >= 2.0)
  (b) Sensitivity / specificity / PPV / NPV at thresholds M* in {2.0, 2.5, 3.0}
  (c) Per-model tail weight (count(human >= 2.5) / count(human > 0)) and
      Spearman with log_params across dense models with >=5 human errors.
  (d) Confusion matrix on the 9-level grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RATINGS = ROOT / "data" / "pilot" / "human_ratings_claude_rater.jsonl"
KEYS = ROOT / "data" / "pilot" / "human_rating_key.jsonl"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "extended_human_validation.json"


def main() -> None:
    print("=" * 70)
    print("EXTENDED HUMAN VALIDATION (v5 Task 1.4) — on 100-item pilot")
    print("=" * 70)

    ratings = {}
    for line in open(RATINGS, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ratings[r["rating_id"]] = float(r["score_11point"])

    joined = []
    for line in open(KEYS, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            k = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = k["rating_id"]
        if rid in ratings and k.get("final_score") is not None:
            joined.append({
                "rating_id": rid,
                "query_id": k["query_id"],
                "model": k.get("model_name"),
                "tier": k.get("tier"),
                "domain": k.get("domain"),
                "human_score": ratings[rid],
                "judge_score": float(k["final_score"]),
            })

    print(f"Joined {len(joined)} items")

    if not joined:
        print("No joined items — aborting.")
        return

    human = np.array([j["human_score"] for j in joined])
    judge = np.array([j["judge_score"] for j in joined])

    # (a) ECE
    # Bin by judge score and compare mean judge to mean human per bin
    bins = np.arange(0.0, 4.5, 0.5)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (judge >= lo - 1e-9) & (judge < hi)
        n_bin = int(mask.sum())
        if n_bin > 0:
            mj = float(judge[mask].mean())
            mh = float(human[mask].mean())
            ece += n_bin / len(human) * abs(mj - mh)
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": n_bin,
                         "mean_judge": mj, "mean_human": mh})

    # Tail-restricted ECE (judge or human >= 2.0)
    tail_mask = (judge >= 2.0) | (human >= 2.0)
    tail_ece = 0.0
    n_tail = int(tail_mask.sum())
    if n_tail > 0:
        tail_human = human[tail_mask]
        tail_judge = judge[tail_mask]
        tail_ece = float(np.mean(np.abs(tail_judge - tail_human)))

    print(f"\n(a) Overall ECE (binned, judge vs human): {ece:.3f}")
    print(f"(a') Tail mean-abs-diff (M>=2 either): {tail_ece:.3f} on {n_tail} items")

    # (b) Sensitivity/specificity at thresholds
    print()
    print(f"(b) Classification at severity thresholds:")
    print(f"    {'M*':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6}")
    thresh_results = {}
    for m_star in (2.0, 2.5, 3.0):
        tp = int(((human >= m_star) & (judge >= m_star)).sum())
        fp = int(((human < m_star) & (judge >= m_star)).sum())
        fn = int(((human >= m_star) & (judge < m_star)).sum())
        tn = int(((human < m_star) & (judge < m_star)).sum())
        sens = tp / max(tp + fn, 1) if (tp + fn) > 0 else None
        spec = tn / max(tn + fp, 1) if (tn + fp) > 0 else None
        ppv = tp / max(tp + fp, 1) if (tp + fp) > 0 else None
        npv = tn / max(tn + fn, 1) if (tn + fn) > 0 else None
        print(f"    {m_star:>4} {tp:>4} {fp:>4} {fn:>4} {tn:>4} "
              f"{(sens or 0):>6.3f} {(spec or 0):>6.3f} "
              f"{(ppv or 0):>6.3f} {(npv or 0):>6.3f}")
        thresh_results[f"m_ge_{m_star}"] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens, "specificity": spec,
            "ppv": ppv, "npv": npv,
        }

    # (c) Per-model tail weight and scaling correlation
    print()
    print("(c) Per-model human-based tail weight:")
    per_model = {}
    for j in joined:
        m = j["model"]
        if m not in per_model:
            per_model[m] = {"scores": [], "n_errors": 0, "n_tail": 0}
        per_model[m]["scores"].append(j["human_score"])
    for m, d in per_model.items():
        arr = np.array(d["scores"])
        d["n"] = int(arr.size)
        d["n_errors"] = int((arr > 0).sum())
        d["n_tail"] = int((arr >= 2.5).sum())
        d["tail_weight"] = (d["n_tail"] / d["n_errors"]) if d["n_errors"] > 0 else None
        print(f"    {m:<28} n={d['n']:>3}  err={d['n_errors']:>3}  "
              f"tail(>=2.5)={d['n_tail']:>2}  "
              f"tw={(d['tail_weight'] or 0):.3f}")

    # Cross-reference with Exp 5 dense subset
    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    dense = {p["name"]: p for p in exp5["points"] if p["architecture"] == "dense"}
    matched = []
    for m, d in per_model.items():
        if m in dense and d["n_errors"] >= 5 and d["tail_weight"] is not None:
            matched.append((dense[m]["log_params"], d["tail_weight"], m))
    if len(matched) >= 3:
        lp = np.array([x[0] for x in matched])
        tw = np.array([x[1] for x in matched])
        rho, p = stats.spearmanr(lp, tw)
        print(f"\n    Human tail-weight Spearman(log_params, tail_weight) = "
              f"{rho:+.3f} (p={p:.3f}, n={len(matched)} dense models)")
        human_scaling = {
            "rho": float(rho), "p": float(p), "n": len(matched),
            "matched_models": [m for _, _, m in matched],
        }
    else:
        print(f"\n    Insufficient dense models with >=5 human errors "
              f"({len(matched)}); skipping human-only scaling test")
        human_scaling = {"error": "insufficient", "n": len(matched)}

    # (d) Confusion matrix
    grid = np.arange(0.0, 4.5, 0.5)
    confmat = np.zeros((9, 9), dtype=int)
    for h, j in zip(human, judge):
        hi = int(np.argmin(np.abs(grid - h)))
        ji = int(np.argmin(np.abs(grid - j)))
        confmat[hi, ji] += 1

    out = {
        "n_items_joined": len(joined),
        "ece_overall": float(ece),
        "ece_bins": rows,
        "tail_mean_abs_diff": float(tail_ece),
        "n_tail": int(n_tail),
        "thresholds": thresh_results,
        "per_model_human": {m: {k: v for k, v in d.items() if k != "scores"}
                            for m, d in per_model.items()},
        "human_only_scaling": human_scaling,
        "confusion_matrix": confmat.tolist(),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

