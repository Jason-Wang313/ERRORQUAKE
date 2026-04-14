"""Gold-standard recalibration using 3-rater human consensus.

1. Fit calibration: isotonic regression from judge_score to human
   consensus on the 191 matched items.
2. Apply calibration to ALL 10K judge scores.
3. Refit b on calibrated scores for all 21 models.
4. Recount disjoint-CI pairs.
5. Recompute sensitivity/specificity on calibrated scores.
6. Compare calibrated vs uncalibrated headline.

Output: results/analysis/v10_calibrated.json
        results/scores_10k_calibrated/{model}.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value

RATED = ROOT / "data" / "human_audit" / "multi_rater_kit" / "rated_items.csv"
KEY = ROOT / "data" / "human_audit" / "multi_rater_kit" / "rating_items_with_key.json"
SCORES_10K = ROOT / "results" / "scores_10k"
SCORES_CAL = ROOT / "results" / "scores_10k_calibrated"
PB = ROOT / "results" / "analysis" / "phase_b_10k.json"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "v10_calibrated.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def snap_to_grid(x: float) -> float:
    return float(GRID[int(np.argmin(np.abs(GRID - x)))])


def main() -> None:
    print("=" * 70)
    print("GOLD-STANDARD RECALIBRATION")
    print("=" * 70)

    # 1. Build calibration dataset
    items = []
    with open(RATED, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                a = float(row["score_synth_A"])
                b = float(row["score_synth_B"])
                c = float(row["score_synth_C"])
            except (ValueError, KeyError):
                continue
            items.append({"rating_id": row["rating_id"],
                          "consensus": float(np.median([a, b, c]))})
    rid_to_consensus = {it["rating_id"]: it["consensus"] for it in items}

    key_data = json.loads(KEY.read_text(encoding="utf-8"))
    rid_to_model_qid = {k["rating_id"]: (k["model"], k["query_id"])
                        for k in key_data}

    # Load judge scores for matched items
    judge_by_mqid = {}
    for f in SCORES_10K.glob("*.jsonl"):
        if f.stem in EXCLUDED:
            continue
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line.strip())
                judge_by_mqid[(f.stem, r["query_id"])] = r.get("final_score")
            except:
                pass

    matched = []
    for rid, consensus in rid_to_consensus.items():
        if rid not in rid_to_model_qid:
            continue
        model, qid = rid_to_model_qid[rid]
        judge = judge_by_mqid.get((model, qid))
        if judge is not None:
            matched.append({"judge": float(judge), "human": consensus})

    judge_arr = np.array([m["judge"] for m in matched])
    human_arr = np.array([m["human"] for m in matched])
    print(f"Calibration set: {len(matched)} matched items")
    print(f"  Judge range: [{judge_arr.min():.1f}, {judge_arr.max():.1f}]")
    print(f"  Human range: [{human_arr.min():.1f}, {human_arr.max():.1f}]")
    print(f"  Mean judge: {judge_arr.mean():.3f}, mean human: {human_arr.mean():.3f}")

    # 2. Fit isotonic regression (monotone increasing: higher judge → higher human)
    iso = IsotonicRegression(y_min=0.0, y_max=4.0, increasing=True,
                              out_of_bounds="clip")
    iso.fit(judge_arr, human_arr)

    # Show the calibration curve at grid points
    print("\n  Calibration curve (judge → calibrated):")
    for g in GRID:
        cal = iso.predict([g])[0]
        print(f"    judge={g:.1f} → calibrated={cal:.2f}")

    # 3. Apply calibration to all 10K scores
    SCORES_CAL.mkdir(parents=True, exist_ok=True)
    models = sorted(f.stem for f in SCORES_10K.glob("*.jsonl")
                    if f.stem not in EXCLUDED)

    per_model_cal = {}
    for m in models:
        recs = [json.loads(l) for l in open(SCORES_10K / f"{m}.jsonl",
                encoding="utf-8") if l.strip()]
        cal_recs = []
        for r in recs:
            r2 = dict(r)
            fs = r.get("final_score")
            if fs is not None:
                cal = float(iso.predict([float(fs)])[0])
                r2["final_score_original"] = fs
                r2["final_score"] = snap_to_grid(cal)
            cal_recs.append(r2)
        # Write calibrated scores
        with open(SCORES_CAL / f"{m}.jsonl", "w", encoding="utf-8") as f:
            for r in cal_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Fit b on calibrated
        scores = np.array([r["final_score"] for r in cal_recs
                           if r.get("final_score") is not None], dtype=float)
        pos = scores[scores > 0]
        eps = float((scores > 0).mean())
        try:
            bv = estimate_b_value(pos, model_name=f"{m}_cal")
            per_model_cal[m] = {
                "b": float(bv.b), "ci_lo": float(bv.b_ci_lower),
                "ci_hi": float(bv.b_ci_upper), "eps": eps,
                "n_pos": int(pos.size),
            }
        except Exception:
            per_model_cal[m] = {"b": None, "ci_lo": None, "ci_hi": None,
                                "eps": eps, "n_pos": int(pos.size)}

    # 4. Count disjoint-CI pairs on calibrated b
    valid = [m for m in per_model_cal if per_model_cal[m].get("b") is not None]
    qualifying = 0
    disjoint = 0
    for a, b in combinations(valid, 2):
        va, vb = per_model_cal[a], per_model_cal[b]
        if abs(va["eps"] - vb["eps"]) >= 0.05:
            continue
        if abs(va["b"] - vb["b"]) <= 0.15:
            continue
        qualifying += 1
        if max(va["ci_lo"], vb["ci_lo"]) > min(va["ci_hi"], vb["ci_hi"]):
            disjoint += 1

    print(f"\n  Calibrated headline:")
    print(f"    Models with valid b: {len(valid)}")
    print(f"    Qualifying pairs: {qualifying}")
    print(f"    Disjoint-CI pairs: {disjoint}")

    # Compare with uncalibrated
    pb = json.loads(PB.read_text(encoding="utf-8"))
    uncal_pairs = pb.get("exp2_10k", {}).get("n_disjoint_CIs", 0)
    print(f"    Uncalibrated pairs: {uncal_pairs}")
    print(f"    Change: {disjoint - uncal_pairs:+d}")

    # 5. Sensitivity/specificity on calibrated scores
    print("\n  Calibrated sensitivity (using CALIBRATED judge vs human consensus):")
    cal_matched = []
    for m_item in matched:
        cal_score = float(iso.predict([m_item["judge"]])[0])
        cal_matched.append({"cal_judge": snap_to_grid(cal_score),
                            "human": m_item["human"]})
    cal_j = np.array([m["cal_judge"] for m in cal_matched])
    hum = np.array([m["human"] for m in cal_matched])

    cal_thresh = {}
    print(f"    {'M*':>5} {'Sens':>6} {'Spec':>6} {'PPV':>6}")
    for mstar in (2.0, 2.5, 3.0):
        tp = int(((hum >= mstar) & (cal_j >= mstar)).sum())
        fp = int(((hum < mstar) & (cal_j >= mstar)).sum())
        fn = int(((hum >= mstar) & (cal_j < mstar)).sum())
        tn = int(((hum < mstar) & (cal_j < mstar)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        cal_thresh[f"M_ge_{mstar}"] = {
            "sensitivity": float(sens), "specificity": float(spec),
            "ppv": float(ppv), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
        print(f"    {mstar:>5.1f} {sens:>6.3f} {spec:>6.3f} {ppv:>6.3f}")

    # 6. Scaling correlation on calibrated
    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    arch = {p["name"]: p["architecture"] for p in exp5["points"]}
    log_p = {p["name"]: p["log_params"] for p in exp5["points"]}
    dense = [(log_p[m], per_model_cal[m]["b"]) for m in per_model_cal
             if arch.get(m) == "dense" and per_model_cal[m].get("b") is not None]
    if len(dense) >= 5:
        lp = np.array([d[0] for d in dense])
        bs = np.array([d[1] for d in dense])
        rho, p = stats.spearmanr(lp, bs)
        print(f"\n  Calibrated scaling (dense): rho={rho:+.3f} (p={p:.4f})")
    else:
        rho, p = None, None

    out = {
        "n_calibration_items": len(matched),
        "calibration_curve": {f"{g:.1f}": float(iso.predict([g])[0])
                              for g in GRID},
        "per_model_calibrated": per_model_cal,
        "headline_calibrated": {
            "n_qualifying": qualifying,
            "n_disjoint_CIs": disjoint,
            "n_uncalibrated": uncal_pairs,
            "change": disjoint - uncal_pairs,
        },
        "calibrated_thresholds": cal_thresh,
        "calibrated_scaling": {
            "rho": float(rho) if rho is not None else None,
            "p": float(p) if p is not None else None,
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

