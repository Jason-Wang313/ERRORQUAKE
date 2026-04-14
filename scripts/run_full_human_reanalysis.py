"""Full human-consensus re-analysis on 186K items.

1. Load 3-rater scores, compute median consensus per item
2. Replace ALL judge final_scores with human consensus
3. Refit b for all 21 models
4. Recount disjoint-CI pairs (the new definitive headline)
5. Recompute all key metrics
6. Compare with judge-scored baseline

Output: results/analysis/v10_full_human.json
        results/scores_10k_human/{model}.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value

RATED = ROOT / "data" / "human_audit" / "full_rating_kit" / "rated_items_full.csv"
KEY = ROOT / "data" / "human_audit" / "full_rating_kit" / "rating_key.json"
SCORES_10K = ROOT / "results" / "scores_10k"
SCORES_HUMAN = ROOT / "results" / "scores_10k_human"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
PB = ROOT / "results" / "analysis" / "phase_b_10k.json"
OUT = ROOT / "results" / "analysis" / "v10_full_human.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def snap(x):
    return float(GRID[int(np.argmin(np.abs(GRID - float(x))))])


def icc_2way(matrix):
    n, k = matrix.shape
    grand = matrix.mean()
    SSR = k * np.sum((matrix.mean(axis=1) - grand) ** 2)
    SSC = n * np.sum((matrix.mean(axis=0) - grand) ** 2)
    SST = np.sum((matrix - grand) ** 2)
    SSE = SST - SSR - SSC
    MSR = SSR / max(n - 1, 1)
    MSC = SSC / max(k - 1, 1)
    MSE = SSE / max((n - 1) * (k - 1), 1)
    d1 = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    icc21 = (MSR - MSE) / d1 if d1 > 0 else 0
    dk = MSR + (MSC - MSE) / n
    icc2k = (MSR - MSE) / dk if dk > 0 else 0
    return {"icc_2_1": float(icc21), "icc_2_k": float(icc2k)}


def main() -> None:
    print("=" * 70)
    print("FULL HUMAN-CONSENSUS RE-ANALYSIS (186K items)")
    print("=" * 70)

    # 1. Load ratings
    ratings = {}
    scores_A, scores_B, scores_C = [], [], []
    with open(RATED, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                a = float(row["score_rater_A"])
                b = float(row["score_rater_B"])
                c = float(row["score_rater_C"])
            except (ValueError, KeyError):
                continue
            consensus = snap(float(np.median([a, b, c])))
            ratings[row["rating_id"]] = consensus
            scores_A.append(a)
            scores_B.append(b)
            scores_C.append(c)
    print(f"Loaded {len(ratings)} rated items")

    # ICC
    matrix = np.column_stack([scores_A, scores_B, scores_C])
    icc = icc_2way(matrix)
    print(f"ICC(2,1) = {icc['icc_2_1']:.3f}")
    print(f"ICC(2,k=3) = {icc['icc_2_k']:.3f}")

    # 2. Map rating_id -> (model, query_id)
    key_data = json.loads(KEY.read_text(encoding="utf-8"))
    rid_to_mq = {k["rating_id"]: (k["model"], k["query_id"]) for k in key_data}

    # Build human consensus by (model, query_id)
    human_by_mq = {}
    for rid, consensus in ratings.items():
        if rid in rid_to_mq:
            m, qid = rid_to_mq[rid]
            human_by_mq[(m, qid)] = consensus

    print(f"Matched to model-query pairs: {len(human_by_mq)}")

    # 3. Replace judge scores with human consensus and write new score files
    SCORES_HUMAN.mkdir(parents=True, exist_ok=True)
    models = sorted(f.stem for f in SCORES_10K.glob("*.jsonl")
                    if f.stem not in EXCLUDED)

    per_model = {}
    for m in models:
        recs = [json.loads(l) for l in open(SCORES_10K / f"{m}.jsonl",
                encoding="utf-8") if l.strip()]
        human_recs = []
        replaced = 0
        for r in recs:
            r2 = dict(r)
            key = (m, r.get("query_id"))
            if key in human_by_mq:
                r2["final_score_judge"] = r2.get("final_score")
                r2["final_score"] = human_by_mq[key]
                replaced += 1
            human_recs.append(r2)
        with open(SCORES_HUMAN / f"{m}.jsonl", "w", encoding="utf-8") as f:
            for r in human_recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Fit b on human-consensus scores
        scores = np.array([r["final_score"] for r in human_recs
                           if r.get("final_score") is not None], dtype=float)
        pos = scores[scores > 0]
        eps = float((scores > 0).mean())
        try:
            bv = estimate_b_value(pos, model_name=f"{m}_human")
            per_model[m] = {
                "b": float(bv.b), "ci_lo": float(bv.b_ci_lower),
                "ci_hi": float(bv.b_ci_upper), "eps": eps,
                "n_pos": int(pos.size), "n_replaced": replaced,
            }
        except Exception:
            per_model[m] = {"b": None, "ci_lo": None, "ci_hi": None,
                            "eps": eps, "n_pos": int(pos.size),
                            "n_replaced": replaced}

    # 4. Count disjoint-CI pairs
    valid = [m for m in per_model if per_model[m].get("b") is not None]
    qualifying = 0
    disjoint = 0
    for a, b in combinations(valid, 2):
        va, vb = per_model[a], per_model[b]
        if abs(va["eps"] - vb["eps"]) >= 0.05:
            continue
        if abs(va["b"] - vb["b"]) <= 0.15:
            continue
        qualifying += 1
        if max(va["ci_lo"], vb["ci_lo"]) > min(va["ci_hi"], vb["ci_hi"]):
            disjoint += 1

    # Judge-scored baseline
    pb = json.loads(PB.read_text(encoding="utf-8"))
    judge_pairs = pb.get("exp2_10k", {}).get("n_disjoint_CIs", 31)

    print(f"\nHeadline (human-consensus scored):")
    print(f"  Models with valid b: {len(valid)}")
    print(f"  Qualifying pairs: {qualifying}")
    print(f"  Disjoint-CI pairs: {disjoint}")
    print(f"  Judge-scored baseline: {judge_pairs}")
    print(f"  Change: {disjoint - judge_pairs:+d}")

    # 5. Scaling correlation
    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    arch = {p["name"]: p["architecture"] for p in exp5["points"]}
    log_p = {p["name"]: p["log_params"] for p in exp5["points"]}

    dense = [(log_p[m], per_model[m]["b"], per_model[m]["eps"])
             for m in per_model
             if arch.get(m) == "dense" and per_model[m].get("b") is not None]
    if len(dense) >= 5:
        lp = np.array([d[0] for d in dense])
        bs = np.array([d[1] for d in dense])
        eps_arr = np.array([d[2] for d in dense])
        rho, p = stats.spearmanr(lp, bs)
        # Partial
        def resid(y, x):
            Xm = np.column_stack([np.ones(len(x)), x])
            c, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            return y - Xm @ c
        pr, pp = stats.spearmanr(resid(lp, eps_arr), resid(bs, eps_arr))
        rho_eb, p_eb = stats.spearmanr(eps_arr, bs)
        print(f"\nScaling (dense, human-scored):")
        print(f"  rho(log_p, b) = {rho:+.3f} (p={p:.4f})")
        print(f"  partial(b|eps) = {pr:+.3f} (p={pp:.4f})")
        print(f"  rho(eps, b) = {rho_eb:+.3f} (p={p_eb:.4f})")
    else:
        rho = pr = pp = rho_eb = p_eb = None

    # 6. Per-model comparison table
    print(f"\n{'model':<28} {'b_judge':>8} {'b_human':>8} {'Δb':>8} {'eps_h':>7}")
    print("-" * 65)
    pm_judge = pb["per_model_10k"]
    for m in sorted(per_model.keys()):
        bj = pm_judge.get(m, {}).get("b")
        bh = per_model[m].get("b")
        delta = f"{bh - bj:+.3f}" if (bj and bh) else "N/A"
        bjs = f"{bj:.3f}" if bj else "N/A"
        bhs = f"{bh:.3f}" if bh else "N/A"
        print(f"{m:<28} {bjs:>8} {bhs:>8} {delta:>8} {per_model[m]['eps']:>7.3f}")

    out = {
        "n_items_rated": len(ratings),
        "icc": icc,
        "per_model_human": per_model,
        "headline_human": {
            "n_qualifying": qualifying,
            "n_disjoint_CIs": disjoint,
            "n_judge_baseline": judge_pairs,
            "change": disjoint - judge_pairs,
        },
        "scaling_human": {
            "rho_log_p_b": float(rho) if rho is not None else None,
            "p_log_p_b": float(p) if rho is not None else None,
            "partial_rho_b_eps": float(pr) if pr is not None else None,
            "partial_p": float(pp) if pp is not None else None,
            "rho_eps_b": float(rho_eb) if rho_eb is not None else None,
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

