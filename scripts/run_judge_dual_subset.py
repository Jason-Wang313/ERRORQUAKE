"""Task 1.1/1.2 redux: LOO ablation + reweighting on the dense subset
with genuine dual-judge coverage (>=80% of records have both scores).

The raw v4 and v5 Task 1.1 results include 4-5 dense models where the
secondary judge call effectively failed (phi-3.5-mini, qwen2.5-7b,
eurollm-9b, gemma-3-4b, llama-3.2-3b), which artificially inflates
the judge-dependence of the headline because dropping a single judge
removes those models entirely. This script re-runs on models where
>=80% of records have BOTH primary and secondary scores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value

SCORES = ROOT / "results" / "scores"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "judge_dual_subset.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
DUAL_THRESHOLD = 0.80  # fraction of records with both scores


def load_records(stem: str) -> list[dict]:
    return [json.loads(l) for l in open(SCORES / f"{stem}.jsonl",
            encoding="utf-8") if l.strip()]


def dual_coverage(recs: list[dict]) -> float:
    if not recs:
        return 0.0
    both = sum(1 for r in recs if r.get("primary_score") is not None
               and r.get("secondary_score") is not None)
    return both / len(recs)


def fit_b_eps(scores, name):
    arr = np.array([s for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return None, float("nan")
    pos = arr[arr > 0]
    eps = float((arr > 0).mean())
    if pos.size < 30:
        return None, eps
    try:
        return float(estimate_b_value(pos, model_name=name).b), eps
    except Exception:
        return None, eps


def residualise(y, x):
    Xm = np.column_stack([np.ones(len(x)), x])
    c, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return y - Xm @ c


def main() -> None:
    print("=" * 70)
    print("JUDGE ABLATION on DUAL-JUDGE-COVERED subset")
    print("=" * 70)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    log_p_by = {p["name"]: p["log_params"] for p in exp5["points"]}
    arch_by = {p["name"]: p["architecture"] for p in exp5["points"]}
    dense_all = [p["name"] for p in exp5["points"] if p["architecture"] == "dense"]

    model_recs = {}
    coverage = {}
    for m in dense_all:
        if m in EXCLUDED:
            continue
        recs = load_records(m)
        model_recs[m] = recs
        coverage[m] = dual_coverage(recs)

    print()
    print(f"{'model':<28} dual coverage")
    print("-" * 50)
    for m, c in sorted(coverage.items(), key=lambda kv: -kv[1]):
        mark = "✓" if c >= DUAL_THRESHOLD else "✗"
        print(f"  {m:<28} {c*100:>6.1f}%  {mark}")

    dense_dual = [m for m, c in coverage.items() if c >= DUAL_THRESHOLD]
    dense_drop = [m for m, c in coverage.items() if c < DUAL_THRESHOLD]
    print()
    print(f"Dense with >={int(DUAL_THRESHOLD*100)}% dual coverage: {len(dense_dual)}")
    print(f"Dense dropped:                      {len(dense_drop)} -> {dense_drop}")

    # Compute b on this subset. score_fn receives (record, *extra) and returns
    # a float or None. This lets the baseline use the stored final_score
    # while LOO/reweight schemes can re-aggregate from primary/secondary.
    def compute_b_on_subset(models, score_fn):
        b_by = {}
        eps_by = {}
        for m in models:
            if m not in model_recs:
                continue
            scores = []
            for r in model_recs[m]:
                s = score_fn(r)
                scores.append(s)
            b, eps = fit_b_eps(scores, m)
            b_by[m] = b
            eps_by[m] = eps
        return b_by, eps_by

    def headline_on(models, b_by, eps_by):
        valid = [(log_p_by[m], b_by[m], eps_by[m]) for m in models
                 if b_by.get(m) is not None]
        if len(valid) < 5:
            return None
        lp = np.array([v[0] for v in valid])
        bs = np.array([v[1] for v in valid])
        eps = np.array([v[2] for v in valid])
        rho, p = stats.spearmanr(lp, bs)
        lp_r = residualise(lp, eps)
        b_r = residualise(bs, eps)
        prho, pp = stats.spearmanr(lp_r, b_r)
        return {
            "n": len(valid),
            "rho": float(rho),
            "p": float(p),
            "partial_rho_given_eps": float(prho),
            "partial_p_given_eps": float(pp),
        }

    def stored_final(r):
        return r.get("final_score")

    # Baseline on dual-coverage subset — use the stored final_score
    # (includes tiebreaking), matching Exp 5 published b-values.
    b_by, eps_by = compute_b_on_subset(dense_dual, stored_final)
    baseline = headline_on(dense_dual, b_by, eps_by)
    print()
    print(f"Baseline on dual-coverage subset (n={baseline['n']}):")
    print(f"  rho = {baseline['rho']:+.3f} (p={baseline['p']:.4f})")
    print(f"  partial(|eps) = {baseline['partial_rho_given_eps']:+.3f} "
          f"(p={baseline['partial_p_given_eps']:.4f})")

    # Gather judges
    judges = set()
    for m in dense_dual:
        for r in model_recs[m]:
            if r.get("primary_judge"):
                judges.add(r["primary_judge"])
            if r.get("secondary_judge"):
                judges.add(r["secondary_judge"])
    judges = sorted(judges)

    # LOO on dual-coverage subset
    print()
    print(f"LOO ablation on dual-coverage dense subset (n={baseline['n']}):")
    loo_results = []
    for j in judges:
        # LOO policy: if judge J wasn't involved in a record, keep its stored
        # final_score (so tiebreaking is preserved for unaffected records).
        # If J was involved, rebuild from the surviving judge(s) only.
        def score_fn(r, exclude_j=j):
            pj = r.get("primary_judge")
            sj = r.get("secondary_judge")
            if pj != exclude_j and sj != exclude_j:
                return r.get("final_score")
            pp = r.get("primary_score") if pj != exclude_j else None
            ss = r.get("secondary_score") if sj != exclude_j else None
            if pp is not None and ss is not None:
                return (float(pp) + float(ss)) / 2.0
            if pp is not None:
                return float(pp)
            if ss is not None:
                return float(ss)
            return None
        b_by, eps_by = compute_b_on_subset(dense_dual, score_fn)
        r = headline_on(dense_dual, b_by, eps_by)
        short = j.split("/")[-1]
        if r:
            loo_results.append({"judge": short, "full": j, **r,
                                "sign_preserved": r["rho"] < 0,
                                "p_below_05": r["p"] < 0.05})
            print(f"  drop {short:<36} n={r['n']}  rho={r['rho']:+.3f} (p={r['p']:.3f})")
        else:
            loo_results.append({"judge": short, "full": j, "error": "insufficient"})

    n_sign_ok = sum(1 for r in loo_results if r.get("sign_preserved"))
    n_p_ok = sum(1 for r in loo_results if r.get("p_below_05"))
    print(f"\nSign preserved (dual-subset LOO): {n_sign_ok}/{len(loo_results)}")
    print(f"p<0.05 preserved: {n_p_ok}/{len(loo_results)}")

    # Reweighting schemes on dual-coverage subset
    print()
    print("Reweighting schemes on dual-coverage dense subset:")
    reweight_results = {}
    for scheme_name, fn in [
        ("primary_only", lambda r: float(r["primary_score"])
             if r.get("primary_score") is not None else None),
        ("secondary_only", lambda r: float(r["secondary_score"])
             if r.get("secondary_score") is not None else None),
        ("max", lambda r: max(float(r["primary_score"]), float(r["secondary_score"]))
             if (r.get("primary_score") is not None and r.get("secondary_score") is not None)
             else (r.get("primary_score") if r.get("primary_score") is not None
                   else r.get("secondary_score"))),
        ("min", lambda r: min(float(r["primary_score"]), float(r["secondary_score"]))
             if (r.get("primary_score") is not None and r.get("secondary_score") is not None)
             else (r.get("primary_score") if r.get("primary_score") is not None
                   else r.get("secondary_score"))),
    ]:
        b_by, eps_by = compute_b_on_subset(dense_dual, fn)
        r = headline_on(dense_dual, b_by, eps_by)
        if r:
            reweight_results[scheme_name] = r
            print(f"  {scheme_name:<16} n={r['n']}  rho={r['rho']:+.3f} (p={r['p']:.3f})  "
                  f"partial={r['partial_rho_given_eps']:+.3f}")

    out = {
        "dual_coverage_threshold": DUAL_THRESHOLD,
        "dense_with_dual_coverage": dense_dual,
        "dense_dropped_for_low_coverage": dense_drop,
        "baseline_on_dual_subset": baseline,
        "loo_on_dual_subset": loo_results,
        "reweight_on_dual_subset": reweight_results,
        "summary": {
            "n_loo_sign_preserved": int(n_sign_ok),
            "n_loo_p_below_05": int(n_p_ok),
            "n_loo_total": len(loo_results),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
