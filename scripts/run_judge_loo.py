"""Task 1.1: Leave-one-judge-out ablation on the Exp 5 headline.

For each judge J in the 8-model pool (plus any others that appear as
primary/secondary), remove all judgments it produced, recompute the
per-record final_score from the remaining judges, refit b for all 21
models, then recompute the dense-model Spearman(log_p, b).

Final-score reconstruction rule (per v3 Task 1.1):
  if J was primary    and secondary survived: new = secondary
  if J was secondary  and primary survived:   new = primary
  if both survived (J absent): new = mean(primary, secondary) if |diff|<=1 else primary
  if neither survived: drop the record
Note: we don't have tiebreaker records separately; the stored
'final_score' already folded tie-breaking in, but for LOO we rebuild
from primary+secondary only since that's the material we can recover.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value

SCORES = ROOT / "results" / "scores"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "judge_loo_ablation.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def compute_score_excluding(record: dict, excluded_judge: str) -> float | None:
    pj = record.get("primary_judge")
    sj = record.get("secondary_judge")
    ps = record.get("primary_score")
    ss = record.get("secondary_score")
    pj_ok = pj != excluded_judge and ps is not None
    sj_ok = sj != excluded_judge and ss is not None
    if pj_ok and sj_ok:
        return (float(ps) + float(ss)) / 2.0
    if pj_ok:
        return float(ps)
    if sj_ok:
        return float(ss)
    return None


def load_records(stem: str) -> list[dict]:
    out = []
    for line in open(SCORES / f"{stem}.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def fit_b_on_scores(scores: list[float], name: str) -> tuple[float | None, float]:
    """Return (b_value, error_rate)."""
    arr = np.asarray([s for s in scores if s is not None], dtype=float)
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


def _residualise(y, x):
    Xm = np.column_stack([np.ones(len(x)), x])
    c, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return y - Xm @ c


def main() -> None:
    print("=" * 70)
    print("JUDGE LEAVE-ONE-OUT ABLATION (v5 Task 1.1)")
    print("=" * 70)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    dense = [p["name"] for p in exp5["points"] if p["architecture"] == "dense"]
    all_models = [p["name"] for p in exp5["points"]]
    log_p_by = {p["name"]: p["log_params"] for p in exp5["points"]}

    # Preload all records per model
    print("Loading score records...")
    model_recs = {m: load_records(m) for m in all_models if m not in EXCLUDED}

    # Identify judges (pool)
    judges = set()
    for recs in model_recs.values():
        for r in recs:
            if r.get("primary_judge"):
                judges.add(r["primary_judge"])
            if r.get("secondary_judge"):
                judges.add(r["secondary_judge"])
    judges = sorted(judges)
    print(f"Found {len(judges)} distinct judges")

    baseline_rho, baseline_p = stats.spearmanr(
        [log_p_by[m] for m in dense if m in model_recs],
        [exp5_p_b := {p["name"]: p["b_value"] for p in exp5["points"]}[m]
         for m in dense if m in model_recs],
    )
    print(f"Baseline dense rho = {baseline_rho:+.3f} (p={baseline_p:.4f})")

    results = []
    for j in judges:
        # Recompute b and eps for each model excluding judge j
        b_by_model = {}
        eps_by_model = {}
        n_affected = 0
        for m in all_models:
            if m not in model_recs:
                continue
            scores = []
            for r in model_recs[m]:
                if r.get("primary_judge") == j or r.get("secondary_judge") == j:
                    n_affected += 1
                s = compute_score_excluding(r, j)
                if s is not None:
                    scores.append(s)
            b, eps = fit_b_on_scores(scores, f"{m}_loo_{j}")
            b_by_model[m] = b
            eps_by_model[m] = eps

        # Headline on dense (only models with valid b)
        valid = [(log_p_by[m], b_by_model[m], eps_by_model[m]) for m in dense
                 if b_by_model.get(m) is not None]
        if len(valid) >= 5:
            lp = np.array([v[0] for v in valid])
            bs = np.array([v[1] for v in valid])
            eps_arr = np.array([v[2] for v in valid])
            rho, p = stats.spearmanr(lp, bs)
            # Partial rho(log_p, b | eps)
            lp_resid = _residualise(lp, eps_arr)
            b_resid = _residualise(bs, eps_arr)
            partial_rho, partial_p = stats.spearmanr(lp_resid, b_resid)
        else:
            rho, p = float("nan"), float("nan")
            partial_rho, partial_p = float("nan"), float("nan")
        short = j.split("/")[-1] if "/" in j else j
        results.append({
            "judge_removed": j,
            "judge_short": short,
            "n_records_affected": n_affected,
            "n_dense_with_valid_b": len(valid),
            "rho_dense": float(rho) if not np.isnan(rho) else None,
            "p_dense": float(p) if not np.isnan(p) else None,
            "partial_rho_given_eps": float(partial_rho) if not np.isnan(partial_rho) else None,
            "partial_p_given_eps": float(partial_p) if not np.isnan(partial_p) else None,
            "sign_preserved": bool(rho < 0) if not np.isnan(rho) else False,
            "p_below_05": bool(p < 0.05) if not np.isnan(p) else False,
        })
        print(f"  drop {short:<38} n_aff={n_affected:>6}  "
              f"rho={rho:+.3f} (p={p:.3f})  "
              f"partial={partial_rho:+.3f} (p={partial_p:.3f})")

    n_sig = sum(1 for r in results if r["p_below_05"])
    n_sign = sum(1 for r in results if r["sign_preserved"])
    print()
    print(f"Sign preserved: {n_sign}/{len(results)}")
    print(f"p<0.05 preserved: {n_sig}/{len(results)}")

    out = {
        "baseline_rho": float(baseline_rho),
        "baseline_p": float(baseline_p),
        "n_judges_tested": len(judges),
        "n_sign_preserved": int(n_sign),
        "n_p_below_05": int(n_sig),
        "per_judge": results,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
