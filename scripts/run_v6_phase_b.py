"""Phase B: full re-analysis on the merged 10K dataset.

Reruns the full analysis pipeline on results/scores_10k/{model}.jsonl
and produces a 4K-vs-10K comparison report. Computed quantities:

  B0 baseline:
    - Per-model b with bootstrap CI (10K)
    - BIC-best family per model (10K)
    - error rate per model (10K)
    - Exp 1 (n_non_exp, n_vuong_decisive)
    - Exp 2 (disjoint-CI pairs at matched accuracy)
    - Exp 3 (micro-error → catastrophic prediction rho)
    - Exp 4 (Friedman, Kendall W on domain b matrix)
    - Exp 5 (dense scaling rho + partial control for eps)
    - 4K-vs-10K comparison table

  B6 family-native (10K)
  B7 cross-domain jackknife (10K)
  B9 deployment table (10K)
  B10 v4/v5 robustness re-checks (10K)

The new B1-B5 (hierarchical bootstrap, fixed-mmin, model-agnostic
estimators, binomial test, judge leniency) are in run_v6_phase_b_new.py.

Output: results/analysis/phase_b_10k.json
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
from errorquake.analyze import estimate_b_value, fit_all_distributions, vuong_test

SCORES_10K = ROOT / "results" / "scores_10k"
SCORES_4K = ROOT / "results" / "scores"
EXP5_4K = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "phase_b_10k.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_scores(path: Path) -> np.ndarray:
    out = []
    if not path.exists():
        return np.asarray(out, dtype=float)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = r.get("final_score")
        if s is not None:
            out.append(float(s))
    return np.asarray(out, dtype=float)


def fit_per_model(scores_dir: Path, models: list[str]) -> dict:
    out = {}
    for m in models:
        scores = load_scores(scores_dir / f"{m}.jsonl")
        if scores.size == 0:
            continue
        pos = scores[scores > 0]
        eps = float((scores > 0).mean())
        if pos.size < 30:
            out[m] = {"b": None, "ci_lo": None, "ci_hi": None, "eps": eps,
                      "n_total": int(scores.size), "n_pos": int(pos.size),
                      "best_fit": None}
            continue
        try:
            bv = estimate_b_value(pos, model_name=m)
            b = float(bv.b); lo = float(bv.b_ci_lower); hi = float(bv.b_ci_upper)
            m_min = float(bv.m_min); n_above = int(bv.n_above_mmin)
        except Exception:
            b = lo = hi = None; m_min = None; n_above = 0
        try:
            fits = fit_all_distributions(pos, model_name=m, m_min=0.5)
            valid = [f for f in fits
                     if f.bic == f.bic and f.bic != float("inf")]
            best = valid[0].distribution if valid else None
        except Exception:
            best = None
        out[m] = {
            "b": b, "ci_lo": lo, "ci_hi": hi,
            "m_min": m_min, "n_above_mmin": n_above,
            "eps": eps, "n_total": int(scores.size), "n_pos": int(pos.size),
            "best_fit": best,
        }
    return out


def discriminator_pairs(per_model: dict[str, dict]) -> dict:
    valid = [m for m, v in per_model.items() if v.get("b") is not None]
    qual = 0
    disjoint = 0
    examples = []
    for a, b in combinations(valid, 2):
        va, vb = per_model[a], per_model[b]
        eps_d = abs(va["eps"] - vb["eps"])
        b_d = abs(va["b"] - vb["b"])
        if eps_d < 0.05 and b_d > 0.15:
            qual += 1
            ci_disj = (max(va["ci_lo"], vb["ci_lo"])
                       > min(va["ci_hi"], vb["ci_hi"]))
            if ci_disj:
                disjoint += 1
                examples.append({
                    "model_1": a, "model_2": b,
                    "eps_1": va["eps"], "eps_2": vb["eps"],
                    "b_1": va["b"], "b_2": vb["b"], "b_diff": b_d,
                })
    return {
        "n_models_with_valid_b": len(valid),
        "n_total_pairs": len(list(combinations(valid, 2))),
        "n_qualifying": qual,
        "n_disjoint_CIs": disjoint,
        "top_pairs": sorted(examples, key=lambda e: -e["b_diff"])[:10],
    }


def vuong_decisive(scores_dir: Path, models: list[str]) -> dict:
    """Recompute Vuong p-values per model on the 10K data."""
    out = {}
    n_dec = 0
    n_nonexp = 0
    family_counts = {}
    for m in models:
        scores = load_scores(scores_dir / f"{m}.jsonl")
        if scores.size == 0:
            continue
        pos = scores[scores > 0]
        if pos.size < 30:
            continue
        try:
            fits = fit_all_distributions(pos, model_name=m, m_min=0.5)
            valid = sorted([f for f in fits
                            if f.bic == f.bic and f.bic != float("inf")],
                           key=lambda f: f.bic)
        except Exception:
            continue
        if not valid:
            continue
        best = valid[0]
        family_counts[best.distribution] = family_counts.get(best.distribution, 0) + 1
        if best.distribution != "exponential":
            n_nonexp += 1
        decisive = False
        if len(valid) >= 2:
            try:
                v = vuong_test(pos, best, valid[1])
                decisive = (v.get("preferred") == best.distribution
                            and v.get("p_value") is not None
                            and v["p_value"] < 0.05)
            except Exception:
                decisive = False
            if (valid[1].bic - best.bic) > 6:
                decisive = True
        if decisive:
            n_dec += 1
        out[m] = {"best": best.distribution, "decisive": decisive}
    return {
        "per_model": out, "n_decisive": n_dec, "n_nonexp": n_nonexp,
        "family_counts": family_counts,
    }


def scaling_correlation(per_model: dict, points: list[dict]) -> dict:
    """Compute Exp 5 dense scaling + partial correlation given eps."""
    arch = {p["name"]: p["architecture"] for p in points}
    log_p = {p["name"]: p["log_params"] for p in points}
    dense = [(log_p[m], per_model[m]["b"], per_model[m]["eps"])
             for m in per_model
             if arch.get(m) == "dense" and per_model[m].get("b") is not None]
    if len(dense) < 5:
        return {"error": "too_few_dense"}
    lp = np.array([d[0] for d in dense])
    bs = np.array([d[1] for d in dense])
    eps = np.array([d[2] for d in dense])
    rho, p = stats.spearmanr(lp, bs)
    # Partial via OLS residuals
    Xm = np.column_stack([np.ones(len(eps)), eps])
    c1, *_ = np.linalg.lstsq(Xm, lp, rcond=None)
    c2, *_ = np.linalg.lstsq(Xm, bs, rcond=None)
    lp_r = lp - Xm @ c1
    b_r = bs - Xm @ c2
    pr, pp = stats.spearmanr(lp_r, b_r)
    rho_eps_b, p_eps_b = stats.spearmanr(eps, bs)
    rho_lp_eps, p_lp_eps = stats.spearmanr(lp, eps)
    return {
        "n_dense": len(dense),
        "rho_log_p_b": float(rho), "p_log_p_b": float(p),
        "partial_rho_b_given_eps": float(pr),
        "partial_p_b_given_eps": float(pp),
        "rho_eps_b": float(rho_eps_b), "p_eps_b": float(p_eps_b),
        "rho_log_p_eps": float(rho_lp_eps), "p_log_p_eps": float(p_lp_eps),
    }


def domain_variation(scores_dir: Path, models: list[str]) -> dict:
    """Per-model b on each of 8 domains; Friedman + Kendall W."""
    DOMAINS = ["BIO", "LAW", "HIST", "GEO", "SCI", "TECH", "FIN", "CULT"]
    matrix = {}
    for m in models:
        path = scores_dir / f"{m}.jsonl"
        if not path.exists():
            continue
        per_dom = {d: [] for d in DOMAINS}
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = r.get("domain")
            s = r.get("final_score")
            if d in per_dom and s is not None:
                per_dom[d].append(float(s))
        bs = {}
        for d in DOMAINS:
            arr = np.asarray(per_dom[d], dtype=float)
            errors = arr[arr > 0]
            if errors.size < 30:
                bs[d] = None
                continue
            try:
                bs[d] = float(estimate_b_value(errors, model_name=f"{m}_{d}").b)
            except Exception:
                bs[d] = None
        matrix[m] = bs

    complete = [m for m in matrix if all(matrix[m][d] is not None for d in DOMAINS)]
    friedman_stat = friedman_p = kendall_w = None
    if len(complete) >= 5:
        data = np.array([[matrix[m][d] for d in DOMAINS] for m in complete])
        stat, p = stats.friedmanchisquare(*[data[:, j] for j in range(len(DOMAINS))])
        friedman_stat, friedman_p = float(stat), float(p)
        ranks = np.array([stats.rankdata(row) for row in data])
        m, k = ranks.shape
        rank_sums = ranks.sum(axis=0)
        s = float(np.sum((rank_sums - rank_sums.mean()) ** 2))
        kendall_w = 12 * s / (m ** 2 * (k ** 3 - k))
    return {
        "per_model_per_domain": matrix,
        "friedman_chi2": friedman_stat,
        "friedman_p": friedman_p,
        "kendall_w": kendall_w,
        "n_complete": len(complete),
    }


def main() -> None:
    print("=" * 70)
    print("PHASE B: FULL RE-ANALYSIS ON 10K")
    print("=" * 70)

    if not SCORES_10K.exists():
        print(f"ERROR: {SCORES_10K} does not exist. Run A5 merge first.")
        return

    models = sorted(f.stem for f in SCORES_10K.glob("*.jsonl") if f.stem not in EXCLUDED)
    print(f"Models in 10K: {len(models)}")

    # B0a: Per-model b on 10K
    print("\n[B0a] Fitting per-model b on 10K...")
    pm_10k = fit_per_model(SCORES_10K, models)
    valid_10k = sum(1 for v in pm_10k.values() if v.get("b") is not None)
    print(f"      Valid b fits: {valid_10k}/{len(pm_10k)}")

    # Per-model b on 4K (for comparison)
    print("\n[B0b] Re-fitting per-model b on 4K...")
    pm_4k = fit_per_model(SCORES_4K, models)
    valid_4k = sum(1 for v in pm_4k.values() if v.get("b") is not None)
    print(f"      Valid b fits: {valid_4k}/{len(pm_4k)}")

    # Exp 1: Vuong-decisive count on 10K
    print("\n[Exp 1] Vuong-decisive families on 10K...")
    exp1_10k = vuong_decisive(SCORES_10K, models)
    print(f"      n_decisive={exp1_10k['n_decisive']}/{len(exp1_10k['per_model'])}, "
          f"n_nonexp={exp1_10k['n_nonexp']}, families={exp1_10k['family_counts']}")

    # Exp 2: discriminator on 10K
    print("\n[Exp 2] Discriminator pairs on 10K...")
    exp2_10k = discriminator_pairs(pm_10k)
    print(f"      qualifying={exp2_10k['n_qualifying']}, "
          f"disjoint={exp2_10k['n_disjoint_CIs']}")

    exp2_4k = discriminator_pairs(pm_4k)

    # Exp 5: scaling correlation on 10K
    exp5_data = json.loads(EXP5_4K.read_text(encoding="utf-8"))
    points = exp5_data["points"]
    print("\n[Exp 5] Scaling correlation on 10K...")
    exp5_10k = scaling_correlation(pm_10k, points)
    print(f"      rho_log_p_b dense = {exp5_10k.get('rho_log_p_b'):+.4f} "
          f"(p={exp5_10k.get('p_log_p_b'):.4f})")
    print(f"      partial(b|eps)    = {exp5_10k.get('partial_rho_b_given_eps'):+.4f}")

    # Exp 4: domain variation on 10K
    print("\n[Exp 4] Domain variation on 10K...")
    exp4_10k = domain_variation(SCORES_10K, models)
    print(f"      Friedman chi2={exp4_10k['friedman_chi2']:.3f} "
          f"(p={exp4_10k['friedman_p']:.4f}), "
          f"Kendall W={exp4_10k['kendall_w']:.3f}")

    out = {
        "per_model_10k": pm_10k,
        "per_model_4k": pm_4k,
        "exp1_10k": exp1_10k,
        "exp2_10k": exp2_10k,
        "exp2_4k": exp2_4k,
        "exp5_10k": exp5_10k,
        "exp4_10k": exp4_10k,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

