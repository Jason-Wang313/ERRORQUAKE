"""Phase B new tasks (B1-B5) — the v6-prompt robustness suite.

These run on the merged 10K data. Each is independent and safe to
re-run.

  B1 hierarchical bootstrap: resample queries + simulate judge noise
     (swap primary/secondary with p=0.5; perturb single-judge by
     U(-0.5, +0.5)). Recount discriminator pairs.
  B2 fixed-mmin discriminator: refit b at fixed mmin in {1.5, 2.0,
     2.5, 3.0}; recount pairs.
  B3 model-agnostic tail-slope estimators: log-linear regression
     over {2.5, 3.0, 3.5, 4.0} bins; empirical tail ratio. Bootstrap
     CIs and pair counts.
  B4 binomial catastrophic-rate test: Fisher's exact on counts at
     M >= 3.0 for the matched-accuracy pair set; BH FDR correction.
  B5 judge family leniency: per-judge mean score, pairwise
     Kruskal-Wallis on per-target scores.

Output: results/analysis/phase_b_new_10k.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import _estimate_b, _quantize_to_grid, estimate_b_value

SCORES_10K = ROOT / "results" / "scores_10k"
OUT = ROOT / "results" / "analysis" / "phase_b_new_10k.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def load_records(stem: str) -> list[dict]:
    path = SCORES_10K / f"{stem}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def all_models() -> list[str]:
    return sorted(f.stem for f in SCORES_10K.glob("*.jsonl") if f.stem not in EXCLUDED)


# ----------------------------------------------------------------------
# B1: hierarchical bootstrap
# ----------------------------------------------------------------------

def b1_hierarchical_bootstrap(n_iter: int = 200) -> dict:
    print("\n[B1] Hierarchical bootstrap (judge-noise-aware)...")
    rng = np.random.default_rng(42)
    models = all_models()
    recs_by_model = {m: load_records(m) for m in models}

    pair_counts = []
    b_distributions = defaultdict(list)
    eps_distributions = defaultdict(list)

    for it in range(n_iter):
        per_model_b = {}
        per_model_ci_lo = {}
        per_model_ci_hi = {}
        per_model_eps = {}
        for m in models:
            recs = recs_by_model[m]
            n = len(recs)
            if n == 0:
                continue
            # Resample queries with replacement
            idx = rng.integers(0, n, size=n)
            sample = [recs[i] for i in idx]
            scores = []
            for r in sample:
                ps = r.get("primary_score")
                ss = r.get("secondary_score")
                fs = r.get("final_score")
                # Judge-noise model: swap primary/secondary with p=0.5
                # if both exist; otherwise perturb single-judge by U(-0.5, +0.5)
                if ps is not None and ss is not None:
                    if rng.random() < 0.5:
                        ps, ss = ss, ps
                    # Re-resolve: mean if |diff|<=1, else fall back to primary
                    if abs(ps - ss) <= 1.0:
                        scores.append((ps + ss) / 2.0)
                    else:
                        scores.append(ps)
                elif fs is not None:
                    noise = rng.uniform(-0.5, 0.5)
                    s = fs + noise
                    s = max(0.0, min(4.0, s))
                    s = round(s * 2) / 2  # snap to 0.5 grid
                    scores.append(s)
            arr = np.asarray(scores, dtype=float)
            if arr.size == 0:
                continue
            pos = arr[arr > 0]
            eps = float((arr > 0).mean())
            per_model_eps[m] = eps
            if pos.size < 30:
                continue
            try:
                bv = estimate_b_value(pos, model_name=f"{m}_boot{it}",
                                      n_bootstrap=200)
                per_model_b[m] = float(bv.b)
                per_model_ci_lo[m] = float(bv.b_ci_lower)
                per_model_ci_hi[m] = float(bv.b_ci_upper)
                b_distributions[m].append(float(bv.b))
                eps_distributions[m].append(eps)
            except Exception:
                continue

        # Count pairs
        valid = [m for m in per_model_b if m in per_model_eps]
        pairs = 0
        for a, b in combinations(valid, 2):
            if abs(per_model_eps[a] - per_model_eps[b]) >= 0.05:
                continue
            if abs(per_model_b[a] - per_model_b[b]) <= 0.15:
                continue
            if max(per_model_ci_lo[a], per_model_ci_lo[b]) > min(per_model_ci_hi[a], per_model_ci_hi[b]):
                pairs += 1
        pair_counts.append(pairs)
        if (it + 1) % 25 == 0:
            print(f"      iter {it+1}/{n_iter}: median_pairs_so_far="
                  f"{int(np.median(pair_counts))}", flush=True)

    pair_counts = np.array(pair_counts)
    return {
        "n_iter": n_iter,
        "pair_count_mean": float(pair_counts.mean()),
        "pair_count_median": float(np.median(pair_counts)),
        "pair_count_p2_5": float(np.percentile(pair_counts, 2.5)),
        "pair_count_p97_5": float(np.percentile(pair_counts, 97.5)),
        "pair_count_min": int(pair_counts.min()),
        "pair_count_max": int(pair_counts.max()),
        "per_model_b_distributions": {
            m: {"mean": float(np.mean(v)), "p2_5": float(np.percentile(v, 2.5)),
                "p97_5": float(np.percentile(v, 97.5))}
            for m, v in b_distributions.items() if len(v) >= 10
        },
    }


# ----------------------------------------------------------------------
# B2: fixed-mmin discriminator
# ----------------------------------------------------------------------

def b2_fixed_mmin() -> dict:
    print("\n[B2] Fixed-mmin discriminator pair counts...")
    models = all_models()
    recs = {m: load_records(m) for m in models}
    rows = {}

    def fit_at(scores: np.ndarray, m_min: float):
        pos = scores[scores >= m_min - 1e-9]
        if pos.size < 30:
            return None
        try:
            b = float(_estimate_b(scores, m_min))
            # Bootstrap CI
            rng = np.random.default_rng(7)
            samples = rng.choice(pos, size=(500, pos.size), replace=True)
            boot_b = np.array([_estimate_b(s, m_min) for s in samples])
            boot_b = boot_b[np.isfinite(boot_b)]
            if boot_b.size == 0:
                return None
            ci_lo, ci_hi = np.percentile(boot_b, [2.5, 97.5])
            return {"b": b, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                    "n_above": int(pos.size)}
        except Exception:
            return None

    for m_min in (1.5, 2.0, 2.5, 3.0):
        per_model = {}
        for m in models:
            scores = []
            for r in recs[m]:
                s = r.get("final_score")
                if s is not None:
                    scores.append(float(s))
            arr = _quantize_to_grid(np.asarray(scores, dtype=float))
            arr_pos = arr[arr > 0]
            eps = float((np.asarray(scores) > 0).mean()) if scores else 0.0
            fit = fit_at(arr_pos, m_min)
            if fit is None:
                continue
            per_model[m] = {**fit, "eps": eps}
        # Count pairs
        valid = list(per_model.keys())
        qual = 0
        disjoint = 0
        for a, b in combinations(valid, 2):
            va, vb = per_model[a], per_model[b]
            if abs(va["eps"] - vb["eps"]) >= 0.05:
                continue
            if abs(va["b"] - vb["b"]) <= 0.15:
                continue
            qual += 1
            if max(va["ci_lo"], vb["ci_lo"]) > min(va["ci_hi"], vb["ci_hi"]):
                disjoint += 1
        rows[f"mmin_{m_min}"] = {
            "m_min": m_min,
            "n_models_with_valid_b": len(valid),
            "n_qualifying": qual,
            "n_disjoint_CIs": disjoint,
        }
        print(f"      mmin={m_min}: n_models={len(valid)}, qual={qual}, disjoint={disjoint}")

    return rows


# ----------------------------------------------------------------------
# B3: model-agnostic tail-slope estimators
# ----------------------------------------------------------------------

def b3_model_agnostic() -> dict:
    print("\n[B3] Model-agnostic tail-slope estimators...")
    models = all_models()
    recs = {m: load_records(m) for m in models}

    per_model = {}
    for m in models:
        scores = [r.get("final_score") for r in recs[m] if r.get("final_score") is not None]
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            continue
        pos = arr[arr > 0]
        if pos.size < 30:
            continue
        eps = float((arr > 0).mean())
        # B3a: Log-linear regression over upper bins
        bins = np.array([2.5, 3.0, 3.5, 4.0])
        counts = np.array([float((pos >= b - 1e-9).sum()) for b in bins])
        positives = counts > 0
        if positives.sum() >= 2:
            xs = bins[positives]
            ys = np.log10(counts[positives])
            slope, intercept = np.polyfit(xs, ys, 1)
            b_loglinear = float(-slope)
        else:
            b_loglinear = None
        # B3b: Empirical tail ratio
        n_ge1 = int((pos >= 1.0 - 1e-9).sum())
        n_ge3 = int((pos >= 3.0 - 1e-9).sum())
        tail_ratio = n_ge3 / max(n_ge1, 1) if n_ge1 > 0 else None
        per_model[m] = {
            "b_loglinear": b_loglinear,
            "tail_ratio": tail_ratio,
            "eps": eps,
            "n_pos": int(pos.size),
        }

    # Pair counts on the new estimators
    def count_pairs(values_key: str, threshold: float) -> int:
        valid = [m for m, v in per_model.items() if v.get(values_key) is not None]
        n = 0
        for a, b in combinations(valid, 2):
            va, vb = per_model[a], per_model[b]
            if abs(va["eps"] - vb["eps"]) >= 0.05:
                continue
            if abs(va[values_key] - vb[values_key]) > threshold:
                n += 1
        return n

    return {
        "per_model": per_model,
        "n_pairs_loglinear_b_diff_gt_0_15": count_pairs("b_loglinear", 0.15),
        "n_pairs_tail_ratio_diff_gt_0_005": count_pairs("tail_ratio", 0.005),
        "n_pairs_tail_ratio_diff_gt_0_01": count_pairs("tail_ratio", 0.01),
        "n_pairs_tail_ratio_diff_gt_0_02": count_pairs("tail_ratio", 0.02),
    }


# ----------------------------------------------------------------------
# B4: binomial catastrophic-rate test (Fisher's exact + BH-FDR)
# ----------------------------------------------------------------------

def b4_binomial_test() -> dict:
    print("\n[B4] Binomial catastrophic-rate test (Fisher's exact + BH-FDR)...")
    models = all_models()
    recs = {m: load_records(m) for m in models}

    per_model = {}
    for m in models:
        scores = [r.get("final_score") for r in recs[m] if r.get("final_score") is not None]
        arr = np.asarray(scores, dtype=float)
        n_total = int(arr.size)
        if n_total == 0:
            continue
        per_model[m] = {
            "n_total": n_total,
            "n_ge_2_5": int((arr >= 2.5 - 1e-9).sum()),
            "n_ge_3_0": int((arr >= 3.0 - 1e-9).sum()),
            "eps": float((arr > 0).mean()),
        }

    # Test all matched-accuracy pairs
    def fisher(pair: tuple[str, str], threshold_key: str) -> dict:
        a, b = pair
        ka, kb = per_model[a][threshold_key], per_model[b][threshold_key]
        na, nb = per_model[a]["n_total"], per_model[b]["n_total"]
        # Fisher exact on 2x2 (k_a, n_a-k_a, k_b, n_b-k_b)
        try:
            from scipy.stats import fisher_exact
            _, p = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")
            return {"k_a": ka, "n_a": na, "k_b": kb, "n_b": nb, "p": float(p)}
        except Exception:
            return {"k_a": ka, "n_a": na, "k_b": kb, "n_b": nb, "p": None}

    pairs = list(combinations(per_model.keys(), 2))
    results_3 = []
    results_25 = []
    for a, b in pairs:
        if abs(per_model[a]["eps"] - per_model[b]["eps"]) >= 0.05:
            continue
        r3 = fisher((a, b), "n_ge_3_0")
        r25 = fisher((a, b), "n_ge_2_5")
        results_3.append((a, b, r3))
        results_25.append((a, b, r25))

    def bh_fdr(pvals: list[float], alpha: float = 0.05) -> tuple[int, list[bool]]:
        n = len(pvals)
        if n == 0:
            return 0, []
        order = sorted(range(n), key=lambda i: pvals[i] if pvals[i] is not None else 1.0)
        rejected = [False] * n
        passed = 0
        for rank, i in enumerate(order, start=1):
            crit = (rank / n) * alpha
            if pvals[i] is not None and pvals[i] <= crit:
                passed = rank
        for i in order[:passed]:
            rejected[i] = True
        return sum(rejected), rejected

    p3 = [r[2]["p"] for r in results_3]
    p25 = [r[2]["p"] for r in results_25]
    n_sig_3, _ = bh_fdr(p3)
    n_sig_25, _ = bh_fdr(p25)

    return {
        "n_matched_accuracy_pairs": len(results_3),
        "n_significant_at_M_ge_3_0_BH_q05": int(n_sig_3),
        "n_significant_at_M_ge_2_5_BH_q05": int(n_sig_25),
        "per_model": per_model,
    }


# ----------------------------------------------------------------------
# B5: judge family leniency
# ----------------------------------------------------------------------

def b5_judge_leniency() -> dict:
    print("\n[B5] Judge family leniency analysis...")
    models = all_models()
    judge_scores = defaultdict(list)
    judge_target_scores = defaultdict(lambda: defaultdict(list))
    for m in models:
        for r in load_records(m):
            for jrole in ("primary", "secondary"):
                jname = r.get(f"{jrole}_judge")
                jscore = r.get(f"{jrole}_score")
                if jname and jscore is not None:
                    judge_scores[jname].append(float(jscore))
                    judge_target_scores[jname][m].append(float(jscore))

    rows = []
    for jname, scores in sorted(judge_scores.items(),
                                 key=lambda kv: -len(kv[1])):
        arr = np.asarray(scores)
        rows.append({
            "judge": jname,
            "n_records": int(arr.size),
            "mean_score": float(arr.mean()),
            "median_score": float(np.median(arr)),
            "frac_zero": float((arr == 0).mean()),
            "frac_ge_2": float((arr >= 2).mean()),
            "frac_ge_3": float((arr >= 3).mean()),
        })

    # Kruskal-Wallis on judge mean scores across all targets
    if len(judge_scores) >= 2:
        groups = [np.asarray(v) for v in judge_scores.values() if len(v) > 30]
        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            kw = {"H": float(stat), "p": float(p)}
        else:
            kw = None
    else:
        kw = None
    return {"per_judge": rows, "kruskal_wallis": kw}


def main() -> None:
    print("=" * 70)
    print("PHASE B NEW (B1-B5) on 10K data")
    print("=" * 70)

    if not SCORES_10K.exists():
        print(f"ERROR: {SCORES_10K} does not exist. Run A5 merge first.")
        return

    out = {}
    out["B1_hierarchical_bootstrap"] = b1_hierarchical_bootstrap(n_iter=200)
    out["B2_fixed_mmin"] = b2_fixed_mmin()
    out["B3_model_agnostic"] = b3_model_agnostic()
    out["B4_binomial_test"] = b4_binomial_test()
    out["B5_judge_leniency"] = b5_judge_leniency()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")
    print()
    print("Summary:")
    print(f"  B1 hier-bootstrap median pairs = {out['B1_hierarchical_bootstrap']['pair_count_median']:.0f} "
          f"(95% CI [{out['B1_hierarchical_bootstrap']['pair_count_p2_5']:.0f}, "
          f"{out['B1_hierarchical_bootstrap']['pair_count_p97_5']:.0f}])")
    for k, v in out["B2_fixed_mmin"].items():
        print(f"  B2 {k}: {v['n_disjoint_CIs']} disjoint pairs ({v['n_models_with_valid_b']} models)")
    print(f"  B3 log-linear pairs (|Δb|>0.15): {out['B3_model_agnostic']['n_pairs_loglinear_b_diff_gt_0_15']}")
    print(f"  B3 tail-ratio pairs (|Δr|>0.01): {out['B3_model_agnostic']['n_pairs_tail_ratio_diff_gt_0_01']}")
    print(f"  B4 Fisher M>=3 BH q<0.05: {out['B4_binomial_test']['n_significant_at_M_ge_3_0_BH_q05']}")
    print(f"  B4 Fisher M>=2.5 BH q<0.05: {out['B4_binomial_test']['n_significant_at_M_ge_2_5_BH_q05']}")
    if out["B5_judge_leniency"]["kruskal_wallis"]:
        print(f"  B5 KW H={out['B5_judge_leniency']['kruskal_wallis']['H']:.1f} "
              f"p={out['B5_judge_leniency']['kruskal_wallis']['p']:.3g}")


if __name__ == "__main__":
    main()
