"""ERRORQUAKE v8 — combined zero-NIM analyses.

Runs on the 10K merged data. Each section is independent.

 1. Cost analysis (jawdrop §4)
 2. Accuracy illusion examples (jawdrop §1)
 3. Phase transition detection (jawdrop §2)
 4. Universality clustering (jawdrop §7)
 5. Calibration × severity quadrant (jawdrop §6)
 6. Tolerance sensitivity sweep (v8 base §3.1)
 7. Fixed upper-tail analysis (v8 base §3.2)
 8. Parametric bootstrap GOF (v8 base §3.3)
 9. Domain-weighted risk metric (v8 base §3.4)
10. Info-theoretic b interpretation (v8 base §1.2)
11. Error-in-variables adjustment (v8 base §2.2)
12. Propensity-matched pairs (v8 base §3.8)

Output: results/analysis/v8_combined.json
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value, fit_all_distributions

SCORES_10K = ROOT / "results" / "scores_10k"
EVALS_10K = ROOT / "results" / "evaluations_10k"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
PB = ROOT / "results" / "analysis" / "phase_b_10k.json"
OUT = ROOT / "results" / "analysis" / "v8_combined.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
COST_MODEL = {0.0: 0, 0.5: 0, 1.0: 0, 1.5: 10, 2.0: 10,
              2.5: 1000, 3.0: 1000, 3.5: 100000, 4.0: 100000}


def load_scores(stem: str) -> list[dict]:
    path = SCORES_10K / f"{stem}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_responses(stem: str) -> list[dict]:
    path = EVALS_10K / f"{stem}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def all_models() -> list[str]:
    return sorted(f.stem for f in SCORES_10K.glob("*.jsonl") if f.stem not in EXCLUDED)


def load_pb():
    return json.loads(PB.read_text(encoding="utf-8"))


def load_exp5():
    return json.loads(EXP5.read_text(encoding="utf-8"))


# =====================================================================
# 1. COST ANALYSIS
# =====================================================================
def cost_analysis() -> dict:
    print("\n[1] Cost analysis...")
    models = all_models()
    rows = []
    for m in models:
        recs = load_scores(m)
        scores = [r.get("final_score") for r in recs if r.get("final_score") is not None]
        arr = np.asarray(scores, dtype=float)
        n = len(arr)
        if n == 0:
            continue
        # Per-severity bin counts
        cost_per_query = 0.0
        for level, cost in COST_MODEL.items():
            count = int(np.sum(np.abs(arr - level) < 0.26))
            cost_per_query += (count / n) * cost
        cost_per_million = cost_per_query * 1e6
        eps = float((arr > 0).mean())
        pb = load_pb()
        b = pb["per_model_10k"].get(m, {}).get("b")
        rows.append({
            "model": m, "eps": eps, "b": b,
            "cost_per_query": cost_per_query,
            "cost_per_million": cost_per_million,
        })
    rows.sort(key=lambda r: -r["cost_per_million"])
    print(f"  Cost range: ${min(r['cost_per_million'] for r in rows):,.0f} – "
          f"${max(r['cost_per_million'] for r in rows):,.0f} per million queries")
    # Find matched-accuracy pairs with biggest cost gap
    best_pair = None
    best_ratio = 0
    for a, b in combinations(rows, 2):
        if abs(a["eps"] - b["eps"]) >= 0.05:
            continue
        if a["cost_per_million"] > 0 and b["cost_per_million"] > 0:
            ratio = max(a["cost_per_million"], b["cost_per_million"]) / min(a["cost_per_million"], b["cost_per_million"])
            if ratio > best_ratio:
                best_ratio = ratio
                best_pair = (a, b) if a["cost_per_million"] > b["cost_per_million"] else (b, a)
    if best_pair:
        print(f"  Biggest matched-accuracy cost gap: "
              f"{best_pair[0]['model']} ${best_pair[0]['cost_per_million']:,.0f} vs "
              f"{best_pair[1]['model']} ${best_pair[1]['cost_per_million']:,.0f} "
              f"({best_ratio:.1f}×)")
    return {"per_model": rows, "best_matched_pair_ratio": best_ratio,
            "best_pair": [best_pair[0]["model"], best_pair[1]["model"]] if best_pair else None}


# =====================================================================
# 2. ACCURACY ILLUSION EXAMPLES
# =====================================================================
def accuracy_illusion() -> dict:
    print("\n[2] Accuracy illusion examples...")
    pb = load_pb()
    pm = pb["per_model_10k"]
    # Find the most dramatic pair
    target_a, target_b = "deepseek-v3.2", "ministral-14b"
    if target_a not in pm or target_b not in pm:
        # fallback to any pair with large b gap at matched eps
        all_m = [m for m in pm if pm[m].get("b") is not None]
        best = None
        for a, b in combinations(all_m, 2):
            if abs(pm[a]["eps"] - pm[b]["eps"]) >= 0.05:
                continue
            gap = abs(pm[a]["b"] - pm[b]["b"])
            if best is None or gap > best[2]:
                best = (a, b, gap)
        if best:
            target_a, target_b = best[0], best[1]

    # Load responses and scores for both models
    recs_a = {r["query_id"]: r for r in load_scores(target_a)}
    recs_b = {r["query_id"]: r for r in load_scores(target_b)}
    resp_a = {r["query_id"]: r for r in load_responses(target_a)}
    resp_b = {r["query_id"]: r for r in load_responses(target_b)}

    # Find queries where one is catastrophic and the other is mild
    examples = []
    common = set(recs_a.keys()) & set(recs_b.keys())
    for qid in common:
        sa = recs_a[qid].get("final_score")
        sb = recs_b[qid].get("final_score")
        if sa is None or sb is None:
            continue
        # A is heavy-tail model → want A catastrophic, B mild
        if pm[target_a]["b"] < pm[target_b]["b"]:
            heavy, light = target_a, target_b
            sh, sl = sa, sb
            rh, rl = resp_a.get(qid, {}), resp_b.get(qid, {})
        else:
            heavy, light = target_b, target_a
            sh, sl = sb, sa
            rh, rl = resp_b.get(qid, {}), resp_a.get(qid, {})
        if sh >= 3.0 and sl <= 1.5 and sl > 0:
            examples.append({
                "query_id": qid,
                "query": rh.get("question", ""),
                "heavy_model": heavy,
                "heavy_response": rh.get("response_text", "")[:300],
                "heavy_score": sh,
                "light_model": light,
                "light_response": rl.get("response_text", "")[:300],
                "light_score": sl,
            })
    examples.sort(key=lambda e: e["heavy_score"] - e["light_score"], reverse=True)
    print(f"  Found {len(examples)} contrast examples for {target_a} vs {target_b}")
    print(f"  Top 5 severity gaps: {[e['heavy_score']-e['light_score'] for e in examples[:5]]}")
    return {
        "pair": [target_a, target_b],
        "pair_eps": [pm[target_a]["eps"], pm[target_b]["eps"]],
        "pair_b": [pm[target_a]["b"], pm[target_b]["b"]],
        "n_contrast_examples": len(examples),
        "top_5_examples": examples[:5],
    }


# =====================================================================
# 3. PHASE TRANSITION DETECTION
# =====================================================================
def phase_transition() -> dict:
    print("\n[3] Phase transition detection...")
    exp5 = load_exp5()
    pb = load_pb()
    pm = pb["per_model_10k"]
    points = {p["name"]: p for p in exp5["points"]}
    dense = [(m, points[m]["log_params"], pm[m].get("best_fit"))
             for m in pm if points.get(m, {}).get("architecture") == "dense"
             and pm[m].get("best_fit") is not None]
    dense.sort(key=lambda x: x[1])
    # Classify: small (<10B log_p ~10.0) vs large (>=10B)
    threshold = 10.0
    small = [d for d in dense if d[1] < threshold]
    large = [d for d in dense if d[1] >= threshold]
    small_exp = sum(1 for _, _, f in small if f == "exponential")
    small_nonexp = len(small) - small_exp
    large_exp = sum(1 for _, _, f in large if f == "exponential")
    large_nonexp = len(large) - large_exp
    # Fisher's exact
    table = [[small_exp, small_nonexp], [large_exp, large_nonexp]]
    _, p_fisher = stats.fisher_exact(table)
    print(f"  Small (<10B): {small_exp} exp, {small_nonexp} non-exp")
    print(f"  Large (>=10B): {large_exp} exp, {large_nonexp} non-exp")
    print(f"  Fisher p = {p_fisher:.4f}")
    return {
        "threshold_log_params": threshold,
        "small_n": len(small), "small_exp": small_exp,
        "large_n": len(large), "large_exp": large_exp,
        "fisher_p": float(p_fisher),
        "per_model": [(m, lp, f) for m, lp, f in dense],
    }


# =====================================================================
# 4. UNIVERSALITY CLUSTERING
# =====================================================================
def universality_clustering() -> dict:
    print("\n[4] Universality clustering (JS divergence)...")
    models = all_models()
    # Build severity histograms
    hists = {}
    for m in models:
        recs = load_scores(m)
        scores = [r.get("final_score") for r in recs if r.get("final_score") is not None]
        arr = np.asarray(scores, dtype=float)
        pos = arr[arr > 0]
        if pos.size < 30:
            continue
        hist = np.array([np.sum(np.abs(pos - g) < 0.26) for g in GRID[1:]], dtype=float)
        hist = hist / hist.sum()
        hists[m] = hist

    model_list = sorted(hists.keys())
    n = len(model_list)
    # JS divergence matrix
    def js_div(p, q):
        m = 0.5 * (p + q)
        # Safe KL: skip bins where p or q is zero
        kl_pm = np.sum(np.where(p > 0, p * np.log(p / (m + 1e-15)), 0))
        kl_qm = np.sum(np.where(q > 0, q * np.log(q / (m + 1e-15)), 0))
        d = float(0.5 * kl_pm + 0.5 * kl_qm)
        return max(0.0, d)  # numerical floor

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = js_div(hists[model_list[i]], hists[model_list[j]])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")
    clusters_3 = fcluster(Z, t=3, criterion="maxclust")
    cluster_map = {model_list[i]: int(clusters_3[i]) for i in range(n)}

    # Report
    for c in sorted(set(clusters_3)):
        members = [model_list[i] for i in range(n) if clusters_3[i] == c]
        print(f"  Cluster {c}: {members}")
    return {"clusters": cluster_map, "n_clusters": 3, "linkage_method": "ward",
            "model_list": model_list}


# =====================================================================
# 5. CALIBRATION × SEVERITY QUADRANT
# =====================================================================
def calibration_severity() -> dict:
    print("\n[5] Calibration × severity quadrant...")
    models = all_models()
    HEDGE_PATTERNS = [
        "i'm not sure", "i don't know", "i cannot", "it's possible",
        "may be", "might be", "i believe", "it seems", "approximately",
        "roughly", "i think", "not entirely certain",
    ]
    rows = []
    for m in models:
        resps = load_responses(m)
        scores_recs = {r["query_id"]: r for r in load_scores(m)}
        confident_errors = 0
        confident_total = 0
        for r in resps:
            text = (r.get("response_text") or "").lower()
            qid = r.get("query_id")
            if not text or qid not in scores_recs:
                continue
            hedges = any(h in text for h in HEDGE_PATTERNS)
            if not hedges:
                confident_total += 1
                fs = scores_recs[qid].get("final_score")
                if fs is not None and fs > 0:
                    confident_errors += 1
        overconf = confident_errors / max(confident_total, 1)
        pb = load_pb()
        b = pb["per_model_10k"].get(m, {}).get("b")
        rows.append({"model": m, "overconfidence_rate": overconf,
                     "confident_total": confident_total, "b": b})
    # Quadrant assignment
    median_overconf = np.median([r["overconfidence_rate"] for r in rows])
    median_b = np.median([r["b"] for r in rows if r["b"] is not None])
    for r in rows:
        if r["b"] is None:
            r["quadrant"] = "unknown"
        elif r["overconfidence_rate"] < median_overconf and r["b"] > median_b:
            r["quadrant"] = "IDEAL"
        elif r["overconfidence_rate"] < median_overconf and r["b"] <= median_b:
            r["quadrant"] = "DANGEROUS"
        elif r["overconfidence_rate"] >= median_overconf and r["b"] > median_b:
            r["quadrant"] = "TOLERABLE"
        else:
            r["quadrant"] = "WORST"
    for q in ["IDEAL", "TOLERABLE", "DANGEROUS", "WORST"]:
        members = [r["model"] for r in rows if r["quadrant"] == q]
        print(f"  {q}: {members}")
    return {"per_model": rows, "median_overconf": float(median_overconf),
            "median_b": float(median_b)}


# =====================================================================
# 6. TOLERANCE SENSITIVITY SWEEP
# =====================================================================
def tolerance_sweep() -> dict:
    print("\n[6] Tolerance sensitivity sweep...")
    pb = load_pb()
    pm = pb["per_model_10k"]
    valid = [m for m in pm if pm[m].get("b") is not None]
    rows = []
    for tol in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        eligible = 0
        disjoint = 0
        for a, b in combinations(valid, 2):
            if abs(pm[a]["eps"] - pm[b]["eps"]) < tol:
                eligible += 1
                if abs(pm[a]["b"] - pm[b]["b"]) > 0.15:
                    if max(pm[a]["ci_lo"], pm[b]["ci_lo"]) > min(pm[a]["ci_hi"], pm[b]["ci_hi"]):
                        disjoint += 1
        frac = disjoint / max(eligible, 1)
        rows.append({"tolerance": tol, "eligible": eligible,
                     "disjoint": disjoint, "fraction": frac})
        print(f"  tol={tol:.2f}: eligible={eligible}, disjoint={disjoint}, frac={frac:.3f}")
    return {"rows": rows}


# =====================================================================
# 7. FIXED UPPER-TAIL ANALYSIS
# =====================================================================
def fixed_upper_tail() -> dict:
    print("\n[7] Fixed upper-tail analysis...")
    from errorquake.analyze import _estimate_b, _quantize_to_grid
    models = all_models()
    pb = load_pb()
    pm = pb["per_model_10k"]
    results = {}
    for mstar in [2.0, 2.5, 3.0]:
        per_model = {}
        for m in models:
            recs = load_scores(m)
            scores = np.array([r.get("final_score") for r in recs
                               if r.get("final_score") is not None], dtype=float)
            pos = _quantize_to_grid(scores[scores > 0])
            # Log-linear regression on upper bins
            bins = np.arange(mstar, 4.5, 0.5)
            counts = np.array([float((pos >= b - 1e-9).sum()) for b in bins])
            mask = counts > 0
            if mask.sum() >= 2:
                xs = bins[mask]
                ys = np.log10(counts[mask])
                slope, _ = np.polyfit(xs, ys, 1)
                b_ll = float(-slope)
            else:
                b_ll = None
            eps = pm.get(m, {}).get("eps", 0)
            per_model[m] = {"b_loglinear": b_ll, "eps": eps}
        # Count pairs
        valid = [m for m in per_model if per_model[m]["b_loglinear"] is not None]
        n_pairs = 0
        for a, b in combinations(valid, 2):
            if abs(per_model[a]["eps"] - per_model[b]["eps"]) >= 0.05:
                continue
            if abs(per_model[a]["b_loglinear"] - per_model[b]["b_loglinear"]) > 0.15:
                n_pairs += 1
        results[f"mstar_{mstar}"] = {
            "n_models": len(valid),
            "n_pairs_b_diff_gt_015": n_pairs,
        }
        print(f"  M>={mstar}: {len(valid)} models, {n_pairs} pairs with |Δb_ll|>0.15")
    return results


# =====================================================================
# 8. PARAMETRIC BOOTSTRAP GOF
# =====================================================================
def parametric_bootstrap_gof(n_sim: int = 200) -> dict:
    print("\n[8] Parametric bootstrap GOF...")
    from errorquake.analyze import _quantize_to_grid, _pmf_for_fit
    models = all_models()
    rng = np.random.default_rng(42)
    results = {}
    for m in models:
        recs = load_scores(m)
        scores = np.array([r.get("final_score") for r in recs
                           if r.get("final_score") is not None], dtype=float)
        pos = _quantize_to_grid(scores[scores > 0])
        if pos.size < 30:
            continue
        try:
            fits = fit_all_distributions(pos, model_name=m, m_min=0.5)
            valid = [f for f in fits if f.bic == f.bic and f.bic != float("inf")]
            if not valid:
                continue
            best = valid[0]
            support = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
            fitted_pmf = _pmf_for_fit(support, best)
            # Observed KS
            obs_hist = np.array([(np.abs(pos - g) < 0.26).sum() for g in support], dtype=float)
            obs_pmf = obs_hist / obs_hist.sum()
            ks_obs = float(np.max(np.abs(np.cumsum(obs_pmf) - np.cumsum(fitted_pmf))))
            # Simulate
            ks_sims = []
            for _ in range(n_sim):
                sim = rng.choice(support, size=pos.size, p=fitted_pmf)
                sim_hist = np.array([(np.abs(sim - g) < 0.26).sum() for g in support], dtype=float)
                sim_pmf = sim_hist / sim_hist.sum()
                ks_sim = float(np.max(np.abs(np.cumsum(sim_pmf) - np.cumsum(fitted_pmf))))
                ks_sims.append(ks_sim)
            p_val = float(np.mean(np.array(ks_sims) >= ks_obs))
            results[m] = {"family": best.distribution, "ks_obs": ks_obs,
                          "p_gof": p_val, "adequate": p_val > 0.05}
            status = "✓" if p_val > 0.05 else "✗"
            print(f"  {m:<28} {best.distribution:<18} p={p_val:.3f} {status}")
        except Exception as exc:
            results[m] = {"error": str(exc)[:80]}
    n_adequate = sum(1 for r in results.values() if r.get("adequate"))
    print(f"  Adequate fits: {n_adequate}/{len(results)}")
    return results


# =====================================================================
# 9. DOMAIN-WEIGHTED RISK
# =====================================================================
def domain_weighted_risk() -> dict:
    print("\n[9] Domain-weighted risk metric...")
    models = all_models()
    SAFETY_WEIGHTS = {"LAW": 2, "BIO": 2, "FIN": 2, "HIST": 1,
                      "GEO": 1, "SCI": 1, "TECH": 1, "CULT": 1}
    total_w = sum(SAFETY_WEIGHTS.values())
    SAFETY_WEIGHTS = {k: v / total_w for k, v in SAFETY_WEIGHTS.items()}
    rows = []
    for m in models:
        recs = load_scores(m)
        by_dom = defaultdict(list)
        for r in recs:
            d = r.get("domain")
            s = r.get("final_score")
            if d and s is not None:
                by_dom[d].append(float(s))
        risk_uniform = 0.0
        risk_safety = 0.0
        for d in SAFETY_WEIGHTS:
            arr = np.asarray(by_dom.get(d, []), dtype=float)
            if arr.size == 0:
                continue
            p_cat = float((arr >= 3.0).mean())
            risk_uniform += (1.0 / 8) * p_cat
            risk_safety += SAFETY_WEIGHTS[d] * p_cat
        pb = load_pb()
        b = pb["per_model_10k"].get(m, {}).get("b")
        rows.append({"model": m, "risk_uniform": risk_uniform,
                     "risk_safety": risk_safety, "b": b})
    # Spearman(risk, b)
    bs = np.array([r["b"] for r in rows if r["b"] is not None])
    rs = np.array([r["risk_safety"] for r in rows if r["b"] is not None])
    rho, p = stats.spearmanr(bs, rs)
    print(f"  Spearman(b, safety-weighted-risk) = {rho:+.3f} (p={p:.4f})")
    return {"per_model": rows, "rho_b_risk": float(rho), "p_b_risk": float(p)}


# =====================================================================
# 10. INFO-THEORETIC b
# =====================================================================
def info_theoretic_b() -> dict:
    print("\n[10] Information-theoretic b interpretation...")
    models = all_models()
    pb = load_pb()
    pm = pb["per_model_10k"]
    rows = []
    for m in models:
        recs = load_scores(m)
        scores = [r.get("final_score") for r in recs if r.get("final_score") is not None]
        arr = np.asarray(scores, dtype=float)
        pos = arr[arr > 0]
        if pos.size < 30:
            continue
        # Empirical entropy on 8-bin positive distribution
        hist = np.array([np.sum(np.abs(pos - g) < 0.26) for g in GRID[1:]], dtype=float)
        p = hist / hist.sum()
        p = p[p > 0]
        H = float(-np.sum(p * np.log2(p)))
        b = pm.get(m, {}).get("b")
        rows.append({"model": m, "b": b, "entropy": H})
    bs = np.array([r["b"] for r in rows if r["b"] is not None])
    hs = np.array([r["entropy"] for r in rows if r["b"] is not None])
    rho, p = stats.spearmanr(bs, hs)
    print(f"  Spearman(b, entropy) = {rho:+.3f} (p={p:.4f})")
    print(f"  (Expected: negative — low b = heavy tail = high entropy)")
    return {"per_model": rows, "rho_b_entropy": float(rho), "p_b_entropy": float(p)}


# =====================================================================
# 11. ERROR-IN-VARIABLES ADJUSTMENT
# =====================================================================
def error_in_variables() -> dict:
    print("\n[11] Error-in-variables adjustment...")
    # Use ICC(2,1) = 0.374 as reliability
    # The 100-item pilot has human vs judge
    RATINGS = ROOT / "data" / "pilot" / "human_ratings_claude_rater.jsonl"
    KEYS = ROOT / "data" / "pilot" / "human_rating_key.jsonl"
    ratings = {}
    for line in open(RATINGS, encoding="utf-8"):
        r = json.loads(line.strip())
        ratings[r["rating_id"]] = float(r["score_11point"])
    judge_scores = []
    human_scores = []
    for line in open(KEYS, encoding="utf-8"):
        k = json.loads(line.strip())
        rid = k["rating_id"]
        if rid in ratings and k.get("final_score") is not None:
            judge_scores.append(float(k["final_score"]))
            human_scores.append(ratings[rid])
    sigma2_error = float(np.var(np.array(judge_scores) - np.array(human_scores)))
    icc_21 = 0.374  # from v4
    # Attenuation correction
    rho_obs_eps_b = -0.842  # from 10K
    rho_corrected = rho_obs_eps_b / math.sqrt(icc_21)
    rho_corrected = max(-1.0, min(1.0, rho_corrected))
    print(f"  sigma2_error = {sigma2_error:.3f}")
    print(f"  ICC(2,1) = {icc_21}")
    print(f"  rho(eps,b) observed = {rho_obs_eps_b:+.3f}")
    print(f"  rho(eps,b) attenuation-corrected = {rho_corrected:+.3f}")
    return {
        "sigma2_error": sigma2_error,
        "icc_21": icc_21,
        "rho_eps_b_observed": rho_obs_eps_b,
        "rho_eps_b_corrected": rho_corrected,
    }


# =====================================================================
# 12. PROPENSITY-MATCHED PAIRS
# =====================================================================
def propensity_matched() -> dict:
    print("\n[12] Propensity-matched pairs...")
    models = all_models()
    pb = load_pb()
    pm = pb["per_model_10k"]
    DOMAINS = ["BIO", "LAW", "HIST", "GEO", "SCI", "TECH", "FIN", "CULT"]
    features = {}
    for m in models:
        recs = load_scores(m)
        by_dom = defaultdict(list)
        by_tier = defaultdict(list)
        for r in recs:
            d = r.get("domain")
            t = r.get("tier")
            s = r.get("final_score")
            if d and s is not None:
                by_dom[d].append(1 if s > 0 else 0)
            if t is not None and s is not None:
                by_tier[t].append(1 if s > 0 else 0)
        vec = []
        for d in DOMAINS:
            vec.append(np.mean(by_dom.get(d, [0])))
        for t in range(1, 6):
            vec.append(np.mean(by_tier.get(t, [0])))
        features[m] = np.array(vec)

    valid = [m for m in features if pm.get(m, {}).get("b") is not None]
    X = np.array([features[m] for m in valid])
    # Mahalanobis
    cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
    cov_inv = np.linalg.inv(cov)

    for threshold in [1.0, 1.5, 2.0]:
        matched = 0
        disjoint = 0
        for i, j in combinations(range(len(valid)), 2):
            diff = X[i] - X[j]
            d = float(np.sqrt(diff @ cov_inv @ diff))
            if d < threshold:
                matched += 1
                a, b = valid[i], valid[j]
                if abs(pm[a]["b"] - pm[b]["b"]) > 0.15:
                    if max(pm[a]["ci_lo"], pm[b]["ci_lo"]) > min(pm[a]["ci_hi"], pm[b]["ci_hi"]):
                        disjoint += 1
        print(f"  Mahalanobis < {threshold}: {matched} matched, {disjoint} disjoint b-CI pairs")

    return {"note": "per-threshold counts logged above"}


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    print("=" * 70)
    print("ERRORQUAKE v8 — COMBINED ANALYSES")
    print("=" * 70)

    out = {}
    out["cost_analysis"] = cost_analysis()
    out["accuracy_illusion"] = accuracy_illusion()
    out["phase_transition"] = phase_transition()
    out["universality_clustering"] = universality_clustering()
    out["calibration_severity"] = calibration_severity()
    out["tolerance_sweep"] = tolerance_sweep()
    out["fixed_upper_tail"] = fixed_upper_tail()
    out["parametric_bootstrap_gof"] = parametric_bootstrap_gof(n_sim=200)
    out["domain_weighted_risk"] = domain_weighted_risk()
    out["info_theoretic_b"] = info_theoretic_b()
    out["error_in_variables"] = error_in_variables()
    out["propensity_matched"] = propensity_matched()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
