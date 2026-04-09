"""Task 2.2: simulation validation.

Generate synthetic "models" with known severity distributions on the
9-point grid, plant a scaling correlation, run the full b-fitting
pipeline, and check recovery. Also add κ≈0.37 judge noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value, fit_all_distributions

OUT = ROOT / "results" / "analysis" / "simulation_validation.json"
GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
RNG = np.random.default_rng(1234)


def make_severity_dist(family: str, params: dict, eps: float) -> np.ndarray:
    """Return a PMF over GRID where P(0.0) = 1 - eps and the positive
    mass follows the given family scaled to match the observed range."""
    m = GRID
    pos_m = m[1:]  # exclude 0.0

    if family == "exponential":
        lam = params["lambda"]
        w = np.exp(-lam * (pos_m - pos_m[0]))
    elif family == "stretched_exp":
        lam = params["lambda"]
        gamma = params["gamma"]
        w = np.exp(-lam * np.power(pos_m, gamma))
    elif family == "truncated_power_law":
        beta = params["beta"]
        lam = params["lambda"]
        w = (pos_m ** (-beta)) * np.exp(-lam * pos_m)
    elif family == "lognormal":
        mu = params["mu"]
        sigma = params["sigma"]
        w = (1.0 / (pos_m * sigma)) * np.exp(-(np.log(pos_m) - mu) ** 2 / (2 * sigma ** 2))
    else:
        raise ValueError(family)
    w = w / w.sum()
    pmf = np.zeros_like(m)
    pmf[0] = 1.0 - eps
    pmf[1:] = eps * w
    return pmf / pmf.sum()


def sample(pmf: np.ndarray, n: int) -> np.ndarray:
    idx = RNG.choice(len(GRID), size=n, p=pmf)
    return GRID[idx]


def add_judge_noise(scores: np.ndarray, noise_prob: float,
                    empirical_dist: np.ndarray) -> np.ndarray:
    mask = RNG.random(size=scores.size) < noise_prob
    n_noise = int(mask.sum())
    if n_noise == 0:
        return scores
    replacement = RNG.choice(GRID, size=n_noise, p=empirical_dist)
    out = scores.copy()
    out[mask] = replacement
    return out


def build_catalog() -> list[dict]:
    """21 synthetic models with known tail heaviness parameters."""
    catalog = []
    # Stretched exponential (varying stretch)
    for i, gamma in enumerate([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]):
        catalog.append({
            "name": f"synth_strexp_{gamma:.1f}",
            "family": "stretched_exp",
            "params": {"lambda": 0.8, "gamma": gamma},
            "eps": 0.45 + 0.04 * i,
            "true_tail_heaviness": 1.0 / gamma,  # higher = heavier
        })
    # Exponential (varying rate)
    for i, lam in enumerate([0.6, 0.9, 1.2, 1.5, 1.9]):
        catalog.append({
            "name": f"synth_exp_{lam:.1f}",
            "family": "exponential",
            "params": {"lambda": lam},
            "eps": 0.30 + 0.05 * i,
            "true_tail_heaviness": 1.0 / lam,
        })
    # Truncated power law
    for i, beta in enumerate([0.5, 0.8, 1.2, 1.6, 2.0]):
        catalog.append({
            "name": f"synth_tpl_{beta:.1f}",
            "family": "truncated_power_law",
            "params": {"beta": beta, "lambda": 0.5},
            "eps": 0.35 + 0.04 * i,
            "true_tail_heaviness": 1.0 / max(beta, 0.3),
        })
    # Lognormal
    for i, sigma in enumerate([0.3, 0.5, 0.8, 1.2]):
        catalog.append({
            "name": f"synth_lognorm_{sigma:.1f}",
            "family": "lognormal",
            "params": {"mu": 0.5, "sigma": sigma},
            "eps": 0.30 + 0.08 * i,
            "true_tail_heaviness": sigma,
        })
    return catalog


def run_pipeline(scores: np.ndarray, name: str):
    pos = scores[scores > 0]
    if pos.size < 30:
        return None
    try:
        bv = estimate_b_value(pos, model_name=name)
        fits = fit_all_distributions(pos, model_name=name)
        best = min(fits, key=lambda f: f.bic if np.isfinite(f.bic) else 1e9)
        return {
            "b": float(bv.b),
            "b_ci_lo": float(bv.b_ci_lower),
            "b_ci_hi": float(bv.b_ci_upper),
            "best_fit": best.distribution,
        }
    except Exception as exc:
        return {"error": str(exc)[:120]}


def main() -> None:
    print("=" * 70)
    print("SIMULATION VALIDATION (Task 2.2)")
    print("=" * 70)

    catalog = build_catalog()
    print(f"Synthetic catalog: {len(catalog)} models")

    # Plant a negative correlation between true tail heaviness and "scale"
    # by assigning log_params in descending order of heaviness.
    catalog_sorted = sorted(catalog, key=lambda m: -m["true_tail_heaviness"])
    for i, m in enumerate(catalog_sorted):
        m["log_params"] = 9.0 + 1.5 * i / max(len(catalog_sorted) - 1, 1)

    # Empirical score distribution from the real data (for noise model)
    from collections import Counter
    import json as _j
    pooled = []
    for f in Path(ROOT / "results" / "scores").glob("*.jsonl"):
        if "reasoning" in f.stem or "70b" in f.stem:
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = _j.loads(line)
                except _j.JSONDecodeError:
                    continue
                s = r.get("final_score")
                if s is not None:
                    idx = int(np.argmin(np.abs(GRID - float(s))))
                    pooled.append(idx)
                    if len(pooled) > 20000:
                        break
        if len(pooled) > 20000:
            break
    c = Counter(pooled)
    empirical_pmf = np.array([c.get(i, 0) for i in range(len(GRID))], dtype=float)
    empirical_pmf = empirical_pmf / empirical_pmf.sum()

    conditions = {
        "noise_0": 0.0,     # clean
        "noise_15": 0.15,   # modest
        "noise_35": 0.35,   # ~matches kappa=0.374 regime
    }

    all_results = {}
    for cond_name, noise in conditions.items():
        print(f"\n--- condition: {cond_name} (noise_prob={noise}) ---")
        results = []
        for m in catalog_sorted:
            pmf = make_severity_dist(m["family"], m["params"], m["eps"])
            raw_scores = sample(pmf, n=4000)
            if noise > 0:
                raw_scores = add_judge_noise(raw_scores, noise, empirical_pmf)
            pipe = run_pipeline(raw_scores, m["name"])
            if pipe and "b" in pipe:
                results.append({
                    "name": m["name"],
                    "family_true": m["family"],
                    "family_fit": pipe["best_fit"],
                    "true_heaviness": m["true_tail_heaviness"],
                    "estimated_b": pipe["b"],
                    "log_params": m["log_params"],
                })

        # Recovery: spearman between true heaviness and estimated b
        th = np.array([r["true_heaviness"] for r in results])
        eb = np.array([r["estimated_b"] for r in results])
        lp = np.array([r["log_params"] for r in results])
        rho_th_b, p_th_b = stats.spearmanr(th, eb)
        rho_lp_b, p_lp_b = stats.spearmanr(lp, eb)
        # Family recovery
        correct_family = sum(1 for r in results if r["family_true"] == r["family_fit"])
        print(f"  n recovered = {len(results)}/{len(catalog_sorted)}")
        print(f"  rho(true_heaviness, est_b)  = {rho_th_b:+.3f}  (heavier true -> lower b)")
        print(f"  rho(log_params, est_b)      = {rho_lp_b:+.3f}  (planted = negative)")
        print(f"  correct BIC-best family     = {correct_family}/{len(results)}")

        all_results[cond_name] = {
            "noise_prob": noise,
            "n_recovered": len(results),
            "rho_true_vs_b": float(rho_th_b),
            "p_true_vs_b": float(p_th_b),
            "rho_logp_vs_b": float(rho_lp_b),
            "p_logp_vs_b": float(p_lp_b),
            "correct_family_count": int(correct_family),
            "results": results,
        }

    OUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
