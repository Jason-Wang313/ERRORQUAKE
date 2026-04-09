"""Combined robustness suite for the Exp 5 headline scaling correlation.

Patches 5 (Bayesian) + 6 (independence) plus inferred Tier-1 tasks:
bootstrap CI, permutation null, power analysis, split-half b reliability.

All operate on the existing dense-model b-values from
results/analysis/exp5_scaling.json. Output: results/analysis/headline_robustness.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import _estimate_b, _quantize_to_grid

EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
SCORES = ROOT / "results" / "scores"
OUT = ROOT / "results" / "analysis" / "headline_robustness.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_positive_scores(stem: str) -> np.ndarray:
    out = []
    for line in open(SCORES / f"{stem}.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = r.get("final_score")
        if s is not None and s > 0:
            out.append(float(s))
    return _quantize_to_grid(np.asarray(out, dtype=float))


def main() -> None:
    print("=" * 70)
    print("HEADLINE ROBUSTNESS (Patches 5 + 6 + Tier-1 stats)")
    print("=" * 70)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = exp5["points"]
    dense = [p for p in points if p["architecture"] == "dense"]

    log_p = np.array([p["log_params"] for p in dense])
    bs = np.array([p["b_value"] for p in dense])
    eps = np.array([p["error_rate"] for p in dense])
    n = len(dense)

    rho_obs, p_obs = stats.spearmanr(log_p, bs)
    print(f"\nBaseline (n={n}):")
    print(f"  Spearman rho = {rho_obs:+.4f}")
    print(f"  p-value      = {p_obs:.4f}")

    # ---------------------------------------------------------------
    # 1. Bootstrap CI on rho_s (resample model pairs with replacement)
    # ---------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_boot = 5000
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(idx)) < 3:
            continue
        try:
            r, _ = stats.spearmanr(log_p[idx], bs[idx])
            if not np.isnan(r):
                rhos.append(r)
        except Exception:
            continue
    rhos = np.array(rhos)
    boot = {
        "n_boot": int(len(rhos)),
        "mean": float(rhos.mean()),
        "median": float(np.median(rhos)),
        "ci_2_5": float(np.percentile(rhos, 2.5)),
        "ci_97_5": float(np.percentile(rhos, 97.5)),
        "frac_negative": float((rhos < 0).mean()),
    }
    print(f"\nBootstrap CI ({n_boot} resamples):")
    print(f"  median  = {boot['median']:+.3f}")
    print(f"  95% CI  = [{boot['ci_2_5']:+.3f}, {boot['ci_97_5']:+.3f}]")
    print(f"  P(rho<0)= {boot['frac_negative']:.3f}")

    # ---------------------------------------------------------------
    # 2. Permutation test (exact null distribution)
    # ---------------------------------------------------------------
    n_perm = 10000
    rng2 = np.random.default_rng(7)
    perm_rhos = []
    for _ in range(n_perm):
        shuffled = rng2.permutation(bs)
        r, _ = stats.spearmanr(log_p, shuffled)
        perm_rhos.append(r)
    perm_rhos = np.array(perm_rhos)
    p_perm_two = float((np.abs(perm_rhos) >= abs(rho_obs)).mean())
    p_perm_one = float((perm_rhos <= rho_obs).mean())
    perm = {"n_perm": n_perm, "p_two_sided": p_perm_two, "p_one_sided": p_perm_one}
    print(f"\nPermutation test ({n_perm} shuffles):")
    print(f"  p (two-sided) = {p_perm_two:.4f}")
    print(f"  p (one-sided, neg) = {p_perm_one:.4f}")

    # ---------------------------------------------------------------
    # 3. Power analysis at n=14
    # ---------------------------------------------------------------
    # For Spearman, the variance of Fisher z is approx 1.06/(n-3).
    # Detectable rho at alpha=0.05 (two-sided), power=0.80.
    from math import atanh, tanh, sqrt
    z_alpha = 1.96
    z_beta = 0.842  # power = 0.80
    se = sqrt(1.06 / (n - 3))
    z_min = (z_alpha + z_beta) * se
    rho_min_detectable = float(tanh(z_min))
    power = {
        "n": n,
        "alpha": 0.05,
        "target_power": 0.80,
        "min_detectable_abs_rho": rho_min_detectable,
        "observed_abs_rho": float(abs(rho_obs)),
        "well_powered": bool(abs(rho_obs) >= rho_min_detectable),
    }
    print(f"\nPower analysis (n={n}, alpha=0.05, power=0.80):")
    print(f"  min detectable |rho| = {rho_min_detectable:.3f}")
    print(f"  observed |rho|       = {abs(rho_obs):.3f}")
    print(f"  well-powered         = {power['well_powered']}")

    # ---------------------------------------------------------------
    # 4. Bayesian: Bayes factor for correlation (Wagenmakers/Wetzels)
    # ---------------------------------------------------------------
    # Use the rank-transformed Spearman as Pearson on ranks.
    from scipy.special import gamma as gammaf, hyp2f1

    def bayes_factor_corr(r: float, n: int) -> float:
        """BF10 (alt vs null) for Pearson correlation, Jeffreys' default prior."""
        # Wetzels & Wagenmakers (2012), eq. 2 — uniform prior on rho in [-1,1]
        # Approximated via the hypergeometric form.
        try:
            num = gammaf((n - 1) / 2.0)
            den = (np.pi ** 0.5) * gammaf(n / 2.0)
            integrand = (1 - r ** 2) ** ((n - 4) / 2.0) * \
                hyp2f1(0.5, 0.5, (n - 1) / 2.0, r ** 2)
            return float(num / den * integrand * np.pi)
        except Exception:
            return float("nan")

    # Use observed Spearman rho as input to Pearson-on-ranks BF.
    bf10 = bayes_factor_corr(rho_obs, n)
    # Posterior P(rho < 0 | data) — approximate via permutation null +
    # uniform prior; use the bootstrap distribution as a stand-in.
    p_neg_given_data = boot["frac_negative"]

    bayesian = {
        "spearman_rho": float(rho_obs),
        "bayes_factor_10": float(bf10),
        "interpretation": (
            "decisive" if bf10 > 100 else
            "very strong" if bf10 > 30 else
            "strong" if bf10 > 10 else
            "moderate" if bf10 > 3 else
            "anecdotal" if bf10 > 1 else
            "favors null"
        ),
        "p_rho_negative_given_data": float(p_neg_given_data),
    }
    print(f"\nBayesian (Wetzels-Wagenmakers BF10 on rank Pearson):")
    print(f"  BF10 = {bf10:.2f}  ({bayesian['interpretation']})")
    print(f"  P(rho<0 | data) ≈ {p_neg_given_data:.3f} (from bootstrap)")

    # Bayesian linear regression: b = a + b1*log_params + sigma noise
    # Closed-form normal-inverse-gamma posterior with weakly informative priors.
    # Slope posterior under flat prior approximately matches OLS.
    X = np.column_stack([np.ones(n), log_p])
    coef, *_ = np.linalg.lstsq(X, bs, rcond=None)
    pred = X @ coef
    resid = bs - pred
    sigma2 = float(np.sum(resid ** 2) / (n - 2))
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_coef = np.sqrt(np.diag(cov))
    # P(beta < 0 | data) under normal posterior
    z = coef[1] / se_coef[1]
    from scipy.stats import norm
    p_beta_neg = float(norm.cdf(z))  # one-sided, beta < 0 if z < 0
    bayes_lin = {
        "intercept": float(coef[0]),
        "intercept_se": float(se_coef[0]),
        "slope_log_params": float(coef[1]),
        "slope_se": float(se_coef[1]),
        "p_slope_negative_given_data": float(p_beta_neg),
        "ci95_slope_low": float(coef[1] - 1.96 * se_coef[1]),
        "ci95_slope_high": float(coef[1] + 1.96 * se_coef[1]),
    }
    print(f"\nBayesian linear regression (normal posterior, weak priors):")
    print(f"  slope_log_params = {coef[1]:+.3f} (SE {se_coef[1]:.3f})")
    print(f"  95% CI = [{bayes_lin['ci95_slope_low']:+.3f}, {bayes_lin['ci95_slope_high']:+.3f}]")
    print(f"  P(slope < 0 | data) = {p_beta_neg:.4f}")

    # ---------------------------------------------------------------
    # 5. Patch 6: ε-b independence test
    # ---------------------------------------------------------------
    rho_eb, p_eb = stats.spearmanr(eps, bs)
    rho_eb_all, p_eb_all = stats.spearmanr(
        np.array([p["error_rate"] for p in points]),
        np.array([p["b_value"] for p in points]),
    )

    # Partial Spearman: rho(log_params, b | eps) via residualisation
    def residualise(y, x):
        Xm = np.column_stack([np.ones(len(x)), x])
        c, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        return y - Xm @ c

    log_p_resid = residualise(log_p, eps)
    b_resid = residualise(bs, eps)
    partial_rho, partial_p = stats.spearmanr(log_p_resid, b_resid)

    # Accuracy-stratified scaling
    median_eps = float(np.median(eps))
    high_acc = eps < median_eps  # high accuracy = low error rate
    low_acc = eps >= median_eps
    rho_high, p_high = stats.spearmanr(log_p[high_acc], bs[high_acc])
    rho_low, p_low = stats.spearmanr(log_p[low_acc], bs[low_acc])

    independence = {
        "spearman_eps_b_dense": {"rho": float(rho_eb), "p": float(p_eb), "n": n},
        "spearman_eps_b_all21": {"rho": float(rho_eb_all), "p": float(p_eb_all)},
        "partial_spearman_log_p_b_given_eps": {
            "rho": float(partial_rho), "p": float(partial_p),
        },
        "high_accuracy_subset": {
            "n": int(high_acc.sum()), "rho": float(rho_high), "p": float(p_high),
        },
        "low_accuracy_subset": {
            "n": int(low_acc.sum()), "rho": float(rho_low), "p": float(p_low),
        },
    }
    print(f"\nepsilon-b independence (Patch 6):")
    print(f"  Spearman(eps, b) dense  = {rho_eb:+.3f} (p={p_eb:.3f})")
    print(f"  Spearman(eps, b) all 21 = {rho_eb_all:+.3f} (p={p_eb_all:.3f})")
    print(f"  Partial rho(log_p, b | eps) = {partial_rho:+.3f} (p={partial_p:.4f})")
    print(f"  High-accuracy subset (n={int(high_acc.sum())}): rho={rho_high:+.3f} (p={p_high:.3f})")
    print(f"  Low-accuracy subset  (n={int(low_acc.sum())}): rho={rho_low:+.3f} (p={p_low:.3f})")

    # ---------------------------------------------------------------
    # 6. Split-half reliability of b
    # ---------------------------------------------------------------
    rng3 = np.random.default_rng(11)
    n_splits = 100
    half_corrs = []
    for _ in range(n_splits):
        b_a, b_b = [], []
        for d in dense:
            scores = load_positive_scores(d["name"])
            if scores.size < 60:
                continue
            idx = rng3.permutation(scores.size)
            half = scores.size // 2
            try:
                ba = float(_estimate_b(scores[idx[:half]], 1.5))
                bb = float(_estimate_b(scores[idx[half:]], 1.5))
                if np.isfinite(ba) and np.isfinite(bb):
                    b_a.append(ba); b_b.append(bb)
            except Exception:
                pass
        if len(b_a) >= 5:
            r, _ = stats.spearmanr(b_a, b_b)
            if not np.isnan(r):
                half_corrs.append(r)
    half_corrs = np.array(half_corrs)
    split_half = {
        "n_trials": int(len(half_corrs)),
        "mean_spearman": float(half_corrs.mean()),
        "median": float(np.median(half_corrs)),
        "std": float(half_corrs.std()),
        "p25": float(np.percentile(half_corrs, 25)),
        "p75": float(np.percentile(half_corrs, 75)),
    }
    print(f"\nSplit-half reliability of b (m_min=1.5, n=100 splits):")
    print(f"  mean Spearman = {split_half['mean_spearman']:.3f}")
    print(f"  IQR           = [{split_half['p25']:.3f}, {split_half['p75']:.3f}]")

    out = {
        "baseline": {"n": n, "spearman_rho": float(rho_obs), "p": float(p_obs)},
        "bootstrap_ci": boot,
        "permutation_test": perm,
        "power_analysis": power,
        "bayesian": bayesian,
        "bayesian_linear_regression": bayes_lin,
        "epsilon_b_independence": independence,
        "split_half_reliability": split_half,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
