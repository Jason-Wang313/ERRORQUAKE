"""
Test the Multiplicative Error Model (Proposition 3).

If severity = product of stage-level errors, then log(severity) ~ Normal.
Tests:
1. Log-normality of severity scores per model (Shapiro-Wilk on log(scores))
2. Correlation between model layer count proxy and log-variance
3. Whether lognormal-best-fit models show better log-normality than others
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO / "results" / "analysis"
OUTPUT_DIR = REPO / "results" / "analysis" / "oral_upgrade"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Approximate layer counts as proxy for multiplicative stages
MODEL_LAYERS = {
    "llama-3.2-3b-instruct": 28,
    "phi-3.5-mini": 32,
    "gemma-3-4b": 26,
    "qwen2.5-7b": 28,
    "llama-3.1-8b-instruct": 32,
    "eurollm-9b": 32,
    "solar-10.7b": 48,
    "gemma-3-12b": 38,
    "ministral-14b": 40,
    "gpt-oss-20b": 44,
    "mistral-small-24b": 56,
    "gemma-2-27b": 46,
    "gemma-3-27b": 46,
    "seed-oss-36b": 52,
    # MoE (active layers may differ but we use total decoder layers)
    "mistral-small-4-119b": 56,
    "mistral-medium-3": 88,
    "kimi-k2-instruct": 64,
    "llama-4-maverick": 48,
    "deepseek-v3.1": 61,
    "deepseek-v3.2": 61,
    "qwen3-next-80b": 80,
}

MODEL_PARAMS = {
    "llama-3.2-3b-instruct": 3.21,
    "phi-3.5-mini": 3.82,
    "gemma-3-4b": 4.3,
    "qwen2.5-7b": 7.62,
    "llama-3.1-8b-instruct": 8.03,
    "eurollm-9b": 9.16,
    "solar-10.7b": 10.7,
    "gemma-3-12b": 12.2,
    "ministral-14b": 14.0,
    "gpt-oss-20b": 20.0,
    "mistral-small-24b": 24.0,
    "gemma-2-27b": 27.0,
    "gemma-3-27b": 27.2,
    "seed-oss-36b": 36.0,
}


def main():
    print("=" * 70)
    print("MULTIPLICATIVE ERROR MODEL TEST (Proposition 3)")
    print("=" * 70)

    with open(ANALYSIS_DIR / "full_21model_analysis.json") as f:
        analysis = json.load(f)

    results = {"per_model": {}, "aggregate": {}}

    # Test 1: Log-normality of severity scores per model
    print("\n--- Test 1: Log-normality of severity scores ---")
    log_normality = []

    for model_name, data in sorted(analysis.items()):
        dist = data.get("distribution", {})
        if not dist:
            continue

        # Reconstruct severity scores from distribution counts
        scores = []
        for score_str, count in dist.items():
            s = float(score_str)
            if s > 0:
                scores.extend([s] * count)

        if len(scores) < 20:
            continue

        scores_arr = np.array(scores)
        log_scores = np.log(scores_arr)

        # Shapiro-Wilk test on log(scores) - tests if log(S) is normal
        # Use subsample if n > 5000 (Shapiro-Wilk limit)
        if len(log_scores) > 5000:
            rng = np.random.default_rng(42)
            log_sub = rng.choice(log_scores, 5000, replace=False)
        else:
            log_sub = log_scores

        sw_stat, sw_p = stats.shapiro(log_sub)

        # Also compute skewness and kurtosis of log(scores)
        skew = float(stats.skew(log_scores))
        kurt = float(stats.kurtosis(log_scores))

        # Log-variance (sigma^2 in the multiplicative model)
        log_var = float(np.var(log_scores))
        log_std = float(np.std(log_scores))

        best_fit = data.get("best_fit", {}).get("distribution", "unknown")
        b_val = data["b_value"]["b"]

        model_result = {
            "n_errors": len(scores),
            "shapiro_wilk_stat": round(sw_stat, 4),
            "shapiro_wilk_p": round(sw_p, 6),
            "log_normal_plausible": sw_p > 0.01,
            "log_skewness": round(skew, 4),
            "log_kurtosis": round(kurt, 4),
            "log_variance": round(log_var, 4),
            "log_std": round(log_std, 4),
            "best_fit_family": best_fit,
            "b_value": round(b_val, 4),
            "layers": MODEL_LAYERS.get(model_name),
        }
        results["per_model"][model_name] = model_result
        log_normality.append(model_result)

        tag = "LN_OK" if sw_p > 0.01 else "NOT_LN"
        print(f"  {model_name:30s} SW_p={sw_p:.4f} [{tag}] "
              f"log_var={log_var:.3f} b={b_val:.3f} fit={best_fit}")

    n_log_normal = sum(1 for r in log_normality if r["log_normal_plausible"])
    print(f"\n  Log-normal plausible: {n_log_normal}/{len(log_normality)} models")

    # Test 2: Correlation between layer count and log-variance (dense only)
    print("\n--- Test 2: Layer count vs log-variance (dense models) ---")
    layers = []
    log_vars = []
    b_vals = []
    dense_models = []

    for model_name, mres in results["per_model"].items():
        if model_name in MODEL_PARAMS and mres.get("layers"):
            layers.append(mres["layers"])
            log_vars.append(mres["log_variance"])
            b_vals.append(mres["b_value"])
            dense_models.append(model_name)

    if len(layers) >= 5:
        layers_arr = np.array(layers)
        logvars_arr = np.array(log_vars)
        bvals_arr = np.array(b_vals)

        # Prediction: more layers → larger log-variance → smaller b
        rho_layers_logvar, p_layers_logvar = stats.spearmanr(layers_arr, logvars_arr)
        rho_layers_b, p_layers_b = stats.spearmanr(layers_arr, bvals_arr)
        rho_logvar_b, p_logvar_b = stats.spearmanr(logvars_arr, bvals_arr)

        # b should be proportional to 1/sqrt(log_variance) under the model
        predicted_b_rank = 1.0 / np.sqrt(logvars_arr + 1e-10)
        rho_predicted, p_predicted = stats.spearmanr(predicted_b_rank, bvals_arr)

        print(f"  n_dense = {len(layers)}")
        print(f"  rho(layers, log_var)   = {rho_layers_logvar:.4f} (p={p_layers_logvar:.4f})")
        print(f"  rho(layers, b)         = {rho_layers_b:.4f} (p={p_layers_b:.4f})")
        print(f"  rho(log_var, b)        = {rho_logvar_b:.4f} (p={p_logvar_b:.4f})")
        print(f"  rho(1/sqrt(log_var), b)= {rho_predicted:.4f} (p={p_predicted:.4f}) <- model prediction")

        results["aggregate"]["dense_layer_analysis"] = {
            "n_dense": len(layers),
            "rho_layers_logvar": round(rho_layers_logvar, 4),
            "p_layers_logvar": round(p_layers_logvar, 6),
            "rho_layers_b": round(rho_layers_b, 4),
            "p_layers_b": round(p_layers_b, 6),
            "rho_logvar_b": round(rho_logvar_b, 4),
            "p_logvar_b": round(p_logvar_b, 6),
            "rho_predicted_b_actual_b": round(rho_predicted, 4),
            "p_predicted": round(p_predicted, 6),
            "prediction_matches": rho_logvar_b < -0.3,
            "note": (
                "Multiplicative model predicts: more layers -> larger log-variance -> smaller b. "
                f"Observed: rho(log_var, b) = {rho_logvar_b:.3f}. "
                f"{'Directionally consistent.' if rho_logvar_b < 0 else 'Not consistent.'}"
            ),
        }

    # Test 3: Do lognormal-best-fit models show better log-normality?
    print("\n--- Test 3: Lognormal-fit vs others on Shapiro-Wilk ---")
    ln_fits = [r for r in log_normality if r["best_fit_family"] in ("lognormal", "stretched_exp")]
    other_fits = [r for r in log_normality if r["best_fit_family"] not in ("lognormal", "stretched_exp")]

    ln_sw = [r["shapiro_wilk_p"] for r in ln_fits]
    other_sw = [r["shapiro_wilk_p"] for r in other_fits]

    if ln_sw and other_sw:
        mw_stat, mw_p = stats.mannwhitneyu(ln_sw, other_sw, alternative="greater")
        print(f"  Lognormal/stretched exp fits (n={len(ln_sw)}): median SW_p = {np.median(ln_sw):.4f}")
        print(f"  Other fits (n={len(other_sw)}): median SW_p = {np.median(other_sw):.4f}")
        print(f"  Mann-Whitney U test (LN > other): U={mw_stat:.1f}, p={mw_p:.4f}")

        results["aggregate"]["lognormal_vs_other_shapiro"] = {
            "n_lognormal_fits": len(ln_sw),
            "n_other_fits": len(other_sw),
            "median_sw_p_lognormal": round(float(np.median(ln_sw)), 4),
            "median_sw_p_other": round(float(np.median(other_sw)), 4),
            "mann_whitney_p": round(mw_p, 4),
        }

    results["aggregate"]["log_normality_summary"] = {
        "n_models_tested": len(log_normality),
        "n_log_normal_plausible": n_log_normal,
        "fraction_log_normal": round(n_log_normal / len(log_normality), 3),
    }

    # Save
    output_path = OUTPUT_DIR / "multiplicative_model_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
