"""Experiment 5: Scale dependence (b vs parameter count).

Spearman rho between log(active params) and b-value, separately for
dense and MoE models. Pre-registered hypothesis: NO clean correlation
(parameter count alone does not predict tail shape).

Output: results/analysis/exp5_scaling.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ANALYSIS = Path("C:/projects/errorquake/results/analysis/full_21model_analysis.json")
OUT = Path("C:/projects/errorquake/results/analysis/exp5_scaling.json")

# Active param counts (billions). For MoE: active params per token.
PARAMS = {
    "gemma-3-4b":            (4.0,    "dense"),
    "llama-3.2-3b-instruct": (3.2,    "dense"),
    "phi-3.5-mini":          (3.8,    "dense"),
    "qwen2.5-7b":            (7.6,    "dense"),
    "llama-3.1-8b-instruct": (8.0,    "dense"),
    "eurollm-9b":            (9.2,    "dense"),
    "solar-10.7b":           (10.7,   "dense"),
    "gemma-3-12b":           (12.0,   "dense"),
    "ministral-14b":         (14.0,   "dense"),  # ministral-8B but listed as 14b in our catalog
    "gpt-oss-20b":           (3.6,    "moe"),    # ~3.6B active of 21B total
    "mistral-small-24b":     (24.0,   "dense"),
    "gemma-2-27b":           (27.0,   "dense"),
    "gemma-3-27b":           (27.0,   "dense"),
    "seed-oss-36b":          (36.0,   "dense"),
    "llama-4-maverick":      (17.0,   "moe"),    # 17B active of 400B total
    "qwen3-next-80b":        (3.0,    "moe"),    # 3B active of 80B total
    "mistral-small-4-119b":  (16.0,   "moe"),    # ~16B active per layer
    "deepseek-v3.1":         (37.0,   "moe"),    # 37B active of 671B
    "deepseek-v3.2":         (37.0,   "moe"),    # 37B active of 671B
    "kimi-k2-instruct":      (32.0,   "moe"),    # 32B active of 1T
    "mistral-medium-3":      (24.0,   "dense"),  # rumored ~24B dense
}


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 5: Scale dependence (b vs parameter count)")
    print("=" * 70)

    data = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    points = []
    for name, d in data.items():
        if name not in PARAMS:
            print(f"  WARN: no param info for {name}")
            continue
        active_b, arch = PARAMS[name]
        bv = d["b_value"]["b"]
        err = d["error_rate"]
        points.append({
            "name": name,
            "active_params_b": active_b,
            "log_params": float(np.log10(active_b * 1e9)),
            "architecture": arch,
            "b_value": float(bv),
            "error_rate": float(err),
        })

    points.sort(key=lambda p: p["active_params_b"])

    print(f"\n{'Model':<28} {'arch':<6} {'params(B)':>10} {'b':>7} {'err':>7}")
    print("-" * 60)
    for p in points:
        print(f"{p['name']:<28} {p['architecture']:<6} {p['active_params_b']:>10.1f} "
              f"{p['b_value']:>7.3f} {p['error_rate']:>7.3f}")

    def corr(subset, label):
        if len(subset) < 3:
            return None
        log_p = np.array([p["log_params"] for p in subset])
        bs = np.array([p["b_value"] for p in subset])
        rho, p_rho = stats.spearmanr(log_p, bs)
        pearson_r, p_r = stats.pearsonr(log_p, bs)
        return {
            "label": label,
            "n": len(subset),
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r": float(pearson_r),
            "pearson_p": float(p_r),
        }

    all_corr = corr(points, "all")
    dense_corr = corr([p for p in points if p["architecture"] == "dense"], "dense")
    moe_corr = corr([p for p in points if p["architecture"] == "moe"], "moe")

    print()
    print("=" * 70)
    print("CORRELATIONS")
    print("=" * 70)
    for c in (all_corr, dense_corr, moe_corr):
        if c is None:
            continue
        print(f"  {c['label']:<8} (n={c['n']:>2}) "
              f"Spearman rho={c['spearman_rho']:+.3f} (p={c['spearman_p']:.3f})  "
              f"Pearson r={c['pearson_r']:+.3f} (p={c['pearson_p']:.3f})")

    out = {
        "points": points,
        "correlations": {
            "all": all_corr,
            "dense": dense_corr,
            "moe": moe_corr,
        },
        "interpretation": (
            "Pre-registered prediction was NO clean correlation. "
            "Spearman rho on dense models is the headline number."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
