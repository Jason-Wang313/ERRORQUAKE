"""Master figure generator for ERRORQUAKE Phase 5.

Generates Figures 1, 2, 3, 5, 6, 7, 8, 10, 12, 13, 14.
(Fig 4 from make_fig4.py, Fig 9 from magnitude.py, Fig 11 from synthetic data.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.report import set_errorquake_style

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
PAPER_DIR = PROJECT_ROOT / "paper" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
GRID = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    fig.savefig(PAPER_DIR / f"{name}.pdf")
    print(f"  saved {name}")
    plt.close(fig)


def load_scores(stem: str) -> np.ndarray:
    out = []
    for line in open(SCORES_DIR / f"{stem}.jsonl", encoding="utf-8"):
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


def cumulative(scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.array([(scores >= g).sum() for g in grid], dtype=float)


# ----------------------------------------------------------------------
# FIGURE 1: Flagship — 4 models, magnitude-frequency with fits
# ----------------------------------------------------------------------
def fig1_flagship(analysis: dict) -> None:
    set_errorquake_style()
    flagship = ["seed-oss-36b", "deepseek-v3.2", "llama-3.1-8b-instruct", "qwen2.5-7b"]
    titles = [
        "seed-oss-36b (b=0.57, heaviest)",
        "deepseek-v3.2 (b=0.66, frontier)",
        "llama-3.1-8b (b=1.00, mid)",
        "qwen2.5-7b (b=1.26, lightest)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    for ax, name, title in zip(axes.flat, flagship, titles):
        scores = load_scores(name)
        cum = cumulative(scores, GRID)
        ax.plot(GRID, cum, "o-", color="#1f77b4", lw=1.5, ms=6, label="observed N(M$\\geq$m)")

        bv = analysis[name]["b_value"]
        b = bv["b"]
        m_min = bv["m_min"]
        n_above = bv["n_above_mmin"]
        x = np.linspace(m_min, 4.0, 30)
        y = n_above * 10 ** (-b * (x - m_min))
        ax.plot(x, y, "r--", lw=1.4, label=f"GR fit  b={b:.2f}")

        ax.set_yscale("log")
        ax.set_xlabel("severity magnitude $m$")
        ax.set_ylabel("$N(M \\geq m)$")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.3, 4.2)
        ax.set_ylim(max(0.5, cum.min() / 2), cum.max() * 1.5)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.suptitle("Figure 1: Magnitude-frequency curves (4 representative models)",
                 fontsize=12, fontweight="bold")
    save(fig, "fig1_flagship")


# ----------------------------------------------------------------------
# FIGURE 2: All 21 models — small multiples
# ----------------------------------------------------------------------
def fig2_grid(analysis: dict) -> None:
    set_errorquake_style()
    models = sorted(analysis.keys(),
                    key=lambda n: analysis[n]["b_value"]["b"])
    n_cols = 5
    n_rows = (len(models) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.0 * n_rows),
                              constrained_layout=True)
    for ax, name in zip(axes.flat, models):
        scores = load_scores(name)
        cum = cumulative(scores, GRID)
        ax.plot(GRID, cum, "o-", color="#1f77b4", lw=1.0, ms=3)
        b = analysis[name]["b_value"]["b"]
        m_min = analysis[name]["b_value"]["m_min"]
        n_above = analysis[name]["b_value"]["n_above_mmin"]
        x = np.linspace(m_min, 4.0, 30)
        y = n_above * 10 ** (-b * (x - m_min))
        ax.plot(x, y, "r--", lw=1.0)
        ax.set_yscale("log")
        ax.set_title(f"{name}\nb={b:.2f}", fontsize=7)
        ax.set_xlim(0.3, 4.2)
        ax.tick_params(labelsize=6)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    # Blank unused
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    fig.suptitle("Figure 2: Magnitude-frequency for all 21 models (sorted by b)",
                 fontsize=12, fontweight="bold")
    save(fig, "fig2_grid")


# ----------------------------------------------------------------------
# FIGURE 3: BIC heatmap
# ----------------------------------------------------------------------
def fig3_bic_heatmap(analysis: dict) -> None:
    set_errorquake_style()
    dists = ["power_law", "truncated_power_law", "lognormal",
             "exponential", "stretched_exp"]
    models = sorted(analysis.keys(), key=lambda n: analysis[n]["b_value"]["b"])

    matrix = np.full((len(models), len(dists)), np.nan)
    for i, m in enumerate(models):
        fits = {f["distribution"]: f["bic"] for f in analysis[m].get("all_fits", [])}
        # Compute delta-BIC relative to best
        valid = [v for v in fits.values() if v is not None and v != float("inf")]
        if not valid:
            continue
        best = min(valid)
        for j, d in enumerate(dists):
            if d in fits and fits[d] is not None and fits[d] != float("inf"):
                matrix[i, j] = fits[d] - best

    fig, ax = plt.subplots(figsize=(7, 9))
    cap = 60
    masked = np.minimum(matrix, cap)
    im = ax.imshow(masked, cmap="viridis_r", aspect="auto", vmin=0, vmax=cap)
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([d.replace("_", " ") for d in dists], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m} (b={analysis[m]['b_value']['b']:.2f})" for m in models],
                        fontsize=8)
    # Mark best fit per row with a star
    for i, m in enumerate(models):
        row = matrix[i]
        if not np.all(np.isnan(row)):
            j = int(np.nanargmin(row))
            ax.text(j, i, "*", ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("ΔBIC vs best (capped 60)")
    ax.set_title("Figure 3: ΔBIC by distribution family\n(* = best fit per model)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_bic_heatmap")


# ----------------------------------------------------------------------
# FIGURE 5: b-value heatmap (21 models x 8 domains)
# ----------------------------------------------------------------------
def fig5_domain_heatmap() -> None:
    set_errorquake_style()
    data = json.loads((ANALYSIS_DIR / "exp4_domains.json").read_text(encoding="utf-8"))
    matrix_data = data["matrix"]
    domains = data["domains"]
    models = sorted(matrix_data.keys(),
                    key=lambda m: np.nanmean([matrix_data[m][d]["b"]
                                              for d in domains
                                              if matrix_data[m][d]["b"] is not None]))

    arr = np.full((len(models), len(domains)), np.nan)
    flag = np.zeros_like(arr, dtype=bool)
    for i, m in enumerate(models):
        for j, d in enumerate(domains):
            cell = matrix_data[m][d]
            if cell["b"] is not None:
                arr[i, j] = cell["b"]
                flag[i, j] = cell.get("n_errors", 0) < 50

    fig, ax = plt.subplots(figsize=(7.5, 9))
    sns.heatmap(arr, ax=ax,
                cmap="RdYlBu_r", center=1.0, vmin=0.4, vmax=1.6,
                xticklabels=domains, yticklabels=models,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"label": "b-value"})
    ax.set_title("Figure 5: b-value by model × domain (red = heavy tail)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("domain")
    ax.set_ylabel("model (sorted by mean b)")
    fig.tight_layout()
    save(fig, "fig5_domain_heatmap")


# ----------------------------------------------------------------------
# FIGURE 6: b vs scale
# ----------------------------------------------------------------------
def fig6_scale() -> None:
    set_errorquake_style()
    data = json.loads((ANALYSIS_DIR / "exp5_scaling.json").read_text(encoding="utf-8"))
    points = data["points"]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"dense": "#1f77b4", "moe": "#d62728"}
    for arch in ("dense", "moe"):
        sub = [p for p in points if p["architecture"] == arch]
        xs = [p["log_params"] for p in sub]
        ys = [p["b_value"] for p in sub]
        ax.scatter(xs, ys, s=70, c=colors[arch], alpha=0.85,
                   edgecolor="white", linewidth=0.7, label=arch, zorder=3)

    # Linear fit on dense
    dense = [p for p in points if p["architecture"] == "dense"]
    if len(dense) >= 3:
        xs = np.array([p["log_params"] for p in dense])
        ys = np.array([p["b_value"] for p in dense])
        slope, intercept = np.polyfit(xs, ys, 1)
        xx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xx, slope * xx + intercept, color=colors["dense"], lw=1.2, alpha=0.6,
                label=f"dense fit (slope={slope:.2f})")

    rho_d = data["correlations"]["dense"]["spearman_rho"]
    p_d = data["correlations"]["dense"]["spearman_p"]
    ax.text(0.04, 0.06,
            f"dense $\\rho_s$ = {rho_d:+.3f}\np = {p_d:.3f}",
            transform=ax.transAxes,
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))

    # Annotate model names
    for p in points:
        ax.annotate(p["name"].replace("-instruct", "").replace("-instruct", ""),
                    (p["log_params"], p["b_value"]),
                    fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("$\\log_{10}$(active parameters)")
    ax.set_ylabel("Gutenberg-Richter $b$-value")
    ax.set_title("Figure 6: b-value vs model scale (active params)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)
    save(fig, "fig6_scale")


# ----------------------------------------------------------------------
# FIGURE 7: Gemma-2 vs Gemma-3 27B
# ----------------------------------------------------------------------
def fig7_gemma_pair(analysis: dict) -> None:
    set_errorquake_style()
    pairs = [("gemma-2-27b", "gemma-3-27b")]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#d62728"]
    for name, color in zip(pairs[0], colors):
        scores = load_scores(name)
        cum = cumulative(scores, GRID)
        b = analysis[name]["b_value"]["b"]
        ax.plot(GRID, cum, "o-", color=color, lw=1.5, ms=6,
                label=f"{name}  (b = {b:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("severity magnitude $m$")
    ax.set_ylabel("$N(M \\geq m)$")
    ax.set_title("Figure 7: Gemma-2 vs Gemma-3 (27B) — generation comparison",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    save(fig, "fig7_gemma_pair")


# ----------------------------------------------------------------------
# FIGURE 8: DeepSeek v3.1 vs v3.2
# ----------------------------------------------------------------------
def fig8_deepseek(analysis: dict) -> None:
    set_errorquake_style()
    pairs = [("deepseek-v3.1", "deepseek-v3.2")]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4", "#d62728"]
    for name, color in zip(pairs[0], colors):
        scores = load_scores(name)
        cum = cumulative(scores, GRID)
        b = analysis[name]["b_value"]["b"]
        ax.plot(GRID, cum, "o-", color=color, lw=1.5, ms=6,
                label=f"{name}  (b = {b:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("severity magnitude $m$")
    ax.set_ylabel("$N(M \\geq m)$")
    ax.set_title("Figure 8: DeepSeek v3.1 vs v3.2 — version comparison",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    save(fig, "fig8_deepseek")


# ----------------------------------------------------------------------
# FIGURE 10: Deployment table — expected catastrophic per 1M queries
# ----------------------------------------------------------------------
def fig10_deployment(analysis: dict) -> None:
    set_errorquake_style()
    rows = []
    for name, d in analysis.items():
        n_total = d["n_total"]
        n_errors = d["n_errors"]
        # Use the empirical catastrophe rate from the dataset (>=3.0)
        # Plus the extrapolated rate from b-value
        scores = load_scores(name)
        n_cat = int((scores >= 3.0).sum())
        n_severe = int((scores >= 2.5).sum())
        rate_cat_per_1M = n_cat / n_total * 1e6
        rate_sev_per_1M = n_severe / n_total * 1e6
        rows.append({
            "name": name,
            "b": d["b_value"]["b"],
            "err": d["error_rate"],
            "cat_per_1M": rate_cat_per_1M,
            "sev_per_1M": rate_sev_per_1M,
        })
    rows.sort(key=lambda r: r["cat_per_1M"], reverse=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    names = [r["name"] for r in rows]
    cats = [r["cat_per_1M"] for r in rows]
    sevs = [r["sev_per_1M"] for r in rows]

    y = np.arange(len(rows))
    ax.barh(y - 0.18, sevs, height=0.35, color="#ff9c1a", label="severe (M≥2.5)")
    ax.barh(y + 0.18, cats, height=0.35, color="#d62728", label="catastrophic (M≥3.0)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  (b={r['b']:.2f}, ε={r['err']:.2f})"
                        for n, r in zip(names, rows)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("expected events per 1,000,000 queries")
    ax.set_title("Figure 10: Deployment risk per million queries (empirical rates)",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "fig10_deployment")


# ----------------------------------------------------------------------
# FIGURE 12: Overcall diagnostic
# ----------------------------------------------------------------------
def fig12_overcall() -> None:
    set_errorquake_style()
    path = ANALYSIS_DIR / "overcall_diagnostic.json"
    if not path.exists():
        print("  fig12: overcall_diagnostic.json missing, skipping")
        return
    data = json.loads(path.read_text(encoding="utf-8"))

    per_model = data.get("per_model", {})
    overall = data.get("overall", {})
    if not per_model:
        print("  fig12: no per_model data, skipping")
        return

    names = sorted(per_model.keys(), key=lambda n: per_model[n]["overcall_rate"])
    genuine = [per_model[n]["genuine_rate"] for n in names]
    ambiguous = [per_model[n]["ambiguous_rate"] for n in names]
    overcall = [per_model[n]["overcall_rate"] for n in names]

    fig, ax = plt.subplots(figsize=(8, max(5, 0.32 * len(names))))
    y = np.arange(len(names))
    g = np.array(genuine)
    a = np.array(ambiguous)
    o = np.array(overcall)
    ax.barh(y, g, color="#2ca02c", label="genuine 2.0", edgecolor="white")
    ax.barh(y, a, left=g, color="#cccc44", label="ambiguous", edgecolor="white")
    ax.barh(y, o, left=g + a, color="#d62728", label="overcall", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of score-2.0 judgments (manual classification)")
    ax.axvline(1 - overall.get("overall_overcall_rate", 0.335), color="black",
               ls="--", lw=0.8, alpha=0.6,
               label=f"overall genuine+amb = {1 - overall.get('overall_overcall_rate', 0.335):.0%}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        f"Figure 12: Score-2.0 overcall diagnostic ({overall.get('n_total_sampled', 340)} items, "
        f"{overall.get('overall_overcall_rate', 0.335):.0%} overcall)",
        fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    save(fig, "fig12_overcall")


# ----------------------------------------------------------------------
# FIGURES 13 + 14: Sensitivity
# ----------------------------------------------------------------------
def fig13_14_sensitivity(analysis: dict) -> None:
    set_errorquake_style()
    sens_path = ANALYSIS_DIR / "sensitivity.json"
    if not sens_path.exists():
        print("  fig13/14: sensitivity.json missing yet, skipping")
        return
    sens = json.loads(sens_path.read_text(encoding="utf-8"))

    # Fig 13: scale sensitivity scatter
    s1 = sens["S1_scale"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    names = list(s1["per_model"].keys())
    nine = [s1["per_model"][n]["9pt"] for n in names]
    seven = [s1["per_model"][n]["7pt"] for n in names]
    five = [s1["per_model"][n]["5lvl"] for n in names]
    ax.scatter(nine, seven, s=60, c="#1f77b4", label=f"7-pt (ρ={s1['spearman_9pt_to_7pt']:.3f})", alpha=0.8)
    ax.scatter(nine, five, s=60, c="#d62728", label=f"5-level (ρ={s1['spearman_9pt_to_5lvl']:.3f})", alpha=0.8)
    lims = (min(nine + seven + five) * 0.9, max(nine + seven + five) * 1.05)
    ax.plot(lims, lims, "k-", lw=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("b-value (9-point scale)")
    ax.set_ylabel("b-value (coarsened scale)")
    ax.set_title("Figure 13: b-value ranking is preserved under scale coarsening (S1)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)
    save(fig, "fig13_scale_sensitivity")

    # Fig 14: overcall correction histogram of bootstrap rho
    s2 = sens["S2_overcall"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # We don't have per-trial rhos in the JSON; use min/mean/max bars
    labels = ["min", "mean", "max"]
    values = [s2["min_spearman"], s2["mean_spearman"], s2["max_spearman"]]
    ax.bar(labels, values, color=["#d62728", "#1f77b4", "#2ca02c"], edgecolor="white")
    for i, v in enumerate(values):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0.7, 1.02)
    ax.axhline(0.85, color="black", ls="--", lw=0.8, label="0.85 threshold")
    ax.set_ylabel("Spearman ρ vs original ranking")
    ax.set_title(f"Figure 14: Overcall correction ranking stability (S2, n_trials={s2['n_trials']})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    save(fig, "fig14_overcall_sensitivity")


def main() -> None:
    print("=" * 70)
    print("FIGURE GENERATION")
    print("=" * 70)
    analysis = json.loads((ANALYSIS_DIR / "full_21model_analysis.json").read_text(encoding="utf-8"))
    fig1_flagship(analysis)
    fig2_grid(analysis)
    fig3_bic_heatmap(analysis)
    fig5_domain_heatmap()
    fig6_scale()
    fig7_gemma_pair(analysis)
    fig8_deepseek(analysis)
    fig10_deployment(analysis)
    fig12_overcall()
    fig13_14_sensitivity(analysis)
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
