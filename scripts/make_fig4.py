"""Figure 4: Predicted vs observed catastrophic count, two-panel.

Left:  M >= 3.0 (pre-registered primary)
Right: M >= 2.5 (secondary, exploratory)

Each panel shows 21 model points, 1:1 reference line, 1.5x bands,
and the headline rho/within-1.5x stats.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results" / "analysis" / "exp3_prediction.json"
OUT_DIR = PROJECT_ROOT / "results" / "figures"
PAPER_DIR = PROJECT_ROOT / "paper" / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)


def panel(ax, target_data: dict, title: str) -> None:
    rows = [r for r in target_data["rows"] if "predicted" in r and "observed" in r]
    pred = np.array([max(r["predicted"], 0.5) for r in rows])  # 0.5 for log axis
    obs = np.array([max(r["observed"], 0.5) for r in rows])

    ax.scatter(obs, pred, s=70, c="#1f77b4", alpha=0.85, edgecolor="white", linewidth=0.7, zorder=3)

    # 1:1 reference
    lo = min(pred.min(), obs.min()) * 0.7
    hi = max(pred.max(), obs.max()) * 1.5
    ref = np.array([lo, hi])
    ax.plot(ref, ref, "k-", lw=1.0, label="1:1 line")
    ax.plot(ref, ref * 1.5, "k--", lw=0.7, alpha=0.6)
    ax.plot(ref, ref / 1.5, "k--", lw=0.7, alpha=0.6, label="±1.5×")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed catastrophic count")
    ax.set_ylabel("Predicted catastrophic count")

    rho = target_data["spearman_rho_counts"]
    p = target_data["spearman_p_counts"]
    n = target_data["n_valid"]
    frac = target_data["within_1_5x_fraction"]

    ax.text(
        0.04, 0.96,
        f"$\\rho_s$ = {rho:.3f}\n$p$ = {p:.3f}\n$n$ = {n}\nwithin 1.5×: {frac:.0%}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7"),
    )
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    panel(axes[0], data["target_3.0"], "(a) Catastrophic threshold M $\\geq$ 3.0 (pre-registered)")
    panel(axes[1], data["target_2.5"], "(b) Catastrophic threshold M $\\geq$ 2.5 (secondary)")

    fig.suptitle(
        "Figure 4: Predicting catastrophic-error counts from easy-tier b-values",
        fontsize=12, fontweight="bold",
    )

    for ext, root in [("png", OUT_DIR), ("pdf", PAPER_DIR)]:
        out = root / f"fig4_prediction_calibration.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  saved {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
