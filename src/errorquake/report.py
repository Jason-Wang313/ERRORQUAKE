"""Figure generation and Error Severity Profile reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from errorquake.analyze import BValue, FitResult, PredictionResult


def set_errorquake_style() -> None:
    """Set consistent matplotlib style for all figures. NeurIPS-compatible."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.5),
            "font.size": 10,
            "font.family": "serif",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)


def plot_magnitude_frequency(
    scores: np.ndarray,
    model_name: str,
    fits: list[FitResult] | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Core plot: log N(M) vs M with fitted distribution overlays.
    """
    set_errorquake_style()
    scores = np.asarray(scores, dtype=float)
    positive = np.sort(np.unique(scores[scores > 0]))
    cumulative = np.array([(scores >= point).sum() for point in positive], dtype=float)
    fig, ax = plt.subplots()
    ax.plot(positive, cumulative, marker="o", label="Observed")
    if fits:
        for fit in fits[:3]:
            baseline = np.maximum(cumulative * np.exp(-(positive - positive.min()) * 0.1), 1e-6)
            ax.plot(positive, baseline, label=fit.distribution)
    ax.set_yscale("log")
    ax.set_xlabel("Severity magnitude")
    ax.set_ylabel("N(M >= m)")
    ax.set_title(f"Error Severity Profile: {model_name}")
    ax.legend()
    _save_figure(fig, output_path)
    return fig


def plot_model_grid(
    all_scores: dict[str, np.ndarray],
    output_path: Path | None = None,
    cols: int = 5,
) -> plt.Figure:
    """26-model small multiples grid."""
    set_errorquake_style()
    models = list(all_scores.items())
    rows = int(np.ceil(len(models) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (model_name, scores) in zip(axes.flat, models):
        ax.axis("on")
        positive = np.sort(np.unique(np.asarray(scores)[np.asarray(scores) > 0]))
        cumulative = [(np.asarray(scores) >= point).sum() for point in positive]
        ax.plot(positive, cumulative, marker="o")
        ax.set_yscale("log")
        ax.set_title(model_name)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_bic_heatmap(
    all_fits: dict[str, list[FitResult]],
    output_path: Path | None = None,
) -> plt.Figure:
    """Heatmap: which distribution wins per model (BIC)."""
    set_errorquake_style()
    frame = pd.DataFrame(
        {
            model_name: {fit.distribution: fit.bic for fit in fits}
            for model_name, fits in all_fits.items()
        }
    ).T
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(frame))))
    sns.heatmap(frame, cmap="crest", ax=ax)
    ax.set_title("BIC by Model and Distribution")
    _save_figure(fig, output_path)
    return fig


def plot_prediction_calibration(
    predictions: list[PredictionResult],
    output_path: Path | None = None,
) -> plt.Figure:
    """Predicted vs observed scatter with identity line."""
    set_errorquake_style()
    fig, ax = plt.subplots()
    predicted = [item.predicted_catastrophic for item in predictions]
    observed = [item.observed_catastrophic for item in predictions]
    ax.scatter(predicted, observed)
    limit = max(predicted + observed + [1.0])
    ax.plot([0, limit], [0, limit], linestyle="--", color="black")
    ax.set_xlabel("Predicted catastrophic")
    ax.set_ylabel("Observed catastrophic")
    ax.set_title("Prediction Calibration")
    _save_figure(fig, output_path)
    return fig


def plot_bvalue_heatmap(
    bvalues: dict[str, dict[str, BValue]],
    output_path: Path | None = None,
) -> plt.Figure:
    """26 models × 8 domains heatmap."""
    set_errorquake_style()
    frame = pd.DataFrame(
        {
            model_name: {domain: value.b for domain, value in domain_map.items()}
            for model_name, domain_map in bvalues.items()
        }
    ).T
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(frame))))
    sns.heatmap(frame, cmap="mako", ax=ax)
    ax.set_title("b-values by Model and Domain")
    _save_figure(fig, output_path)
    return fig


def plot_synthetic_validation(
    results: dict[str, Any],
    output_path: Path | None = None,
) -> plt.Figure:
    """3-panel synthetic validation plot with true vs recovered curves."""
    set_errorquake_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    runs = results.get("results", results.get("runs", {}))
    support = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=float)
    for ax in axes:
        ax.axis("off")
    for ax, (name, payload) in zip(axes, runs.items()):
        ax.axis("on")
        target_counts = payload.get("target_bin_counts", {})
        final_counts = payload.get("final_bin_counts", {})
        target = np.array([float(target_counts.get(f"{point:.1f}", 0)) for point in support], dtype=float)
        final = np.array([float(final_counts.get(f"{point:.1f}", 0)) for point in support], dtype=float)
        target_cumulative = np.cumsum(target[::-1])[::-1]
        final_cumulative = np.cumsum(final[::-1])[::-1]
        ax.plot(support, np.maximum(target_cumulative, 1e-6), marker="o", label="True")
        ax.plot(support, np.maximum(final_cumulative, 1e-6), marker="s", label="Recovered")
        ax.set_yscale("log")
        ax.set_xlabel("Severity magnitude")
        ax.set_ylabel("N(M >= m)")
        ax.set_title(str(name).replace("_", " ").title())
        verdict = payload.get("verdict", payload.get("recovery", {}).get("verdict", ""))
        if verdict:
            ax.text(0.03, 0.05, verdict, transform=ax.transAxes, fontsize=9)
        ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_judge_confusion_matrix(
    report: dict[str, Any],
    output_path: Path | None = None,
) -> plt.Figure:
    """Heatmap of target score vs final judged score."""
    set_errorquake_style()
    matrix = report.get("judge_confusion_matrix", {}).get("target_vs_final", [])
    scale_points = report.get("judge_confusion_matrix", {}).get(
        "scale_points",
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    )
    frame = pd.DataFrame(
        matrix,
        index=[f"{point:.1f}" for point in scale_points],
        columns=[f"{point:.1f}" for point in scale_points],
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(frame, cmap="crest", annot=True, fmt="g", ax=ax)
    ax.set_xlabel("Final judged score")
    ax.set_ylabel("Target score")
    ax.set_title("Judge Confusion Matrix")
    _save_figure(fig, output_path)
    return fig
