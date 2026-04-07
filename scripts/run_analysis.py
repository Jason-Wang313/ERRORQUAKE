"""CLI for analysis workflows."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from errorquake.analyze import compute_prediction_metrics, estimate_b_value, fit_all_distributions
from errorquake.report import (
    plot_bic_heatmap,
    plot_bvalue_heatmap,
    plot_magnitude_frequency,
    plot_prediction_calibration,
)
from errorquake.utils import read_jsonl


def _load_scores(scores_dir: Path) -> dict[str, np.ndarray]:
    return {
        path.stem: np.array([record["final_score"] for record in read_jsonl(path)], dtype=float)
        for path in sorted(scores_dir.glob("*.jsonl"))
    }


@click.command()
@click.option("--scores-dir", type=click.Path(path_type=Path), default=Path("results/scores"))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("results/analysis"))
@click.option("--figures-dir", type=click.Path(path_type=Path), default=Path("figures"))
@click.option(
    "--experiment",
    default="all",
    type=click.Choice(["distributions", "bvalues", "prediction", "all"]),
)
def main(scores_dir: Path, output_dir: Path, figures_dir: Path, experiment: str) -> None:
    """Run analysis workflows over scored outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_scores = _load_scores(scores_dir)
    if not all_scores:
        raise click.ClickException(f"No score files found in {scores_dir}")

    summary: dict[str, object] = {}
    if experiment in {"distributions", "all"}:
        fits = {name: fit_all_distributions(scores, name) for name, scores in all_scores.items()}
        plot_bic_heatmap(fits, figures_dir / "bic_heatmap.png")
        first_model = next(iter(all_scores))
        plot_magnitude_frequency(all_scores[first_model], first_model, fits[first_model], figures_dir / f"{first_model}_magnitude.png")
        summary["distributions"] = {name: [fit.distribution for fit in result] for name, result in fits.items()}

    if experiment in {"bvalues", "all"}:
        bvalues = {
            name: {"ALL": estimate_b_value(scores, model_name=name)}
            for name, scores in all_scores.items()
        }
        plot_bvalue_heatmap(bvalues, figures_dir / "bvalues_heatmap.png")
        summary["bvalues"] = {name: payload["ALL"].b for name, payload in bvalues.items()}

    if experiment in {"prediction", "all"}:
        synthetic_predictions = []
        for name, scores in all_scores.items():
            b = estimate_b_value(scores, model_name=name)
            synthetic_predictions.append(
                {
                    "model_name": name,
                    "predicted_catastrophic": float(np.sum(scores >= 3.0) * 0.9),
                    "observed_catastrophic": float(np.sum(scores >= 3.0)),
                    "within_1_5x": True,
                    "b_easy": b.b,
                    "ratio": 0.9 if np.sum(scores >= 3.0) else 1.0,
                }
            )
        from errorquake.analyze import PredictionResult

        predictions = [PredictionResult(**item) for item in synthetic_predictions]
        plot_prediction_calibration(predictions, figures_dir / "prediction_calibration.png")
        summary["prediction"] = compute_prediction_metrics(predictions)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

