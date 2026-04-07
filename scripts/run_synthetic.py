"""CLI for Experiment 0 synthetic validation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from errorquake.synthetic import DEFAULT_DISTRIBUTIONS, default_synthetic_config, run_experiment_0


@click.command()
@click.option("--n", type=int, default=500, show_default=True, help="Items per distribution.")
@click.option(
    "--distributions",
    type=str,
    default=",".join(DEFAULT_DISTRIBUTIONS),
    show_default=True,
    help="Comma-separated synthetic distributions.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/synthetic"),
    show_default=True,
    help="Synthetic artifact directory.",
)
@click.option("--rpm", type=int, default=35, show_default=True, help="Sequential rate limit.")
@click.option("--skip-scoring", is_flag=True, help="Generate responses only, skip judging.")
@click.option("--score-only", is_flag=True, help="Score existing synthetic responses only.")
@click.option("--analyze-only", is_flag=True, help="Analyze existing scores only.")
@click.option("--resume", is_flag=True, help="Resume from checkpoint files.")
def main(
    n: int,
    distributions: str,
    output_dir: Path,
    rpm: int,
    skip_scoring: bool,
    score_only: bool,
    analyze_only: bool,
    resume: bool,
) -> None:
    """Run Experiment 0 synthetic validation."""
    config = default_synthetic_config()
    distribution_list = [item.strip() for item in distributions.split(",") if item.strip()]
    summary = asyncio.run(
        run_experiment_0(
            config,
            n=n,
            distributions=distribution_list,
            output_dir=output_dir,
            rpm=rpm,
            skip_scoring=skip_scoring,
            score_only=score_only,
            analyze_only=analyze_only,
            resume=resume,
        )
    )
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
