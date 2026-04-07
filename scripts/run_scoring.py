"""CLI for scoring model responses."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from errorquake.queries import load_queries
from errorquake.score import ScoringPipeline
from errorquake.utils import ProjectConfig


def _iter_response_files(responses_dir: Path, models: str | None) -> list[Path]:
    files = sorted(responses_dir.glob("*.jsonl"))
    if not models:
        return files
    names = {item.strip() for item in models.split(",") if item.strip()}
    return [path for path in files if path.stem in names]


@click.command()
@click.option("--models", default=None, help="Models to score (default: all with results)")
@click.option("--responses-dir", type=click.Path(path_type=Path), default=Path("results/evaluations"))
@click.option("--queries-dir", type=click.Path(path_type=Path), default=Path("data/queries"))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("results/scores"))
@click.option("--scale", default="11-point", help='"11-point", "7-point", or "5-level"')
def main(
    models: str | None,
    responses_dir: Path,
    queries_dir: Path,
    output_dir: Path,
    scale: str,
) -> None:
    """Score evaluation outputs using the dual-judge pipeline."""
    queries = load_queries(queries_dir.parent if queries_dir.name == "queries" else queries_dir)
    pipeline = ScoringPipeline(ProjectConfig())
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for response_file in _iter_response_files(responses_dir, models):
        output_path = output_dir / response_file.name
        summary[response_file.stem] = asyncio.run(
            pipeline.score_responses(response_file, queries, output_path, scale_name=scale)
        )
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

