"""CLI for model evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from errorquake.evaluate import ALL_MODELS, EvaluationEngine, verify_model_access
from errorquake.queries import load_queries
from errorquake.utils import ProjectConfig


def _select_models(value: str | None):
    if not value:
        return ALL_MODELS
    names = {item.strip() for item in value.split(",") if item.strip()}
    return [model for model in ALL_MODELS if model.name in names]


@click.command()
@click.option("--models", default=None, help="Comma-separated model names (default: all)")
@click.option("--queries-dir", type=click.Path(path_type=Path), default=Path("data/queries"))
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("results/evaluations"))
@click.option("--concurrency", type=int, default=10, help="Concurrent requests per model")
@click.option("--verify-access", is_flag=True, help="Only test API access, don't evaluate")
def main(
    models: str | None,
    queries_dir: Path,
    output_dir: Path,
    concurrency: int,
    verify_access: bool,
) -> None:
    """Run model evaluation or API access verification."""
    selected_models = _select_models(models)
    if verify_access:
        report = asyncio.run(verify_model_access(selected_models, attempt_live=True))
        click.echo(json.dumps(report, indent=2))
        return

    config = ProjectConfig(eval_concurrency=concurrency)
    queries = load_queries(queries_dir.parent if queries_dir.name == "queries" else queries_dir)
    if not queries:
        raise click.ClickException(f"No queries found in {queries_dir}")
    engine = EvaluationEngine(config)
    results = asyncio.run(engine.evaluate_batch(selected_models, queries, output_dir))
    click.echo(json.dumps({name: str(path) for name, path in results.items()}, indent=2))


if __name__ == "__main__":
    main()
