"""CLI for query generation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from errorquake.generate import run_full_generation
from errorquake.queries import DOMAINS
from errorquake.utils import ProjectConfig


def _parse_domains(value: str | None) -> list[str]:
    if not value:
        return DOMAINS
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_tiers(value: str | None) -> list[int]:
    if not value:
        return [1, 2, 3, 4, 5]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@click.command()
@click.option("--domains", default=None, help="Comma-separated domains (default: all)")
@click.option("--tiers", default=None, help="Comma-separated tiers (default: 1,2,3,4,5)")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/queries"))
@click.option("--rpm", type=int, default=35, help="Rate limit (default: 35)")
@click.option("--skip-verify", is_flag=True, help="Skip verification step (for testing)")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--dry-run", is_flag=True, help="Print plan without executing")
@click.option("--verify-only", is_flag=True, help="Run only verification on existing raw candidates")
def main(
    domains: str | None,
    tiers: str | None,
    output_dir: Path,
    rpm: int,
    skip_verify: bool,
    resume: bool,
    dry_run: bool,
    verify_only: bool,
) -> None:
    """Run query generation."""
    selected_domains = _parse_domains(domains)
    selected_tiers = _parse_tiers(tiers)
    config = ProjectConfig()

    summary = asyncio.run(
        run_full_generation(
            config.prompts_dir,
            output_dir,
            config,
            domains=selected_domains,
            tiers=selected_tiers,
            rpm=rpm,
            skip_verify=skip_verify,
            resume=resume,
            dry_run=dry_run,
            verify_only=verify_only,
        )
    )
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
