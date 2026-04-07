"""Prefetch future raw generation cells without touching the main progress file."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from errorquake.generate import (
    RateLimiter,
    _generate_cell_candidates,
    _raw_cell_path,
)
from errorquake.queries import DOMAINS, TIERS
from errorquake.utils import ProjectConfig, setup_logger


@click.command()
@click.option(
    "--domains",
    multiple=True,
    type=click.Choice(DOMAINS, case_sensitive=False),
    required=True,
    help="Future domains to prefetch.",
)
@click.option(
    "--tiers",
    multiple=True,
    type=click.IntRange(1, 5),
    default=TIERS,
    show_default=True,
    help="Difficulty tiers to prefetch.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/queries"),
    show_default=True,
)
@click.option("--rpm", type=int, default=40, show_default=True, help="Prefetch RPM cap.")
@click.option(
    "--worker",
    type=str,
    required=True,
    help="Worker label for separate logs and checkpoint files.",
)
def main(
    domains: tuple[str, ...],
    tiers: tuple[int, ...],
    output_dir: Path,
    rpm: int,
    worker: str,
) -> None:
    asyncio.run(
        run_prefetch(
            domains=[domain.upper() for domain in domains],
            tiers=list(tiers),
            output_dir=output_dir,
            rpm=rpm,
            worker=worker,
        )
    )


async def run_prefetch(
    *,
    domains: list[str],
    tiers: list[int],
    output_dir: Path,
    rpm: int,
    worker: str,
) -> None:
    config = ProjectConfig()
    raw_dir = output_dir / "raw"
    prefetch_dir = output_dir / "prefetch"
    worker_dir = prefetch_dir / worker
    errors_path = worker_dir / "generation_errors.jsonl"
    progress_path = worker_dir / "progress.json"
    logger = setup_logger(f"errorquake.prefetch.{worker}", output_dir / "logs")
    rate_limiter = RateLimiter(rpm=rpm)

    main_progress_path = raw_dir / "generation_progress.json"

    for domain in domains:
        for tier in tiers:
            raw_path = _raw_cell_path(raw_dir, domain, tier)

            if raw_path.exists():
                logger.info("[%s T%s] Prefetch skip: raw file already exists", domain, tier)
                continue

            if main_progress_path.exists():
                main_progress = json.loads(main_progress_path.read_text(encoding="utf-8"))
                completed = {tuple(item) for item in main_progress.get("completed_cells", [])}
                if (domain, tier) in completed:
                    logger.info("[%s T%s] Prefetch skip: already completed by main run", domain, tier)
                    continue
                if main_progress.get("current_cell") == [domain, tier]:
                    logger.info("[%s T%s] Prefetch skip: currently owned by main run", domain, tier)
                    continue

            logger.info("[%s T%s] Prefetch starting", domain, tier)
            await _generate_cell_candidates(
                domain=domain,
                tier=tier,
                prompts_dir=config.prompts_dir,
                raw_dir=raw_dir,
                errors_path=errors_path,
                progress_path=progress_path,
                config=config,
                rate_limiter=rate_limiter,
                resume=True,
                logger=logger,
            )
            logger.info("[%s T%s] Prefetch complete", domain, tier)


if __name__ == "__main__":
    main()
