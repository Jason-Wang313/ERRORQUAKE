"""Verify completed prefetch cells ahead of the main generation run."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import click

from errorquake.generate import (
    RateLimiter,
    _validate_candidates,
    _verify_candidates_for_domain,
    deduplicate_queries,
)
from errorquake.queries import DOMAINS, TIERS
from errorquake.utils import ProjectConfig, now_iso, read_jsonl, setup_logger


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"verified_cells": [], "started_at": now_iso(), "last_scan": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cell_key(domain: str, tier: int) -> str:
    return f"{domain}_T{tier}"


def _load_completed_cells(progress_path: Path) -> list[tuple[str, int]]:
    if not progress_path.exists():
        return []
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    completed: list[tuple[str, int]] = []
    for item in payload.get("completed_cells", []):
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        domain = str(item[0]).upper()
        try:
            tier = int(item[1])
        except (TypeError, ValueError):
            continue
        if domain in DOMAINS and tier in TIERS:
            completed.append((domain, tier))
    return completed


@click.command()
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/queries"))
@click.option("--poll-seconds", type=int, default=90, help="How often to scan prefetch progress")
@click.option("--rpm", type=int, default=20, help="Verification RPM")
@click.option("--batch-size", type=int, default=5, help="Verification batch size")
@click.option("--concurrency", type=int, default=4, help="Verification concurrency")
@click.option("--once", is_flag=True, help="Run a single scan and exit")
def main(
    output_dir: Path,
    poll_seconds: int,
    rpm: int,
    batch_size: int,
    concurrency: int,
    once: bool,
) -> None:
    asyncio.run(
        run_prefetch_verifier(
            output_dir=output_dir,
            poll_seconds=poll_seconds,
            rpm=rpm,
            batch_size=batch_size,
            concurrency=concurrency,
            once=once,
        )
    )


async def run_prefetch_verifier(
    *,
    output_dir: Path,
    poll_seconds: int,
    rpm: int,
    batch_size: int,
    concurrency: int,
    once: bool,
) -> None:
    logger = setup_logger("errorquake.prefetch_verify", output_dir / "logs")
    config = replace(
        ProjectConfig(),
        verification_rpm=rpm,
        verification_batch_size=batch_size,
        verification_concurrency=concurrency,
    )
    raw_dir = output_dir / "raw"
    verified_dir = output_dir / "verified"
    prefetch_dir = output_dir / "prefetch"
    discards_dir = output_dir / "prefetch_verify_discards"
    state_path = output_dir / "prefetch_verify_state.json"
    state = _load_state(state_path)
    state.setdefault("verified_cells", [])

    while True:
        progress_files = sorted(prefetch_dir.glob("*/progress.json"))
        pending_cells: list[tuple[str, int]] = []
        seen = set(state.get("verified_cells", []))

        for progress_path in progress_files:
            for domain, tier in _load_completed_cells(progress_path):
                key = _cell_key(domain, tier)
                if key not in seen:
                    pending_cells.append((domain, tier))
                    seen.add(key)

        state["last_scan"] = now_iso()
        _save_state(state_path, state)

        if not pending_cells:
            if once:
                return
            await asyncio.sleep(poll_seconds)
            continue

        for domain, tier in pending_cells:
            logger.info("[%s T%s] Prefetch verification starting", domain, tier)
            raw_candidates = read_jsonl(raw_dir / f"{domain}_T{tier}.jsonl")
            discards_path = discards_dir / f"{domain.lower()}_t{tier}_validation_discards.jsonl"
            valid_candidates, validation_failures = _validate_candidates(raw_candidates, discards_path)
            deduplicated = deduplicate_queries(valid_candidates, threshold=0.80)
            try:
                await _verify_candidates_for_domain(
                    domain=domain,
                    candidates=deduplicated,
                    verified_dir=verified_dir,
                    config=config,
                    rate_limiter=RateLimiter(rpm=rpm),
                    skip_verify=False,
                    logger=logger,
                )
            except Exception:
                logger.exception(
                    "[%s T%s] Prefetch verification failed; will retry on next scan",
                    domain,
                    tier,
                )
                if once:
                    raise
                continue

            logger.info(
                "[%s T%s] Prefetch verification complete | raw=%s valid=%s dedup=%s validation_failures=%s",
                domain,
                tier,
                len(raw_candidates),
                len(valid_candidates),
                len(deduplicated),
                validation_failures,
            )
            state.setdefault("verified_cells", []).append(_cell_key(domain, tier))
            state["verified_cells"] = sorted(set(state["verified_cells"]))
            state["last_completed_cell"] = _cell_key(domain, tier)
            state["last_completed_at"] = now_iso()
            _save_state(state_path, state)

        if once:
            return


if __name__ == "__main__":
    main()
