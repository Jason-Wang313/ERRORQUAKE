"""Background verifier for completed domains during Phase 1 generation."""

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
        return {
            "verified_cells": [],
            "verified_domains": [],
            "started_at": now_iso(),
            "last_scan": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cell_key(domain: str, tier: int) -> str:
    return f"{domain}_T{tier}"


def _completed_cells(progress: dict[str, Any]) -> list[tuple[str, int]]:
    completed: list[tuple[str, int]] = []
    for item in progress.get("completed_cells", []):
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        domain = str(item[0])
        try:
            tier = int(item[1])
        except (TypeError, ValueError):
            continue
        if domain in DOMAINS and tier in TIERS:
            completed.append((domain, tier))
    return completed


@click.command()
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/queries"))
@click.option("--poll-seconds", type=int, default=90, help="How often to scan for completed domains")
@click.option("--rpm", type=int, default=8, help="Verification sidecar RPM")
@click.option("--batch-size", type=int, default=5, help="Verification sidecar batch size")
@click.option("--concurrency", type=int, default=2, help="Verification sidecar concurrency")
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
        run_sidecar(
            output_dir=output_dir,
            poll_seconds=poll_seconds,
            rpm=rpm,
            batch_size=batch_size,
            concurrency=concurrency,
            once=once,
        )
    )


async def run_sidecar(
    *,
    output_dir: Path,
    poll_seconds: int,
    rpm: int,
    batch_size: int,
    concurrency: int,
    once: bool,
) -> None:
    logger = setup_logger("errorquake.verify_sidecar", output_dir / "logs")
    config = replace(
        ProjectConfig(),
        verification_rpm=rpm,
        verification_batch_size=batch_size,
        verification_concurrency=concurrency,
    )
    raw_dir = output_dir / "raw"
    verified_dir = output_dir / "verified"
    state_path = output_dir / "verify_sidecar_state.json"
    progress_path = raw_dir / "generation_progress.json"
    discards_dir = output_dir / "sidecar_discards"
    state = _load_state(state_path)
    state.setdefault("verified_cells", [])
    state.setdefault("verified_domains", [])

    while True:
        if not progress_path.exists():
            logger.info("Waiting for generation progress file...")
            if once:
                return
            await asyncio.sleep(poll_seconds)
            continue

        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        state["last_scan"] = now_iso()
        _save_state(state_path, state)

        pending_cells = [
            (domain, tier)
            for domain, tier in _completed_cells(progress)
            if _cell_key(domain, tier) not in set(state.get("verified_cells", []))
        ]

        if not pending_cells:
            if len(state.get("verified_cells", [])) == len(DOMAINS) * len(TIERS):
                logger.info("All completed cells already verified by sidecar.")
                return
            if once:
                return
            await asyncio.sleep(poll_seconds)
            continue

        for domain, tier in pending_cells:
            logger.info("[%s T%s] Sidecar verification starting", domain, tier)
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
                logger.exception("[%s T%s] Sidecar verification failed; will retry on next scan", domain, tier)
                if once:
                    raise
                continue
            logger.info(
                "[%s T%s] Sidecar verification complete | raw=%s valid=%s dedup=%s validation_failures=%s",
                domain,
                tier,
                len(raw_candidates),
                len(valid_candidates),
                len(deduplicated),
                validation_failures,
            )
            state.setdefault("verified_cells", []).append(_cell_key(domain, tier))
            state["verified_cells"] = sorted(set(state["verified_cells"]))
            if all(_cell_key(domain, value) in set(state["verified_cells"]) for value in TIERS):
                state.setdefault("verified_domains", []).append(domain)
                state["verified_domains"] = sorted(set(state["verified_domains"]))
            state["verified_domains"] = sorted(set(state["verified_domains"]))
            state["last_completed_cell"] = _cell_key(domain, tier)
            state["last_completed_at"] = now_iso()
            _save_state(state_path, state)

        if once:
            return


if __name__ == "__main__":
    main()
