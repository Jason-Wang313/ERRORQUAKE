"""Targeted Phase 1 refill for cells that finished below target."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import click

from errorquake.generate import (
    GenerationParseError,
    GenerationRequestError,
    RateLimiter,
    _batch_sizes,
    _generate_batch,
    _load_prompt_text,
    _normalise_candidate,
    run_full_generation,
)
from errorquake.queries import DOMAINS, TIERS
from errorquake.utils import ProjectConfig, now_iso, read_jsonl, setup_logger, write_jsonl

_SUBTOPIC_LINE_RE = re.compile(r"^\s*\d+\.\s+(?P<name>.+?)\s*$")


def _selected_counts(output_dir: Path) -> dict[str, int]:
    counts = {f"{domain}_T{tier}": 0 for domain in DOMAINS for tier in TIERS}
    for domain in DOMAINS:
        path = output_dir / f"{domain.lower()}.jsonl"
        for record in read_jsonl(path):
            key = f"{record['domain']}_T{int(record['tier'])}"
            counts[key] += 1
    return counts


def _selected_subtopic_counts(output_dir: Path, domain: str, tier: int) -> Counter[str]:
    path = output_dir / f"{domain.lower()}.jsonl"
    counter: Counter[str] = Counter()
    for record in read_jsonl(path):
        if str(record.get("domain")) != domain or int(record.get("tier", 0)) != tier:
            continue
        metadata = record.get("metadata", {})
        subtopic = str(metadata.get("subtopic", "")).strip()
        if subtopic:
            counter[subtopic] += 1
    return counter


def _raw_subtopic_counts(raw_dir: Path, domain: str, tier: int) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in read_jsonl(raw_dir / f"{domain}_T{tier}.jsonl"):
        subtopic = str(record.get("subtopic", "")).strip()
        if subtopic:
            counter[subtopic] += 1
    return counter


def _parse_subtopics(domain_prompt: str) -> list[str]:
    lines = domain_prompt.splitlines()
    collecting = False
    subtopics: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SUB-TOPICS"):
            collecting = True
            continue
        if collecting and stripped.startswith("DIFFICULTY TIERS"):
            break
        if not collecting:
            continue
        match = _SUBTOPIC_LINE_RE.match(line)
        if match:
            subtopics.append(match.group("name").strip())
    return subtopics


def _question_openings(records: list[dict[str, Any]], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for record in records:
        words = re.findall(r"[A-Za-z0-9']+", str(record.get("question", "")))
        if len(words) >= 3:
            counter[" ".join(words[:3]).lower()] += 1
    return [opening for opening, count in counter.most_common(limit) if count >= 3]


def _examples_to_avoid(records: list[dict[str, Any]], limit: int = 12) -> list[str]:
    seen: set[str] = set()
    examples: list[str] = []
    for record in records:
        question = str(record.get("question", "")).strip()
        if not question or question in seen:
            continue
        examples.append(question)
        seen.add(question)
        if len(examples) >= limit:
            break
    return examples


def _focus_subtopics(
    all_subtopics: list[str],
    selected_counts: Counter[str],
    raw_counts: Counter[str],
    limit: int = 5,
) -> list[str]:
    ranked = sorted(
        all_subtopics,
        key=lambda item: (
            selected_counts.get(item, 0),
            raw_counts.get(item, 0),
            item,
        ),
    )
    return ranked[: max(1, min(limit, len(ranked)))]


def _plan_new_candidates(deficit: int, final_count: int, raw_count: int, max_new_per_cell: int) -> int:
    observed_yield = final_count / max(raw_count, 1)
    expected_yield = max(0.20, min(0.65, observed_yield * 1.75))
    planned = math.ceil((deficit / expected_yield) * 1.15)
    return min(max_new_per_cell, max(50, planned))


def _build_refill_domain_prompt(
    *,
    base_domain_prompt: str,
    tier: int,
    focus_subtopic: str,
    banned_openings: list[str],
    examples_to_avoid: list[str],
) -> str:
    banned_section = ""
    if banned_openings:
        banned_section = (
            "Avoid starting questions with these overused openings: "
            + ", ".join(f'"{opening}"' for opening in banned_openings)
            + "."
        )

    examples_section = ""
    if examples_to_avoid:
        examples = "\n".join(f"- {example}" for example in examples_to_avoid)
        examples_section = f"Existing examples to avoid repeating or paraphrasing:\n{examples}"

    return (
        f"{base_domain_prompt.strip()}\n\n"
        "REFILL MODE:\n"
        f"This batch is for difficulty tier T{tier} and must focus ONLY on the subtopic "
        f'"{focus_subtopic}".\n'
        "Every generated question must be materially distinct from the existing pool.\n"
        "Prefer less-canonical entities, cases, places, works, institutions, or facts.\n"
        "Vary the factual framing and avoid textbook-trivia clones.\n"
        f"{banned_section}\n"
        f"{examples_section}\n"
        'Set the "subtopic" field to the exact focus subtopic string.\n'
        "Do not produce paraphrases of the same fact with only surface wording changes."
    ).strip()


async def _refill_cell(
    *,
    output_dir: Path,
    domain: str,
    tier: int,
    target_new_candidates: int,
    config: ProjectConfig,
    rpm: int,
    batch_size: int,
    logger: Any,
) -> int:
    raw_dir = output_dir / "raw"
    errors_path = output_dir / "refill_errors.jsonl"
    raw_path = raw_dir / f"{domain}_T{tier}.jsonl"
    existing = read_jsonl(raw_path)
    system_prompt, base_domain_prompt = _load_prompt_text(config.prompts_dir, domain)
    all_subtopics = _parse_subtopics(base_domain_prompt)
    selected_counts = _selected_subtopic_counts(output_dir, domain, tier)
    raw_counts = _raw_subtopic_counts(raw_dir, domain, tier)
    focus_subtopics = _focus_subtopics(all_subtopics, selected_counts, raw_counts)
    examples = _examples_to_avoid(existing)
    banned_openings = _question_openings(existing)
    rate_limiter = RateLimiter(rpm=rpm)

    generated = 0
    start_index = len(existing)
    previous_subtopics = [
        str(record.get("subtopic", "")).strip()
        for record in existing
        if str(record.get("subtopic", "")).strip()
    ]

    for batch_index, size in enumerate(_batch_sizes(target_new_candidates, batch_size=batch_size), start=1):
        focus_subtopic = focus_subtopics[(batch_index - 1) % len(focus_subtopics)]
        domain_prompt = _build_refill_domain_prompt(
            base_domain_prompt=base_domain_prompt,
            tier=tier,
            focus_subtopic=focus_subtopic,
            banned_openings=banned_openings,
            examples_to_avoid=examples,
        )
        try:
            batch = await _generate_batch(
                system_prompt=system_prompt,
                domain_prompt=domain_prompt,
                tier=tier,
                batch_size=size,
                previous_subtopics=previous_subtopics,
                model=config.generation_model,
                rate_limiter=rate_limiter,
                max_tokens=config.generation_max_tokens,
                timeout_s=config.generation_timeout_s,
            )
        except (GenerationParseError, GenerationRequestError) as exc:
            write_jsonl(
                errors_path,
                [
                    {
                        "timestamp": now_iso(),
                        "domain": domain,
                        "tier": tier,
                        "batch_index": batch_index,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                ],
            )
            logger.warning("[%s T%s refill] Batch %s failed: %s", domain, tier, batch_index, exc)
            continue

        normalised: list[dict[str, Any]] = []
        for item in batch:
            start_index += 1
            normalised.append(
                _normalise_candidate(
                    item,
                    domain=domain,
                    tier=tier,
                    candidate_id=f"RAW_{domain}_T{tier}_{start_index:05d}",
                )
            )
        if normalised:
            write_jsonl(raw_path, normalised)
            generated += len(normalised)
            previous_subtopics.extend(
                str(item.get("subtopic", "")).strip()
                for item in normalised
                if str(item.get("subtopic", "")).strip()
            )
        if batch_index % 5 == 0:
            logger.info(
                "[%s T%s refill] Batch %s complete | added=%s/%s | focus=%s",
                domain,
                tier,
                batch_index,
                generated,
                target_new_candidates,
                focus_subtopic,
            )
    return generated


def _parse_domains(value: str | None) -> list[str]:
    if not value:
        return DOMAINS
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_tiers(value: str | None) -> list[int]:
    if not value:
        return TIERS
    return [int(item.strip()) for item in value.split(",") if item.strip()]


@click.command()
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("data/queries"))
@click.option("--domains", default=None, help="Comma-separated domains to refill")
@click.option("--tiers", default=None, help="Comma-separated tiers to refill")
@click.option("--target-per-cell", type=int, default=250, show_default=True)
@click.option("--rounds", type=int, default=1, show_default=True)
@click.option("--rpm", type=int, default=35, show_default=True)
@click.option("--batch-size", type=int, default=10, show_default=True)
@click.option("--max-new-per-cell", type=int, default=1200, show_default=True)
@click.option("--rebuild", is_flag=True, help="Run verify-only rebuild after each round")
@click.option("--dry-run", is_flag=True, help="Print refill plan only")
def main(
    output_dir: Path,
    domains: str | None,
    tiers: str | None,
    target_per_cell: int,
    rounds: int,
    rpm: int,
    batch_size: int,
    max_new_per_cell: int,
    rebuild: bool,
    dry_run: bool,
) -> None:
    summary = asyncio.run(
        run_refill(
            output_dir=output_dir,
            domains=_parse_domains(domains),
            tiers=_parse_tiers(tiers),
            target_per_cell=target_per_cell,
            rounds=rounds,
            rpm=rpm,
            batch_size=batch_size,
            max_new_per_cell=max_new_per_cell,
            rebuild=rebuild,
            dry_run=dry_run,
        )
    )
    click.echo(json.dumps(summary, indent=2))


async def run_refill(
    *,
    output_dir: Path,
    domains: list[str],
    tiers: list[int],
    target_per_cell: int,
    rounds: int,
    rpm: int,
    batch_size: int,
    max_new_per_cell: int,
    rebuild: bool,
    dry_run: bool,
) -> dict[str, Any]:
    config = ProjectConfig()
    logger = setup_logger("errorquake.refill", output_dir / "logs")
    last_plan: dict[str, Any] = {}

    for round_index in range(1, rounds + 1):
        counts = _selected_counts(output_dir)
        deficits: list[dict[str, Any]] = []
        for domain in domains:
            for tier in tiers:
                key = f"{domain}_T{tier}"
                final_count = counts.get(key, 0)
                deficit = max(0, target_per_cell - final_count)
                if deficit <= 0:
                    continue
                raw_count = len(read_jsonl(output_dir / "raw" / f"{domain}_T{tier}.jsonl"))
                planned = _plan_new_candidates(
                    deficit=deficit,
                    final_count=final_count,
                    raw_count=raw_count,
                    max_new_per_cell=max_new_per_cell,
                )
                deficits.append(
                    {
                        "cell": key,
                        "domain": domain,
                        "tier": tier,
                        "final_count": final_count,
                        "deficit": deficit,
                        "raw_count": raw_count,
                        "planned_new_candidates": planned,
                    }
                )

        plan = {
            "round": round_index,
            "cells": deficits,
            "target_per_cell": target_per_cell,
            "rpm": rpm,
            "batch_size": batch_size,
            "max_new_per_cell": max_new_per_cell,
            "rebuild": rebuild,
        }
        last_plan = plan
        if dry_run:
            return plan
        if not deficits:
            logger.info("Refill complete: no deficits remain.")
            return plan

        logger.info("Refill round %s starting for %s cells", round_index, len(deficits))
        for item in deficits:
            logger.info(
                "[%s] final=%s deficit=%s raw=%s planned_new=%s",
                item["cell"],
                item["final_count"],
                item["deficit"],
                item["raw_count"],
                item["planned_new_candidates"],
            )
            generated = await _refill_cell(
                output_dir=output_dir,
                domain=item["domain"],
                tier=item["tier"],
                target_new_candidates=item["planned_new_candidates"],
                config=config,
                rpm=rpm,
                batch_size=batch_size,
                logger=logger,
            )
            logger.info("[%s] refill generated %s new raw candidates", item["cell"], generated)

        if rebuild:
            logger.info("Running verify-only rebuild after refill round %s", round_index)
            await run_full_generation(
                config.prompts_dir,
                output_dir,
                config,
                domains=DOMAINS,
                tiers=TIERS,
                rpm=rpm,
                skip_verify=False,
                resume=True,
                dry_run=False,
                verify_only=True,
            )

    return last_plan


if __name__ == "__main__":
    main()
