"""Regenerate specific ERRORQUAKE cells with DeepSeek-V3.2, sequentially."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from errorquake.generate import (
    _build_batch_prompt,
    GenerationParseError,
    GenerationRequestError,
    RateLimiter,
    _build_manifest,
    _chunked,
    _generate_batch,
    _load_prompt_text,
    _normalise_candidate,
    _rewrite_jsonl,
    assign_final_ids,
    deduplicate_queries,
    select_final,
    validate_candidate,
    verify_queries_batch,
    verify_query,
)
from errorquake.queries import DOMAINS, TIERS
from errorquake.utils import ProjectConfig, now_iso, read_jsonl, setup_logger, write_jsonl

TARGET_CELLS: list[tuple[str, int]] = [
    ("TECH", 5),
    ("FIN", 5),
    ("SCI", 5),
    ("GEO", 5),
    ("LAW", 1),
    ("BIO", 1),
]
GENERATION_MODEL = "deepseek-ai/deepseek-v3.2"
VERIFICATION_MODEL = "qwen/qwen3-next-80b-a3b-instruct"


def _parse_cells(value: str | None) -> list[tuple[str, int]]:
    if not value:
        return TARGET_CELLS
    cells: list[tuple[str, int]] = []
    for chunk in value.split(","):
        item = chunk.strip().upper()
        if not item:
            continue
        domain, tier = item.split("_T", 1)
        cells.append((domain, int(tier)))
    return cells


def _raw_path(output_dir: Path, domain: str, tier: int) -> Path:
    return output_dir / "raw" / f"{domain}_T{tier}.jsonl"


def _verified_path(output_dir: Path, domain: str) -> Path:
    return output_dir / "verified" / f"{domain.lower()}.jsonl"


def _domain_output_path(output_dir: Path, domain: str) -> Path:
    return output_dir / f"{domain.lower()}.jsonl"


def _cell_key(domain: str, tier: int) -> str:
    return f"{domain}_T{tier}"


def _sort_final_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: (int(item.get("tier", 0)), str(item.get("id", ""))))


def _sort_verified_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: (int(item.get("tier", 0)), str(item.get("id", ""))))


def _save_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_DEEPSEEK_NATIVE_CLIENT: AsyncOpenAI | None = None


def _get_deepseek_native_client() -> AsyncOpenAI:
    global _DEEPSEEK_NATIVE_CLIENT
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY for DeepSeek native API path.")
    if _DEEPSEEK_NATIVE_CLIENT is None:
        _DEEPSEEK_NATIVE_CLIENT = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _DEEPSEEK_NATIVE_CLIENT


async def _generate_batch_deepseek_native(
    *,
    system_prompt: str,
    domain_prompt: str,
    tier: int,
    batch_size: int,
    previous_subtopics: list[str],
    rate_limiter: RateLimiter,
    max_tokens: int,
    timeout_s: int,
) -> list[dict[str, Any]]:
    client = _get_deepseek_native_client()
    prompt = _build_batch_prompt(
        system_prompt,
        domain_prompt,
        tier=tier,
        batch_size=batch_size,
        previous_subtopics=previous_subtopics,
    )
    await rate_limiter.acquire()
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        ),
        timeout=timeout_s,
    )
    raw = response.choices[0].message.content or ""
    from errorquake.generate import parse_generation_response

    return parse_generation_response(raw)[:batch_size]


def _validate_candidates_local(
    candidates: list[dict[str, Any]],
    discards_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    failures = 0
    discards_path.parent.mkdir(parents=True, exist_ok=True)
    discards_path.write_text("", encoding="utf-8")
    for candidate in candidates:
        errors = validate_candidate(candidate)
        if errors:
            failures += 1
            write_jsonl(
                discards_path,
                [
                    {
                        "timestamp": now_iso(),
                        "id": candidate.get("id"),
                        "domain": candidate.get("domain"),
                        "tier": candidate.get("tier"),
                        "question": candidate.get("question"),
                        "reasons": errors,
                    }
                ],
            )
            continue
        valid.append(candidate)
    return valid, failures


async def _generate_exact_candidates(
    *,
    domain: str,
    tier: int,
    prompts_dir: Path,
    target_candidates: int,
    batch_size: int,
    rpm: int,
    max_tokens: int,
    timeout_s: int,
    generation_backend: str,
    logger: Any,
) -> tuple[list[dict[str, Any]], int]:
    system_prompt, domain_prompt = _load_prompt_text(prompts_dir, domain)
    rate_limiter = RateLimiter(rpm=rpm)
    regen_tag = now_iso().replace(":", "").replace("-", "").replace("+", "_")
    generated: list[dict[str, Any]] = []
    parse_failures = 0
    batch_index = 0

    while len(generated) < target_candidates:
        batch_index += 1
        size = min(batch_size, target_candidates - len(generated))
        previous_subtopics = [
            str(item.get("subtopic", "")).strip()
            for item in generated
            if str(item.get("subtopic", "")).strip()
        ]
        attempt = 0
        while True:
            attempt += 1
            try:
                if generation_backend == "deepseek_native":
                    batch = await _generate_batch_deepseek_native(
                        system_prompt=system_prompt,
                        domain_prompt=domain_prompt,
                        tier=tier,
                        batch_size=size,
                        previous_subtopics=previous_subtopics,
                        rate_limiter=rate_limiter,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                    )
                else:
                    batch = await _generate_batch(
                        system_prompt=system_prompt,
                        domain_prompt=domain_prompt,
                        tier=tier,
                        batch_size=size,
                        previous_subtopics=previous_subtopics,
                        model=GENERATION_MODEL,
                        rate_limiter=rate_limiter,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                    )
                break
            except GenerationParseError as exc:
                parse_failures += 1
                if attempt >= 4:
                    raise
                logger.warning(
                    "[%s T%s regen] Parse failure on batch %s attempt %s with %s: %s",
                    domain,
                    tier,
                    batch_index,
                    attempt,
                    GENERATION_MODEL,
                    exc,
                )
                await asyncio.sleep(3)
            except GenerationRequestError as exc:
                if attempt >= 4:
                    raise
                logger.warning(
                    "[%s T%s regen] Request failure on batch %s attempt %s with %s via %s (timeout=%ss): %s",
                    domain,
                    tier,
                    batch_index,
                    attempt,
                    GENERATION_MODEL,
                    generation_backend,
                    timeout_s,
                    exc,
                )
                await asyncio.sleep(5)
            except asyncio.TimeoutError as exc:
                if attempt >= 4:
                    raise GenerationRequestError(f"Timed out after {timeout_s} seconds.") from exc
                logger.warning(
                    "[%s T%s regen] Native timeout on batch %s attempt %s with %s via %s (timeout=%ss)",
                    domain,
                    tier,
                    batch_index,
                    attempt,
                    GENERATION_MODEL,
                    generation_backend,
                    timeout_s,
                )
                await asyncio.sleep(5)

        normalised_batch: list[dict[str, Any]] = []
        for offset, item in enumerate(batch, start=1):
            normalised_batch.append(
                _normalise_candidate(
                    item,
                    domain=domain,
                    tier=tier,
                    candidate_id=f"RAW_{domain}_T{tier}_REGEN_{regen_tag}_{len(generated) + offset:05d}",
                )
            )
        generated.extend(normalised_batch)
        logger.info(
            "[%s T%s regen] Batch %s complete | generated=%s/%s | model=%s | timeout=%ss",
            domain,
            tier,
            batch_index,
            len(generated),
            target_candidates,
            GENERATION_MODEL,
            timeout_s,
        )

    return generated[:target_candidates], parse_failures


async def _verify_candidates_sequential(
    *,
    candidates: list[dict[str, Any]],
    rpm: int,
    batch_size: int,
    max_tokens: int,
    timeout_s: int,
    logger: Any,
    domain: str,
    tier: int,
) -> list[dict[str, Any]]:
    rate_limiter = RateLimiter(rpm=rpm)
    verified: list[dict[str, Any]] = []
    batches = _chunked(candidates, max(1, batch_size))

    for index, batch in enumerate(batches, start=1):
        try:
            verified_batch = await verify_queries_batch(
                batch,
                model=VERIFICATION_MODEL,
                rate_limiter=rate_limiter,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
        except (GenerationParseError, GenerationRequestError) as exc:
            logger.warning(
                "[%s T%s regen] Batch verify fallback on batch %s/%s with %s: %s",
                domain,
                tier,
                index,
                len(batches),
                VERIFICATION_MODEL,
                exc,
            )
            verified_batch = []
            for candidate in batch:
                verified_batch.append(
                    await verify_query(
                        candidate,
                        model=VERIFICATION_MODEL,
                        rate_limiter=rate_limiter,
                        max_tokens=min(300, max_tokens),
                        timeout_s=timeout_s,
                    )
                )
        verified.extend(verified_batch)
        if index % 10 == 0 or index == len(batches):
            logger.info(
                "[%s T%s regen] Verify batch %s/%s complete | verified=%s/%s",
                domain,
                tier,
                index,
                len(batches),
                len(verified),
                len(candidates),
            )
    return verified


def _replace_domain_tier_records(
    *,
    path: Path,
    tier: int,
    new_records: list[dict[str, Any]],
    sorter: Any,
) -> None:
    existing = read_jsonl(path)
    preserved = [record for record in existing if int(record.get("tier", 0)) != tier]
    _rewrite_jsonl(path, sorter(preserved + new_records))


def _load_all_raw_candidates(output_dir: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for tier in TIERS:
            candidates.extend(read_jsonl(_raw_path(output_dir, domain, tier)))
    return candidates


def _load_all_final_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        records.extend(read_jsonl(_domain_output_path(output_dir, domain)))
    return records


def _load_all_verified_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        records.extend(read_jsonl(_verified_path(output_dir, domain)))
    return records


def _rebuild_manifest(
    *,
    output_dir: Path,
    config: ProjectConfig,
    parse_failures: int,
    generation_hours: float,
) -> dict[str, Any]:
    raw_candidates = _load_all_raw_candidates(output_dir)
    valid_candidates: list[dict[str, Any]] = []
    validation_failures = 0
    for candidate in raw_candidates:
        errors = validate_candidate(candidate)
        if errors:
            validation_failures += 1
            continue
        valid_candidates.append(candidate)
    deduplicated = deduplicate_queries(valid_candidates, threshold=0.80)
    final_records = _load_all_final_records(output_dir)
    verified_records = _load_all_verified_records(output_dir)
    verification_matches = sum(1 for item in verified_records if bool(item.get("verified")))
    verification_rate = (
        verification_matches / len(verified_records) if verified_records else 0.0
    )

    per_cell_counts = {
        _cell_key(domain, tier): 0
        for domain in DOMAINS
        for tier in TIERS
    }
    for record in final_records:
        per_cell_counts[_cell_key(str(record["domain"]), int(record["tier"]))] += 1
    cells_below_target = [
        key for key, count in per_cell_counts.items() if count < config.queries_per_cell
    ]

    manifest = _build_manifest(
        final_records=final_records,
        domains=DOMAINS,
        tiers=TIERS,
        config=config,
        candidates_generated=len(raw_candidates),
        candidates_after_validation=len(valid_candidates),
        candidates_after_dedup=len(deduplicated),
        verification_rate=verification_rate,
        cells_below_target=cells_below_target,
        generation_hours=generation_hours,
    )
    manifest["parse_failures"] = parse_failures
    manifest["parse_failure_rate"] = parse_failures / max(len(raw_candidates), 1)
    manifest["validation_failures"] = validation_failures
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


async def run_regeneration(
    *,
    output_dir: Path,
    work_dir: Path,
    cells: list[tuple[str, int]],
    generation_rpm: int,
    verification_rpm: int,
    generation_timeout_s: int,
    verification_timeout_s: int,
    generation_batch_size: int,
    generation_max_tokens: int,
    generation_backend: str,
) -> dict[str, Any]:
    config = ProjectConfig()
    config.generation_model = GENERATION_MODEL
    config.generation_rpm = generation_rpm
    config.generation_timeout_s = generation_timeout_s
    config.generation_batch_size = generation_batch_size
    config.generation_max_tokens = generation_max_tokens
    config.oversample_factor = 2
    config.queries_per_cell = 250
    config.verification_model = VERIFICATION_MODEL
    config.verification_batch_size = 5
    config.verification_rpm = verification_rpm
    config.verification_concurrency = 1
    config.verification_timeout_s = verification_timeout_s

    logger = setup_logger("errorquake.targeted_regen", output_dir / "logs")
    started = time.perf_counter()
    existing_manifest = {}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    progress_path = work_dir / "targeted_regeneration_progress.json"
    progress: dict[str, Any] = {
        "generation_model": GENERATION_MODEL,
        "generation_backend": generation_backend,
        "generation_timeout_s": generation_timeout_s,
        "generation_batch_size": generation_batch_size,
        "generation_max_tokens": generation_max_tokens,
        "verification_model": VERIFICATION_MODEL,
        "verification_timeout_s": verification_timeout_s,
        "strictly_sequential": True,
        "cells": [_cell_key(domain, tier) for domain, tier in cells],
        "completed_cells": [],
        "cell_summaries": {},
    }
    _save_progress(progress_path, progress)

    total_parse_failures = 0
    for domain, tier in cells:
        cell_name = _cell_key(domain, tier)
        logger.info(
            "[%s regen] Starting sequential regeneration with model=%s timeout=%ss",
            cell_name,
            GENERATION_MODEL,
            generation_timeout_s,
        )
        cell_dir = work_dir / cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)

        raw_candidates, parse_failures = await _generate_exact_candidates(
            domain=domain,
            tier=tier,
            prompts_dir=config.prompts_dir,
            target_candidates=config.queries_per_cell * config.oversample_factor,
            batch_size=config.generation_batch_size,
            rpm=generation_rpm,
            max_tokens=config.generation_max_tokens,
            timeout_s=generation_timeout_s,
            generation_backend=generation_backend,
            logger=logger,
        )
        total_parse_failures += parse_failures
        _rewrite_jsonl(cell_dir / "raw_candidates.jsonl", raw_candidates)

        valid_candidates, validation_failures = _validate_candidates_local(
            raw_candidates,
            cell_dir / "validation_discards.jsonl",
        )
        deduplicated = deduplicate_queries(valid_candidates, threshold=0.80)
        verified = await _verify_candidates_sequential(
            candidates=deduplicated,
            rpm=verification_rpm,
            batch_size=config.verification_batch_size,
            max_tokens=config.verification_max_tokens,
            timeout_s=verification_timeout_s,
            logger=logger,
            domain=domain,
            tier=tier,
        )
        selected = select_final(verified, target_per_cell=config.queries_per_cell)
        if len(selected) < config.queries_per_cell:
            raise RuntimeError(
                f"{cell_name} produced only {len(selected)} final queries after selection."
            )

        final_records = assign_final_ids(domain, tier, selected)
        _rewrite_jsonl(cell_dir / "verified_candidates.jsonl", verified)
        _rewrite_jsonl(cell_dir / "final_records.jsonl", final_records)

        _rewrite_jsonl(_raw_path(output_dir, domain, tier), raw_candidates)
        _replace_domain_tier_records(
            path=_verified_path(output_dir, domain),
            tier=tier,
            new_records=verified,
            sorter=_sort_verified_records,
        )
        _replace_domain_tier_records(
            path=_domain_output_path(output_dir, domain),
            tier=tier,
            new_records=final_records,
            sorter=_sort_final_records,
        )

        summary = {
            "domain": domain,
            "tier": tier,
            "generation_model": GENERATION_MODEL,
            "generation_backend": generation_backend,
            "generation_timeout_s": generation_timeout_s,
            "generation_batch_size": generation_batch_size,
            "generation_max_tokens": generation_max_tokens,
            "verification_model": VERIFICATION_MODEL,
            "verification_timeout_s": verification_timeout_s,
            "raw_candidates": len(raw_candidates),
            "validation_failures": validation_failures,
            "validated_candidates": len(valid_candidates),
            "deduplicated_candidates": len(deduplicated),
            "verified_candidates": len(verified),
            "selected_candidates": len(final_records),
            "parse_failures": parse_failures,
        }
        (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        progress["completed_cells"].append(cell_name)
        progress["cell_summaries"][cell_name] = summary
        _save_progress(progress_path, progress)
        logger.info(
            "[%s regen] Complete | raw=%s valid=%s dedup=%s selected=%s",
            cell_name,
            len(raw_candidates),
            len(valid_candidates),
            len(deduplicated),
            len(final_records),
        )

    manifest = _rebuild_manifest(
        output_dir=output_dir,
        config=config,
        parse_failures=total_parse_failures,
        generation_hours=(time.perf_counter() - started) / 3600.0,
    )
    manifest["generation_model"] = "mixed"
    manifest["generation_models_by_cell"] = {
        GENERATION_MODEL: [_cell_key(domain, tier) for domain, tier in cells],
        str(existing_manifest.get("generation_model", "unknown_previous_model")): [
            _cell_key(domain, tier)
            for domain in DOMAINS
            for tier in TIERS
            if (domain, tier) not in cells
        ],
    }
    manifest["regenerated_cells"] = [_cell_key(domain, tier) for domain, tier in cells]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    progress["manifest"] = manifest
    _save_progress(progress_path, progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate specific ERRORQUAKE cells sequentially.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/projects/errorquake/data/queries"),
        help="Main query output directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("C:/projects/errorquake/data/queries/regeneration_work/deepseek_v32_targeted"),
        help="Directory to store per-cell intermediate artifacts and progress.",
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="Optional comma-separated list like TECH_T5,FIN_T5.",
    )
    parser.add_argument(
        "--generation-backend",
        choices=["nim", "deepseek_native"],
        default="nim",
        help="Use NVIDIA NIM or DeepSeek's native API for generation.",
    )
    parser.add_argument("--generation-rpm", type=int, default=35)
    parser.add_argument("--verification-rpm", type=int, default=35)
    parser.add_argument("--generation-timeout", type=int, default=90)
    parser.add_argument("--verification-timeout", type=int, default=90)
    parser.add_argument("--generation-batch-size", type=int, default=25)
    parser.add_argument("--generation-max-tokens", type=int, default=4000)
    args = parser.parse_args()

    summary = asyncio.run(
        run_regeneration(
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            cells=_parse_cells(args.cells),
            generation_rpm=args.generation_rpm,
            verification_rpm=args.verification_rpm,
            generation_timeout_s=args.generation_timeout,
            verification_timeout_s=args.verification_timeout,
            generation_batch_size=args.generation_batch_size,
            generation_max_tokens=args.generation_max_tokens,
            generation_backend=args.generation_backend,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
