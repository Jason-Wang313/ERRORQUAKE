"""Replace only tier-audit-flagged queries in selected cells."""

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
    _build_manifest,
    _chunked,
    _load_prompt_text,
    _normalise_candidate,
    _rewrite_jsonl,
    deduplicate_queries,
    select_final,
    validate_candidate,
    verify_queries_batch,
    verify_query,
    RateLimiter,
)
from errorquake.queries import DOMAINS, TIERS
from errorquake.utils import ProjectConfig, now_iso, read_jsonl, setup_logger, write_jsonl
from run_tier_audit import collect_flagged_records


ROOT = Path(__file__).resolve().parent.parent

TARGET_CELLS: list[tuple[str, int]] = [
    ("TECH", 5),
    ("FIN", 5),
    ("SCI", 5),
    ("GEO", 5),
    ("LAW", 1),
    ("BIO", 1),
]
GENERATION_MODEL = "deepseek-chat"
GENERATION_BASE_URL = "https://api.deepseek.com"
VERIFICATION_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
GENERATION_RETRY_ATTEMPTS = 3
GENERATION_RETRY_BACKOFF_S = 5

_DEEPSEEK_CLIENT: AsyncOpenAI | None = None


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


def _cell_name(domain: str, tier: int) -> str:
    return f"{domain}_T{tier}"


def _cell_output_path(output_dir: Path, domain: str) -> Path:
    return output_dir / f"{domain.lower()}.jsonl"


def _cell_raw_path(output_dir: Path, domain: str, tier: int) -> Path:
    return output_dir / "raw" / f"{domain}_T{tier}.jsonl"


def _verified_domain_path(output_dir: Path, domain: str) -> Path:
    return output_dir / "verified" / f"{domain.lower()}.jsonl"


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_progress(path: Path, payload: dict[str, Any]) -> None:
    _save_json(path, payload)


def _load_progress(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_deepseek_client() -> AsyncOpenAI:
    global _DEEPSEEK_CLIENT
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY.")
    if _DEEPSEEK_CLIENT is None:
        _DEEPSEEK_CLIENT = AsyncOpenAI(api_key=api_key, base_url=GENERATION_BASE_URL)
    return _DEEPSEEK_CLIENT


async def _generate_batch_deepseek_chat(
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
    from errorquake.generate import parse_generation_response

    prompt = _build_batch_prompt(
        system_prompt,
        domain_prompt,
        tier=tier,
        batch_size=batch_size,
        previous_subtopics=previous_subtopics,
    )
    client = _get_deepseek_client()
    last_error: Exception | None = None
    for attempt in range(1, GENERATION_RETRY_ATTEMPTS + 1):
        try:
            await rate_limiter.acquire()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=GENERATION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                ),
                timeout=timeout_s,
            )
            raw = response.choices[0].message.content or ""
            return parse_generation_response(raw)[:batch_size]
        except Exception as exc:
            last_error = exc
            if attempt >= GENERATION_RETRY_ATTEMPTS:
                break
            await asyncio.sleep(GENERATION_RETRY_BACKOFF_S * attempt)

    assert last_error is not None
    raise last_error


def _validate_candidates_local(
    candidates: list[dict[str, Any]],
    discards_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    discards_path.parent.mkdir(parents=True, exist_ok=True)
    discards_path.write_text("", encoding="utf-8")
    valid: list[dict[str, Any]] = []
    failures = 0
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


def _similarity_filter(
    preserved_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    threshold: float = 0.80,
) -> list[dict[str, Any]]:
    if not preserved_records or not candidates:
        return list(candidates)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    preserved_questions = [str(record.get("question", "")) for record in preserved_records]
    candidate_questions = [str(record.get("question", "")) for record in candidates]
    try:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vectorizer.fit_transform(preserved_questions + candidate_questions)
    except ValueError:
        return list(candidates)

    preserved_matrix = matrix[: len(preserved_questions)]
    candidate_matrix = matrix[len(preserved_questions) :]
    similarity = cosine_similarity(candidate_matrix, preserved_matrix)

    filtered: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if similarity.shape[1] and float(similarity[index].max()) > threshold:
            continue
        filtered.append(candidate)
    return filtered


def _preserved_candidate_view(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}))
    return {
        "id": str(record.get("id")),
        "domain": str(record.get("domain")),
        "tier": int(record.get("tier", 0)),
        "question": str(record.get("question", "")),
        "ground_truth": str(record.get("ground_truth", "")),
        "sources": list(record.get("sources", [])),
        "difficulty_rationale": str(record.get("difficulty_rationale", "")),
        "subtopic": str(metadata.get("subtopic", "")).strip(),
        "verified": bool(metadata.get("verified", True)),
        "verification_answer": metadata.get("verification_answer", ""),
        "verification_match": metadata.get("verification_match", True),
        "verification_model": metadata.get("verification_model", VERIFICATION_MODEL),
    }


def _next_fresh_ids(existing_records: list[dict[str, Any]], domain: str, tier: int, count: int) -> list[str]:
    prefix = f"{domain}_T{tier}_"
    used_numbers = {
        int(str(record.get("id"))[len(prefix) :])
        for record in existing_records
        if str(record.get("id", "")).startswith(prefix) and str(record.get("id"))[len(prefix) :].isdigit()
    }
    next_number = max(used_numbers) + 1 if used_numbers else 1
    ids: list[str] = []
    while len(ids) < count:
        if next_number not in used_numbers:
            ids.append(f"{domain}_T{tier}_{next_number:04d}")
            used_numbers.add(next_number)
        next_number += 1
    return ids


def _replacement_final_records(
    *,
    domain: str,
    tier: int,
    candidates: list[dict[str, Any]],
    existing_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fresh_ids = _next_fresh_ids(existing_records, domain, tier, len(candidates))
    records: list[dict[str, Any]] = []
    for new_id, candidate in zip(fresh_ids, candidates, strict=True):
        records.append(
            {
                "id": new_id,
                "domain": domain,
                "tier": tier,
                "question": str(candidate["question"]).strip(),
                "ground_truth": str(candidate["ground_truth"]).strip(),
                "sources": list(candidate.get("sources", [])),
                "difficulty_rationale": str(candidate.get("difficulty_rationale", "")).strip(),
                "metadata": {
                    "verified": bool(candidate.get("verified")),
                    "verification_answer": candidate.get("verification_answer"),
                    "verification_match": candidate.get("verification_match"),
                    "verification_model": candidate.get("verification_model"),
                    "subtopic": candidate.get("subtopic"),
                    "raw_id": candidate.get("id"),
                },
            }
        )
    return records


async def _verify_candidates(
    *,
    candidates: list[dict[str, Any]],
    rpm: int,
    batch_size: int,
    max_tokens: int,
    timeout_s: int,
    logger: Any,
    cell_name: str,
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
        except Exception as exc:
            logger.warning("[%s swap] Verify fallback on batch %s/%s: %s", cell_name, index, len(batches), exc)
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
            logger.info("[%s swap] Verify batch %s/%s complete", cell_name, index, len(batches))
    return verified


def _rebuild_manifest(
    *,
    output_dir: Path,
    config: ProjectConfig,
    generation_hours: float,
) -> dict[str, Any]:
    raw_candidates: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for tier in TIERS:
            raw_candidates.extend(read_jsonl(_cell_raw_path(output_dir, domain, tier)))

    valid_candidates: list[dict[str, Any]] = []
    validation_failures = 0
    for candidate in raw_candidates:
        errors = validate_candidate(candidate)
        if errors:
            validation_failures += 1
            continue
        valid_candidates.append(candidate)

    deduplicated = deduplicate_queries(valid_candidates, threshold=0.80)
    final_records: list[dict[str, Any]] = []
    verified_records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        final_records.extend(read_jsonl(_cell_output_path(output_dir, domain)))
        verified_records.extend(read_jsonl(_verified_domain_path(output_dir, domain)))

    verification_matches = sum(1 for item in verified_records if bool(item.get("verified")))
    verification_rate = verification_matches / len(verified_records) if verified_records else 0.0
    per_cell_counts = {
        _cell_name(domain, tier): 0
        for domain in DOMAINS
        for tier in TIERS
    }
    for record in final_records:
        per_cell_counts[_cell_name(str(record["domain"]), int(record["tier"]))] += 1
    cells_below_target = [name for name, count in per_cell_counts.items() if count < config.queries_per_cell]

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
    manifest["generation_model"] = "mixed"
    manifest["validation_failures"] = validation_failures
    manifest["parse_failures"] = 0
    manifest["parse_failure_rate"] = 0.0
    manifest["generation_models_by_cell"] = {
        GENERATION_MODEL: [_cell_name(domain, tier) for domain, tier in TARGET_CELLS],
        "meta/llama-4-maverick-17b-128e-instruct": [
            _cell_name(domain, tier)
            for domain in DOMAINS
            for tier in TIERS
            if (domain, tier) not in TARGET_CELLS
        ],
    }
    manifest["regenerated_cells"] = [_cell_name(domain, tier) for domain, tier in TARGET_CELLS]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


async def run_replacements(
    *,
    output_dir: Path,
    work_dir: Path,
    report_path: Path,
    cells: list[tuple[str, int]],
    generation_rpm: int,
    generation_batch_size: int,
    generation_max_tokens: int,
    generation_timeout_s: int,
    verification_rpm: int,
    verification_timeout_s: int,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = ProjectConfig()
    config.queries_per_cell = 250
    config.verification_model = VERIFICATION_MODEL
    config.verification_batch_size = 5
    config.verification_rpm = verification_rpm
    config.verification_concurrency = 1
    config.verification_timeout_s = verification_timeout_s

    logger = setup_logger("errorquake.flagged_replacements", output_dir / "logs")
    started = time.perf_counter()
    progress_path = work_dir / "flagged_replacements_progress.json"
    progress: dict[str, Any] = {
        "generation_model": GENERATION_MODEL,
        "generation_base_url": GENERATION_BASE_URL,
        "generation_timeout_s": generation_timeout_s,
        "generation_batch_size": generation_batch_size,
        "generation_max_tokens": generation_max_tokens,
        "verification_model": VERIFICATION_MODEL,
        "strictly_sequential": True,
        "cells": [_cell_name(domain, tier) for domain, tier in cells],
        "completed_cells": [],
        "cell_summaries": {},
    }
    existing_progress = _load_progress(progress_path)
    if existing_progress:
        progress.update(existing_progress)
        progress["generation_model"] = GENERATION_MODEL
        progress["generation_base_url"] = GENERATION_BASE_URL
        progress["generation_timeout_s"] = generation_timeout_s
        progress["generation_batch_size"] = generation_batch_size
        progress["generation_max_tokens"] = generation_max_tokens
        progress["verification_model"] = VERIFICATION_MODEL
        progress["strictly_sequential"] = True
        progress["cells"] = [_cell_name(domain, tier) for domain, tier in cells]
        progress.setdefault("completed_cells", [])
        progress.setdefault("cell_summaries", {})
    _save_progress(progress_path, progress)

    for domain, tier in cells:
        cell_name = _cell_name(domain, tier)
        if cell_name in progress["completed_cells"]:
            logger.info("[%s swap] Already completed in progress file; skipping", cell_name)
            continue
        logger.info("[%s swap] Starting", cell_name)
        progress["current_cell"] = cell_name
        _save_progress(progress_path, progress)
        domain_key = domain
        tier_key = "t5" if tier == 5 else "t1"
        expected_flagged_count = int(report["per_domain"][domain_key][tier_key]["likely_miscalibrated"])

        domain_records = read_jsonl(_cell_output_path(output_dir, domain))
        cell_records = [
            record for record in domain_records if int(record.get("tier", 0)) == tier
        ]
        flagged = collect_flagged_records(cell_records)[tier_key]
        flagged_ids = [str(item["id"]) for item in flagged]
        if len(flagged_ids) != expected_flagged_count:
            raise RuntimeError(
                f"{cell_name}: flagged count mismatch; report={expected_flagged_count} recomputed={len(flagged_ids)}"
            )
        if not flagged_ids:
            logger.info("[%s swap] No flagged IDs; skipping", cell_name)
            progress["completed_cells"].append(cell_name)
            progress["cell_summaries"][cell_name] = {"flagged_count": 0, "replaced_count": 0}
            _save_progress(progress_path, progress)
            continue

        preserved_records = [record for record in cell_records if str(record.get("id")) not in flagged_ids]
        target_new_candidates = len(flagged_ids) * 2

        system_prompt, domain_prompt = _load_prompt_text(config.prompts_dir, domain)
        rate_limiter = RateLimiter(rpm=generation_rpm)
        cell_dir = work_dir / cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)
        raw_candidates_path = cell_dir / "raw_candidates.jsonl"
        raw_candidates: list[dict[str, Any]] = []
        if raw_candidates_path.exists():
            raw_candidates = read_jsonl(raw_candidates_path)[:target_new_candidates]
            logger.info(
                "[%s swap] Resuming with %s existing raw candidates",
                cell_name,
                len(raw_candidates),
            )

        regen_tag = now_iso().replace(":", "").replace("-", "").replace("+", "_")
        batch_index = 0
        while len(raw_candidates) < target_new_candidates:
            batch_index += 1
            size = min(generation_batch_size, target_new_candidates - len(raw_candidates))
            previous_subtopics = [
                str(item.get("subtopic", "")).strip()
                for item in raw_candidates
                if str(item.get("subtopic", "")).strip()
            ]
            logger.info(
                "[%s swap] Generation batch %s starting | target=%s/%s | model=%s",
                cell_name,
                batch_index,
                len(raw_candidates),
                target_new_candidates,
                GENERATION_MODEL,
            )
            batch = await _generate_batch_deepseek_chat(
                system_prompt=system_prompt,
                domain_prompt=domain_prompt,
                tier=tier,
                batch_size=size,
                previous_subtopics=previous_subtopics,
                rate_limiter=rate_limiter,
                max_tokens=generation_max_tokens,
                timeout_s=generation_timeout_s,
            )
            normalised_batch: list[dict[str, Any]] = []
            for offset, item in enumerate(batch, start=1):
                normalised_batch.append(
                    _normalise_candidate(
                        item,
                        domain=domain,
                        tier=tier,
                        candidate_id=f"RAW_{domain}_T{tier}_SWAP_{regen_tag}_{len(raw_candidates) + offset:05d}",
                    )
                )
            raw_candidates.extend(normalised_batch)
            logger.info(
                "[%s swap] Generation batch %s complete | generated=%s/%s",
                cell_name,
                batch_index,
                len(raw_candidates),
                target_new_candidates,
            )

        _rewrite_jsonl(raw_candidates_path, raw_candidates)

        valid_candidates, validation_failures = _validate_candidates_local(
            raw_candidates,
            cell_dir / "validation_discards.jsonl",
        )
        deduped = deduplicate_queries(valid_candidates, threshold=0.80)
        deduped = _similarity_filter(
            [_preserved_candidate_view(record) for record in preserved_records],
            deduped,
            threshold=0.80,
        )
        verified = await _verify_candidates(
            candidates=deduped,
            rpm=verification_rpm,
            batch_size=config.verification_batch_size,
            max_tokens=config.verification_max_tokens,
            timeout_s=verification_timeout_s,
            logger=logger,
            cell_name=cell_name,
        )
        selected = select_final(verified, target_per_cell=len(flagged_ids))
        if len(selected) < len(flagged_ids):
            raise RuntimeError(
                f"{cell_name}: only selected {len(selected)} replacements for {len(flagged_ids)} flagged IDs."
            )

        replacement_records = _replacement_final_records(
            domain=domain,
            tier=tier,
            candidates=selected,
            existing_records=cell_records,
        )
        updated_domain_records = [record for record in domain_records if str(record.get("id")) not in flagged_ids]
        updated_domain_records.extend(replacement_records)
        updated_domain_records = sorted(
            updated_domain_records,
            key=lambda item: (int(item.get("tier", 0)), str(item.get("id", ""))),
        )
        _rewrite_jsonl(_cell_output_path(output_dir, domain), updated_domain_records)

        existing_verified = read_jsonl(_verified_domain_path(output_dir, domain))
        existing_verified.extend(verified)
        existing_verified = sorted(
            existing_verified,
            key=lambda item: (int(item.get("tier", 0)), str(item.get("id", ""))),
        )
        _rewrite_jsonl(_verified_domain_path(output_dir, domain), existing_verified)

        existing_raw = read_jsonl(_cell_raw_path(output_dir, domain, tier))
        existing_raw.extend(raw_candidates)
        _rewrite_jsonl(_cell_raw_path(output_dir, domain, tier), existing_raw)

        _save_json(cell_dir / "flagged_ids.json", flagged_ids)
        _rewrite_jsonl(cell_dir / "verified_candidates.jsonl", verified)
        _rewrite_jsonl(cell_dir / "replacement_records.jsonl", replacement_records)

        summary = {
            "domain": domain,
            "tier": tier,
            "flagged_count": len(flagged_ids),
            "generated_raw_candidates": len(raw_candidates),
            "validated_candidates": len(valid_candidates),
            "validation_failures": validation_failures,
            "deduplicated_candidates": len(deduped),
            "verified_candidates": len(verified),
            "selected_replacements": len(replacement_records),
            "removed_ids": flagged_ids,
            "inserted_ids": [record["id"] for record in replacement_records],
        }
        _save_json(cell_dir / "summary.json", summary)
        progress["completed_cells"].append(cell_name)
        progress["cell_summaries"][cell_name] = summary
        progress["current_cell"] = None
        _save_progress(progress_path, progress)
        logger.info("[%s swap] Complete | replaced=%s", cell_name, len(replacement_records))

    manifest = _rebuild_manifest(
        output_dir=output_dir,
        config=config,
        generation_hours=(time.perf_counter() - started) / 3600.0,
    )
    progress["manifest"] = manifest
    progress["current_cell"] = None
    _save_progress(progress_path, progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace only flagged queries in selected cells.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "queries",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "data" / "queries" / "regeneration_work" / "flagged_swaps",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "data" / "queries" / "tier_audit_report.json",
    )
    parser.add_argument("--cells", default=None)
    parser.add_argument("--generation-rpm", type=int, default=35)
    parser.add_argument("--generation-batch-size", type=int, default=12)
    parser.add_argument("--generation-max-tokens", type=int, default=2500)
    parser.add_argument("--generation-timeout", type=int, default=90)
    parser.add_argument("--verification-rpm", type=int, default=35)
    parser.add_argument("--verification-timeout", type=int, default=90)
    args = parser.parse_args()

    summary = asyncio.run(
        run_replacements(
            output_dir=args.output_dir,
            work_dir=args.work_dir,
            report_path=args.report_path,
            cells=_parse_cells(args.cells),
            generation_rpm=args.generation_rpm,
            generation_batch_size=args.generation_batch_size,
            generation_max_tokens=args.generation_max_tokens,
            generation_timeout_s=args.generation_timeout,
            verification_rpm=args.verification_rpm,
            verification_timeout_s=args.verification_timeout,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
