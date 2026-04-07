"""Query generation pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from errorquake.queries import DOMAINS, TIERS, Query
from errorquake.utils import ProjectConfig, now_iso, read_jsonl, setup_logger, write_jsonl

YES_NO_PREFIXES = ("is", "does", "are", "can", "will", "has", "was", "did")
GENERATION_ERROR_LOG = "generation_errors.jsonl"
VALIDATION_DISCARDS_LOG = "validation_discards.jsonl"
GENERATION_PROGRESS_FILE = "generation_progress.json"

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9./-]*")
_NUMBER_RE = re.compile(r"\b\d[\d,./:-]*\b")
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,}[A-Z0-9.-]*)(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}[A-Z0-9.-]*|\d{4}))*\b"
)

_NIM_CLIENT: Any | None = None
_NIM_CLIENT_KEY: str | None = None


class GenerationParseError(RuntimeError):
    """Raised when generation output cannot be parsed into candidate records."""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


class GenerationRequestError(RuntimeError):
    """Raised when the generation backend fails after retries."""


class RateLimiter:
    """Token-bucket style rate limiter with a safety margin under free-tier limits."""

    def __init__(
        self,
        rpm: int = 35,
        clock: Any | None = None,
        sleep_fn: Any | None = None,
    ) -> None:
        self.interval = 60.0 / rpm
        self.clock = clock or time.monotonic
        self.sleep_fn = sleep_fn or asyncio.sleep
        self.last_call = 0.0

    async def acquire(self) -> None:
        now = self.clock()
        wait = self.interval - (now - self.last_call)
        if wait > 0:
            await self.sleep_fn(wait)
        self.last_call = self.clock()


def _get_nvidia_api_key() -> str:
    for env_name in ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise RuntimeError("Missing API key: NVIDIA_API_KEY")


def _get_nim_client() -> Any:
    from openai import AsyncOpenAI

    global _NIM_CLIENT, _NIM_CLIENT_KEY
    api_key = _get_nvidia_api_key()
    if _NIM_CLIENT is None or _NIM_CLIENT_KEY != api_key:
        _NIM_CLIENT = AsyncOpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        _NIM_CLIENT_KEY = api_key
    return _NIM_CLIENT


def _build_batch_prompt(
    system_prompt: str,
    domain_prompt: str,
    tier: int,
    batch_size: int,
    previous_subtopics: list[str] | None = None,
) -> str:
    diversity_instruction = ""
    if previous_subtopics:
        counts = Counter(topic.strip() for topic in previous_subtopics if topic.strip())
        summary = ", ".join(f"{topic}({count})" for topic, count in counts.most_common(10))
        diversity_instruction = (
            "Avoid overusing previously seen subtopics. "
            f"Subtopics already covered: {summary}. Favor fresh subtopics and varied framing."
        )

    return (
        f"{system_prompt.strip()}\n\n"
        f"{domain_prompt.strip()}\n\n"
        f"Generate exactly {batch_size} questions at difficulty tier T{tier}.\n\n"
        "Return a JSON array only. Each item must be an object with keys:\n"
        '- "question": a standalone factual question ending with "?".\n'
        '- "ground_truth": a concise 1-3 sentence factual answer.\n'
        '- "sources": a list with at least 2 plausible source references.\n'
        '- "difficulty_rationale": why this belongs in the requested tier.\n'
        '- "subtopic": a short subtopic label.\n\n'
        f"{diversity_instruction}\n\n"
        "Respond with a JSON array only. No other text."
    ).strip()


async def _call_nim_text(
    model_id: str,
    prompt: str,
    max_tokens: int,
    rate_limiter: RateLimiter | None = None,
    temperature: float = 0.0,
    timeout_s: int = 60,
) -> str:
    client = _get_nim_client()
    rate_limit_delay = 30
    rate_limit_attempts = 0
    timeout_attempts = 0
    server_attempts = 0

    while True:
        if rate_limiter is not None:
            await rate_limiter.acquire()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout_s,
            )
            return response.choices[0].message.content or ""
        except asyncio.TimeoutError as exc:
            if timeout_attempts < 1:
                timeout_attempts += 1
                await asyncio.sleep(5)
                continue
            raise GenerationRequestError(f"Timed out after {timeout_s} seconds.") from exc
        except Exception as exc:
            message = str(exc)
            lower = message.lower()
            if "429" in lower or "rate limit" in lower:
                if rate_limit_attempts >= 5:
                    raise GenerationRequestError(message) from exc
                await asyncio.sleep(rate_limit_delay)
                rate_limit_delay = min(rate_limit_delay * 2, 120)
                rate_limit_attempts += 1
                continue
            if any(token in lower for token in ("500", "502", "503", "bad gateway", "service unavailable")):
                if server_attempts >= 3:
                    raise GenerationRequestError(message) from exc
                await asyncio.sleep(10)
                server_attempts += 1
                continue
            raise GenerationRequestError(message) from exc


def _strip_json_fences(raw: str) -> str:
    return _JSON_FENCE_RE.sub("", raw.strip())


def _flatten_candidate_payload(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        flattened: list[dict[str, Any]] = []
        for item in value:
            flattened.extend(_flatten_candidate_payload(item))
        return flattened
    if isinstance(value, dict):
        for key in ("queries", "items", "data", "candidates", "answers", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return _flatten_candidate_payload(nested)
        return [value]
    return []


def _salvage_truncated_array(cleaned: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    start = cleaned.find("[")
    if start == -1:
        return []

    items: list[dict[str, Any]] = []
    index = start + 1
    while index < len(cleaned):
        while index < len(cleaned) and cleaned[index] in " \r\n\t,":
            index += 1
        if index >= len(cleaned) or cleaned[index] == "]":
            break
        try:
            value, end = decoder.raw_decode(cleaned, index)
        except json.JSONDecodeError:
            break
        items.extend(_flatten_candidate_payload(value))
        index = end
    return items


def parse_generation_response(raw: str) -> list[dict[str, Any]]:
    cleaned = _strip_json_fences(raw)
    if not cleaned:
        raise GenerationParseError("Empty response.", raw)

    try:
        parsed = json.loads(cleaned)
        flattened = _flatten_candidate_payload(parsed)
    except json.JSONDecodeError:
        flattened = _salvage_truncated_array(cleaned)

    if not flattened:
        raise GenerationParseError("Response did not contain a JSON array of objects.", raw)
    return [item for item in flattened if isinstance(item, dict)]


def parse_verification_batch_response(raw: str) -> dict[str, str]:
    cleaned = _strip_json_fences(raw)
    if not cleaned:
        raise GenerationParseError("Empty verification response.", raw)

    try:
        parsed = json.loads(cleaned)
        flattened = _flatten_candidate_payload(parsed)
    except json.JSONDecodeError:
        flattened = _salvage_truncated_array(cleaned)

    answers: dict[str, str] = {}
    for item in flattened:
        if not isinstance(item, dict):
            continue
        candidate_id = str(
            item.get("id") or item.get("query_id") or item.get("question_id") or ""
        ).strip()
        answer = item.get("answer") or item.get("response") or item.get("text")
        if candidate_id and isinstance(answer, str) and answer.strip():
            answers[candidate_id] = answer.strip()

    if not answers:
        raise GenerationParseError("Verification response did not contain any usable answers.", raw)
    return answers


def _build_verification_batch_prompt(queries: list[dict[str, Any]]) -> str:
    question_block = "\n\n".join(
        f"ID: {query['id']}\nQuestion: {query['question']}" for query in queries
    )
    return (
        "Answer each question accurately and concisely in 1-3 sentences.\n"
        "Return a JSON array only. Each item must have keys: id, answer.\n"
        "Do not include any commentary outside the JSON.\n\n"
        f"{question_block}"
    )


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _normalise_candidate(
    candidate: dict[str, Any],
    *,
    domain: str,
    tier: int,
    candidate_id: str,
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "domain": domain,
        "tier": tier,
        "question": str(candidate.get("question", "")).strip(),
        "ground_truth": str(candidate.get("ground_truth", "")).strip(),
        "sources": [
            str(source).strip()
            for source in candidate.get("sources", [])
            if str(source).strip()
        ],
        "difficulty_rationale": str(candidate.get("difficulty_rationale", "")).strip(),
        "subtopic": str(candidate.get("subtopic", "")).strip(),
        "generated_at": now_iso(),
    }


def _load_prompt_text(prompts_dir: Path, domain: str) -> tuple[str, str]:
    system_prompt = (prompts_dir / "query_generation" / "system.txt").read_text(encoding="utf-8")
    domain_prompt = (prompts_dir / "query_generation" / f"{domain.lower()}.txt").read_text(
        encoding="utf-8"
    )
    return system_prompt, domain_prompt


async def _generate_batch(
    system_prompt: str,
    domain_prompt: str,
    tier: int,
    batch_size: int = 25,
    previous_subtopics: list[str] | None = None,
    model: str = "deepseek-ai/deepseek-v3.2",
    rate_limiter: RateLimiter | None = None,
    max_tokens: int = 4000,
    timeout_s: int = 120,
) -> list[dict[str, Any]]:
    """
    Generate one batch of candidate queries through NIM.
    """
    prompt = _build_batch_prompt(
        system_prompt=system_prompt,
        domain_prompt=domain_prompt,
        tier=tier,
        batch_size=batch_size,
        previous_subtopics=previous_subtopics,
    )
    raw_response = await _call_nim_text(
        model_id=model,
        prompt=prompt,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        timeout_s=timeout_s,
    )
    return parse_generation_response(raw_response)[:batch_size]


def _batch_sizes(total: int, batch_size: int = 25) -> list[int]:
    sizes: list[int] = []
    remaining = total
    while remaining > 0:
        size = min(batch_size, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


async def generate_domain_queries(
    domain: str,
    tier: int,
    n: int,
    prompts_dir: Path,
    model: str = "deepseek-ai/deepseek-v3.2",
    oversample_factor: int = 2,
    rate_limiter: RateLimiter | None = None,
    batch_size: int = 25,
) -> list[dict[str, Any]]:
    """
    Generate n queries for a single domain-tier cell.
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}")
    if tier not in TIERS:
        raise ValueError(f"Unknown tier: {tier}")
    if n <= 0:
        return []

    system_prompt, domain_prompt = _load_prompt_text(prompts_dir, domain)
    candidates: list[dict[str, Any]] = []
    topics_seen: list[str] = []
    generated_count = 0
    for size in _batch_sizes(n * oversample_factor, batch_size=batch_size):
        batch = await _generate_batch(
            system_prompt=system_prompt,
            domain_prompt=domain_prompt,
            tier=tier,
            batch_size=size,
            previous_subtopics=topics_seen,
            model=model,
            rate_limiter=rate_limiter,
        )
        normalised_batch: list[dict[str, Any]] = []
        for item in batch:
            generated_count += 1
            normalised_batch.append(
                _normalise_candidate(
                    item,
                    domain=domain,
                    tier=tier,
                    candidate_id=f"RAW_{domain}_T{tier}_{generated_count:05d}",
                )
            )
        candidates.extend(normalised_batch)
        topics_seen.extend(
            item.get("subtopic", "").strip()
            for item in normalised_batch
            if item.get("subtopic", "").strip()
        )
    return candidates


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    pieces = [part for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
    return len(pieces) or 1


def _is_yes_no_question(question: str) -> bool:
    words = re.findall(r"[A-Za-z0-9']+", question)
    return bool(words) and words[0].lower() in YES_NO_PREFIXES and len(words) <= 12


def validate_candidate(candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field_name in ("question", "ground_truth", "sources", "difficulty_rationale", "subtopic"):
        if field_name not in candidate:
            errors.append(f"Missing field: {field_name}")

    question = str(candidate.get("question", "")).strip()
    if not question:
        errors.append("Empty question")
    elif not question.endswith("?"):
        errors.append("Question must end with '?'")
    elif _is_yes_no_question(question):
        errors.append("Yes/no question")

    ground_truth = str(candidate.get("ground_truth", "")).strip()
    if not ground_truth:
        errors.append("Empty ground truth")
    else:
        sentence_count = _sentence_count(ground_truth)
        if not 1 <= sentence_count <= 3:
            errors.append("Ground truth must be 1-3 sentences")
        if re.fullmatch(r"[\d,.%-]+", ground_truth):
            errors.append("Ground truth cannot be a bare number")
        if len(ground_truth) > 500:
            errors.append("Ground truth is too long")

    sources = candidate.get("sources", [])
    if not isinstance(sources, list) or len([item for item in sources if str(item).strip()]) < 2:
        errors.append("Need at least 2 sources")

    if not str(candidate.get("difficulty_rationale", "")).strip():
        errors.append("Empty difficulty rationale")
    if not str(candidate.get("subtopic", "")).strip():
        errors.append("Empty subtopic")
    return errors


def _normalise_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def extract_key_entities(text: str) -> list[str]:
    seen: list[str] = []
    seen_normalised: set[str] = set()
    for pattern in (_NUMBER_RE, _ENTITY_RE):
        for match in pattern.findall(text):
            normalised = _normalise_text(match)
            if normalised and normalised not in seen_normalised:
                seen.append(match.strip())
                seen_normalised.add(normalised)
    return seen


def heuristic_match(ground_truth: str, verification_answer: str) -> bool:
    key_entities = extract_key_entities(ground_truth)
    normalised_answer = _normalise_text(verification_answer)
    if key_entities:
        matches = sum(
            1
            for entity in key_entities
            if _normalise_text(entity) and _normalise_text(entity) in normalised_answer
        )
        return matches / len(key_entities) >= 0.6

    answer_tokens = {
        token for token in _WORD_RE.findall(normalised_answer) if len(token) >= 4 and not token.isdigit()
    }
    truth_tokens = {
        token
        for token in _WORD_RE.findall(_normalise_text(ground_truth))
        if len(token) >= 4 and not token.isdigit()
    }
    if not truth_tokens:
        return _normalise_text(ground_truth) in normalised_answer
    return len(truth_tokens & answer_tokens) / len(truth_tokens) >= 0.6


async def verify_queries_batch(
    queries: list[dict[str, Any]],
    *,
    model: str = "qwen/qwen3-next-80b-a3b-instruct",
    rate_limiter: RateLimiter | None = None,
    max_tokens: int = 1400,
    timeout_s: int = 90,
) -> list[dict[str, Any]]:
    """
    Verify a small batch of queries in one model call, with per-item fallback if parsing is partial.
    """
    if not queries:
        return []

    prompt = _build_verification_batch_prompt(queries)
    raw_response = await _call_nim_text(
        model_id=model,
        prompt=prompt,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        timeout_s=timeout_s,
    )
    answers = parse_verification_batch_response(raw_response)

    verified_batch: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for query in queries:
        candidate_id = str(query.get("id"))
        answer = answers.get(candidate_id)
        if answer is None:
            missing.append(query)
            continue
        match = heuristic_match(str(query.get("ground_truth", "")), answer)
        verified_batch.append(
            {
                **query,
                "verified": match,
                "verification_answer": answer,
                "verification_match": match,
                "verification_model": model,
            }
        )

    for query in missing:
        verified_batch.append(
            await verify_query(
                query,
                model=model,
                rate_limiter=rate_limiter,
                max_tokens=min(300, max_tokens),
                timeout_s=timeout_s,
            )
        )
    return verified_batch


async def verify_query(
    query: dict[str, Any],
    model: str = "qwen/qwen3-next-80b-a3b-instruct",
    rate_limiter: RateLimiter | None = None,
    max_tokens: int = 300,
    timeout_s: int = 60,
) -> dict[str, Any]:
    """
    Ask a second NIM model to answer the question and compare its answer heuristically.
    """
    prompt = (
        "Answer this question accurately and concisely in 1-3 sentences.\n\n"
        f"Question: {query['question']}\n\n"
        "Respond with ONLY your answer, no preamble."
    )
    verification_answer = await _call_nim_text(
        model_id=model,
        prompt=prompt,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        timeout_s=timeout_s,
    )
    match = heuristic_match(str(query.get("ground_truth", "")), verification_answer)
    verified_query = dict(query)
    verified_query.update(
        {
            "verified": bool(match),
            "verification_answer": verification_answer,
            "verification_match": bool(match),
            "verification_model": model,
        }
    )
    return verified_query


def deduplicate_queries(queries: list[dict[str, Any]], threshold: float = 0.80) -> list[dict[str, Any]]:
    """
    Deduplicate semantically similar questions within each domain using TF-IDF cosine similarity.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    grouped: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        key = str(query.get("domain", "__all__"))
        grouped.setdefault(key, []).append(query)

    deduplicated: list[dict[str, Any]] = []
    for _, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda item: (
                -len(str(item.get("ground_truth", ""))),
                not bool(item.get("verified")),
                str(item.get("question", "")),
            ),
        )
        if len(ordered) <= 1:
            deduplicated.extend(ordered)
            continue
        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf = vectorizer.fit_transform([str(item.get("question", "")) for item in ordered])
            similarity = cosine_similarity(tfidf)
        except ValueError:
            deduplicated.extend(ordered)
            continue

        kept_indices: list[int] = []
        for index, candidate in enumerate(ordered):
            if any(similarity[index, kept] > threshold for kept in kept_indices):
                continue
            kept_indices.append(index)
            deduplicated.append(candidate)
    return deduplicated


def select_final(
    candidates: list[dict[str, Any]],
    target_per_cell: int = 250,
    diversity_cap_ratio: float = 0.20,
) -> list[dict[str, Any]]:
    """
    Select final candidates using verification priority and soft diversity caps.
    """
    ranked = sorted(
        candidates,
        key=lambda item: (
            not bool(item.get("verified")),
            -len(str(item.get("ground_truth", ""))),
            str(item.get("question", "")),
        ),
    )
    cap = max(1, int(target_per_cell * diversity_cap_ratio))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    subtopic_counts: Counter[str] = Counter()

    def maybe_select(require_verified: bool, require_diverse: bool) -> None:
        for candidate in ranked:
            if len(selected) >= target_per_cell:
                return
            candidate_id = str(candidate.get("id", id(candidate)))
            if candidate_id in selected_ids:
                continue
            verified = bool(candidate.get("verified"))
            if require_verified and not verified:
                continue
            if not require_verified and verified:
                continue
            subtopic = str(candidate.get("subtopic", "misc")).strip() or "misc"
            if require_diverse and subtopic_counts[subtopic] >= cap:
                continue
            selected.append(candidate)
            selected_ids.add(candidate_id)
            subtopic_counts[subtopic] += 1

    maybe_select(require_verified=True, require_diverse=True)
    maybe_select(require_verified=True, require_diverse=False)
    maybe_select(require_verified=False, require_diverse=True)
    maybe_select(require_verified=False, require_diverse=False)
    return selected[:target_per_cell]


def assign_final_ids(
    domain: str,
    tier: int,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        query = Query(
            id=f"{domain}_T{tier}_{index:04d}",
            domain=domain,
            tier=tier,
            question=str(candidate["question"]).strip(),
            ground_truth=str(candidate["ground_truth"]).strip(),
            sources=list(candidate.get("sources", [])),
            difficulty_rationale=str(candidate.get("difficulty_rationale", "")).strip(),
            metadata={
                "verified": bool(candidate.get("verified")),
                "verification_answer": candidate.get("verification_answer"),
                "verification_match": candidate.get("verification_match"),
                "verification_model": candidate.get("verification_model"),
                "subtopic": candidate.get("subtopic"),
                "raw_id": candidate.get("id"),
            },
        )
        records.append(query.to_dict())
    return records


def _rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    if records:
        write_jsonl(path, records)


def _raw_cell_path(raw_dir: Path, domain: str, tier: int) -> Path:
    return raw_dir / f"{domain}_T{tier}.jsonl"


def _verified_domain_path(verified_dir: Path, domain: str) -> Path:
    return verified_dir / f"{domain.lower()}.jsonl"


def _load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "completed_cells": [],
            "current_cell": None,
            "batches_done_in_current_cell": 0,
            "total_candidates": 0,
            "parse_failures": 0,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _log_jsonl(path: Path, record: dict[str, Any]) -> None:
    write_jsonl(path, [record])


def _summarise_subtopics(candidates: list[dict[str, Any]]) -> str:
    counts = Counter(
        str(candidate.get("subtopic", "")).strip()
        for candidate in candidates
        if str(candidate.get("subtopic", "")).strip()
    )
    return ", ".join(f"{topic}({count})" for topic, count in counts.most_common(8))


def _normalise_generated_batch(
    batch: list[dict[str, Any]],
    *,
    domain: str,
    tier: int,
    start_index: int,
) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for offset, candidate in enumerate(batch, start=1):
        normalised.append(
            _normalise_candidate(
                candidate,
                domain=domain,
                tier=tier,
                candidate_id=f"RAW_{domain}_T{tier}_{start_index + offset:05d}",
            )
        )
    return normalised


async def _generate_cell_candidates(
    *,
    domain: str,
    tier: int,
    prompts_dir: Path,
    raw_dir: Path,
    errors_path: Path,
    progress_path: Path,
    config: ProjectConfig,
    rate_limiter: RateLimiter,
    resume: bool,
    logger: Any,
) -> tuple[list[dict[str, Any]], int]:
    system_prompt, domain_prompt = _load_prompt_text(prompts_dir, domain)
    progress = _load_progress(progress_path)
    raw_path = _raw_cell_path(raw_dir, domain, tier)
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    total_target = config.queries_per_cell * config.oversample_factor
    batch_sizes = _batch_sizes(total_target, batch_size=config.generation_batch_size)
    total_batches = len(batch_sizes)

    completed_cells = {tuple(item) for item in progress.get("completed_cells", [])}
    if (domain, tier) in completed_cells:
        return existing, 0

    batches_done = 0
    current_cell = progress.get("current_cell")
    if resume and current_cell == [domain, tier]:
        batches_done = int(progress.get("batches_done_in_current_cell", 0))
    elif resume and existing:
        batches_done = min(len(existing) // config.generation_batch_size, total_batches)

    candidates = list(existing)
    parse_failures = 0
    skipped_batches: list[int] = []

    async def run_batch(batch_index: int) -> bool:
        nonlocal candidates, parse_failures, progress
        previous_subtopics = [
            str(item.get("subtopic", "")).strip()
            for item in candidates
            if str(item.get("subtopic", "")).strip()
        ]
        batch_size = batch_sizes[batch_index]
        try:
            generated = await _generate_batch(
                system_prompt=system_prompt,
                domain_prompt=domain_prompt,
                tier=tier,
                batch_size=batch_size,
                previous_subtopics=previous_subtopics,
                model=config.generation_model,
                rate_limiter=rate_limiter,
                max_tokens=config.generation_max_tokens,
                timeout_s=config.generation_timeout_s,
            )
        except GenerationParseError as exc:
            parse_failures += 1
            _log_jsonl(
                errors_path,
                {
                    "timestamp": now_iso(),
                    "domain": domain,
                    "tier": tier,
                    "batch_index": batch_index + 1,
                    "error_type": "parse_failure",
                    "error": str(exc),
                    "raw_response": exc.raw_response,
                },
            )
            progress["current_cell"] = [domain, tier]
            progress["batches_done_in_current_cell"] = batch_index + 1
            progress["parse_failures"] = int(progress.get("parse_failures", 0)) + 1
            _save_progress(progress_path, progress)
            return True
        except GenerationRequestError as exc:
            _log_jsonl(
                errors_path,
                {
                    "timestamp": now_iso(),
                    "domain": domain,
                    "tier": tier,
                    "batch_index": batch_index + 1,
                    "error_type": "request_failure",
                    "error": str(exc),
                },
            )
            return False

        normalised = _normalise_generated_batch(
            generated,
            domain=domain,
            tier=tier,
            start_index=len(candidates),
        )
        candidates.extend(normalised)
        if normalised:
            write_jsonl(raw_path, normalised)
        progress["total_candidates"] = int(progress.get("total_candidates", 0)) + len(normalised)
        progress["current_cell"] = [domain, tier]
        progress["batches_done_in_current_cell"] = batch_index + 1
        progress["parse_failures"] = int(progress.get("parse_failures", 0))
        _save_progress(progress_path, progress)

        if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
            parsed_count = len(normalised)
            yield_pct = (parsed_count / batch_size * 100.0) if batch_size else 0.0
            logger.info(
                "[%s T%s] Batch %s/%s done | %s/%s parsed | %s parse failures | %.0f%% yield",
                domain,
                tier,
                batch_index + 1,
                total_batches,
                parsed_count,
                batch_size,
                parse_failures,
                yield_pct,
            )
            summary = _summarise_subtopics(candidates)
            if summary:
                logger.info("[%s T%s] Subtopics seen: %s", domain, tier, summary)
        return True

    for batch_index in range(batches_done, total_batches):
        success = await run_batch(batch_index)
        if not success:
            skipped_batches.append(batch_index)
        if parse_failures / (batch_index + 1) > 0.20:
            raise RuntimeError(f"Parse failure rate exceeded 20% for {domain} T{tier}.")

    for batch_index in skipped_batches:
        await run_batch(batch_index)

    completed_cells.add((domain, tier))
    progress["completed_cells"] = [list(item) for item in sorted(completed_cells)]
    progress["current_cell"] = None
    progress["batches_done_in_current_cell"] = 0
    progress["parse_failures"] = int(progress.get("parse_failures", 0))
    _save_progress(progress_path, progress)
    return candidates, parse_failures


def _load_raw_candidates(raw_dir: Path, domains: list[str], tiers: list[int]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for domain in domains:
        for tier in tiers:
            candidates.extend(read_jsonl(_raw_cell_path(raw_dir, domain, tier)))
    return candidates


def _validate_candidates(
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
            _log_jsonl(
                discards_path,
                {
                    "timestamp": now_iso(),
                    "id": candidate.get("id"),
                    "domain": candidate.get("domain"),
                    "tier": candidate.get("tier"),
                    "question": candidate.get("question"),
                    "reasons": errors,
                },
            )
            continue
        valid.append(candidate)
    return valid, failures


async def _verify_candidates_for_domain(
    *,
    domain: str,
    candidates: list[dict[str, Any]],
    verified_dir: Path,
    config: ProjectConfig,
    rate_limiter: RateLimiter,
    skip_verify: bool,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    verified_path = _verified_domain_path(verified_dir, domain)
    existing_records = read_jsonl(verified_path)
    existing_by_id = {str(record.get("id")): record for record in existing_records}
    ordered_by_id: dict[str, dict[str, Any]] = dict(existing_by_id)
    pending = [
        candidate for candidate in candidates if str(candidate.get("id")) not in existing_by_id
    ]
    if not pending:
        return [ordered_by_id[str(candidate.get("id"))] for candidate in candidates]

    if skip_verify:
        new_records = []
        for candidate in pending:
            verified = dict(candidate)
            verified.update(
                {
                    "verified": False,
                    "verification_answer": "",
                    "verification_match": False,
                    "verification_model": config.verification_model,
                }
            )
            new_records.append(verified)
            ordered_by_id[str(candidate.get("id"))] = verified
        write_jsonl(verified_path, new_records)
        return [ordered_by_id[str(candidate.get("id"))] for candidate in candidates]

    batches = _chunked(pending, max(1, config.verification_batch_size))
    semaphore = asyncio.Semaphore(max(1, config.verification_concurrency))
    write_lock = asyncio.Lock()
    completed_batches = 0
    total_batches = len(batches)

    async def process_batch(batch: list[dict[str, Any]]) -> None:
        nonlocal completed_batches
        async with semaphore:
            try:
                verified_batch = await verify_queries_batch(
                    batch,
                    model=config.verification_model,
                    rate_limiter=rate_limiter,
                    max_tokens=config.verification_max_tokens,
                    timeout_s=config.verification_timeout_s,
                )
            except (GenerationParseError, GenerationRequestError):
                verified_batch = []
                for candidate in batch:
                    verified_batch.append(
                        await verify_query(
                            candidate,
                            model=config.verification_model,
                            rate_limiter=rate_limiter,
                            max_tokens=min(300, config.verification_max_tokens),
                            timeout_s=config.verification_timeout_s,
                        )
                    )

            async with write_lock:
                write_jsonl(verified_path, verified_batch)
                for item in verified_batch:
                    ordered_by_id[str(item.get("id"))] = item
                completed_batches += 1
                if logger and (completed_batches % 25 == 0 or completed_batches == total_batches):
                    logger.info(
                        "[%s verify] Batch %s/%s complete | %s queries verified",
                        domain,
                        completed_batches,
                        total_batches,
                        completed_batches * max(1, config.verification_batch_size),
                    )

    await asyncio.gather(*(process_batch(batch) for batch in batches))
    return [ordered_by_id[str(candidate.get("id"))] for candidate in candidates]


def _build_manifest(
    *,
    final_records: list[dict[str, Any]],
    domains: list[str],
    tiers: list[int],
    config: ProjectConfig,
    candidates_generated: int,
    candidates_after_validation: int,
    candidates_after_dedup: int,
    verification_rate: float,
    cells_below_target: list[str],
    generation_hours: float,
) -> dict[str, Any]:
    per_domain = {domain: 0 for domain in domains}
    per_tier = {f"T{tier}": 0 for tier in tiers}
    per_cell = {f"{domain}_T{tier}": 0 for domain in domains for tier in tiers}

    for record in final_records:
        query = Query.from_dict(record)
        per_domain[query.domain] += 1
        per_tier[f"T{query.tier}"] += 1
        per_cell[f"{query.domain}_T{query.tier}"] += 1

    return {
        "total_queries": len(final_records),
        "per_domain": per_domain,
        "per_tier": per_tier,
        "per_cell": per_cell,
        "generation_model": config.generation_model,
        "verification_model": config.verification_model,
        "generation_date": now_iso().split("T", 1)[0],
        "oversample_factor": config.oversample_factor,
        "candidates_generated": candidates_generated,
        "candidates_after_validation": candidates_after_validation,
        "candidates_after_dedup": candidates_after_dedup,
        "verification_rate": verification_rate,
        "cells_below_target": cells_below_target,
        "generation_hours": round(generation_hours, 2),
    }


def _merge_existing_domain_output(
    path: Path,
    new_records: list[dict[str, Any]],
    selected_tiers: list[int],
) -> list[dict[str, Any]]:
    if not path.exists():
        return new_records
    selected_tier_set = set(selected_tiers)
    preserved = [
        record
        for record in read_jsonl(path)
        if int(record.get("tier", 0)) not in selected_tier_set
    ]
    return preserved + new_records


async def run_full_generation(
    prompts_dir: Path,
    output_dir: Path,
    config: ProjectConfig,
    *,
    domains: list[str] | None = None,
    tiers: list[int] | None = None,
    rpm: int | None = None,
    skip_verify: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    verify_only: bool = False,
) -> dict[str, Any]:
    """
    Full pipeline: generate -> validate -> deduplicate -> verify -> select -> save.
    """
    selected_domains = domains or DOMAINS
    selected_tiers = tiers or TIERS
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    verified_dir = output_dir / "verified"
    errors_path = output_dir / GENERATION_ERROR_LOG
    discards_path = output_dir / VALIDATION_DISCARDS_LOG
    progress_path = raw_dir / GENERATION_PROGRESS_FILE

    plan = {
        "domains": selected_domains,
        "tiers": selected_tiers,
        "output_dir": str(output_dir),
        "generation_model": config.generation_model,
        "verification_model": config.verification_model,
        "rpm": rpm or config.generation_rpm,
        "oversample_factor": config.oversample_factor,
        "queries_per_cell": config.queries_per_cell,
        "skip_verify": skip_verify,
        "resume": resume,
        "verify_only": verify_only,
    }
    if dry_run:
        return plan

    logger = setup_logger("errorquake.generate", output_dir / "logs")
    start = time.perf_counter()
    generation_rate_limiter = RateLimiter(rpm=rpm or config.generation_rpm)
    verification_rate_limiter = RateLimiter(rpm=config.verification_rpm)

    if not resume:
        for domain in selected_domains:
            for tier in selected_tiers:
                raw_path = _raw_cell_path(raw_dir, domain, tier)
                if raw_path.exists():
                    raise RuntimeError(
                        f"Raw generation output already exists for {domain} T{tier}. "
                        "Use --resume or a different output directory."
                    )

    parse_failures = 0
    if not verify_only:
        for domain in selected_domains:
            for tier in selected_tiers:
                logger.info("[%s T%s] Starting...", domain, tier)
                _, cell_parse_failures = await _generate_cell_candidates(
                    domain=domain,
                    tier=tier,
                    prompts_dir=prompts_dir,
                    raw_dir=raw_dir,
                    errors_path=errors_path,
                    progress_path=progress_path,
                    config=config,
                    rate_limiter=generation_rate_limiter,
                    resume=resume,
                    logger=logger,
                )
                parse_failures += cell_parse_failures

    raw_candidates = _load_raw_candidates(raw_dir, selected_domains, selected_tiers)
    valid_candidates, validation_failures = _validate_candidates(raw_candidates, discards_path)
    deduplicated = deduplicate_queries(valid_candidates, threshold=0.80)

    logger.info("GENERATION COMPLETE")
    logger.info("  Total candidates: %s", len(raw_candidates))
    logger.info(
        "  Parse failures: %s (%.1f%%)",
        parse_failures,
        (parse_failures / max(len(raw_candidates), 1)) * 100.0,
    )
    logger.info(
        "  Validation failures: %s (%.1f%%)",
        validation_failures,
        (validation_failures / max(len(raw_candidates), 1)) * 100.0,
    )
    logger.info("  After dedup: %s", len(deduplicated))
    if not skip_verify:
        logger.info("  Verification: running...")

    final_records: list[dict[str, Any]] = []
    verification_candidates = 0
    verification_matches = 0
    cells_below_target: list[str] = []

    for domain in selected_domains:
        domain_candidates = [
            candidate for candidate in deduplicated if str(candidate.get("domain")) == domain
        ]
        verified_candidates = await _verify_candidates_for_domain(
            domain=domain,
            candidates=domain_candidates,
            verified_dir=verified_dir,
            config=config,
            rate_limiter=verification_rate_limiter,
            skip_verify=skip_verify,
            logger=logger,
        )
        verification_candidates += len(verified_candidates)
        verification_matches += sum(1 for item in verified_candidates if item.get("verified"))

        domain_output: list[dict[str, Any]] = []
        for tier in selected_tiers:
            cell_name = f"{domain}_T{tier}"
            cell_candidates = [
                candidate for candidate in verified_candidates if int(candidate.get("tier", 0)) == tier
            ]
            selected = select_final(cell_candidates, target_per_cell=config.queries_per_cell)
            if len(selected) < config.queries_per_cell:
                cells_below_target.append(cell_name)
                logger.warning(
                    "%s below target: selected %s / %s",
                    cell_name,
                    len(selected),
                    config.queries_per_cell,
                )
            domain_output.extend(assign_final_ids(domain, tier, selected))

        domain_path = output_dir / f"{domain.lower()}.jsonl"
        merged_output = _merge_existing_domain_output(domain_path, domain_output, selected_tiers)
        _rewrite_jsonl(domain_path, merged_output)
        final_records.extend(domain_output)

    verification_rate = (
        verification_matches / verification_candidates if verification_candidates else 0.0
    )
    manifest = _build_manifest(
        final_records=final_records,
        domains=selected_domains,
        tiers=selected_tiers,
        config=config,
        candidates_generated=len(raw_candidates),
        candidates_after_validation=len(valid_candidates),
        candidates_after_dedup=len(deduplicated),
        verification_rate=verification_rate,
        cells_below_target=cells_below_target,
        generation_hours=(time.perf_counter() - start) / 3600.0,
    )
    manifest["parse_failures"] = parse_failures
    manifest["parse_failure_rate"] = parse_failures / max(len(raw_candidates), 1)
    manifest["validation_failures"] = validation_failures
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
