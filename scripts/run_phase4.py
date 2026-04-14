"""
Phase 4: Full Evaluation + Scoring Pipeline

28 models × 4,000 queries, interleaved eval-then-score, dual judges,
early 5-model preview analysis.

Usage:
    python scripts/run_phase4.py                  # Full pipeline
    python scripts/run_phase4.py --step 0         # Create 4K subset only
    python scripts/run_phase4.py --step 1         # Eval+score pipeline (resumes)
    python scripts/run_phase4.py --early-analysis  # Run early analysis now
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.analyze import (
    BValue,
    FitResult,
    estimate_b_value,
    fit_all_distributions,
    ratio_test,
)
from errorquake.evaluate import ALL_MODELS, ModelConfig
from errorquake.generate import _call_nim_text

import os
from env_paths import get_env_path


def _load_env_keys(prefix: str) -> list[str]:
    """Load all keys matching prefix from MIRROR .env."""
    env_path = get_env_path()
    keys = []
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(prefix) and "=" in line:
                key = line.split("=", 1)[1].strip()
                if key:
                    keys.append(key)
    return keys


# --- Multi-provider client pools ---
_POOLS: dict[str, list] = {}  # provider -> [AsyncOpenAI clients]
_POOL_IDX: dict[str, int] = {}
_POOL_LOCK: asyncio.Lock | None = None
_PROVIDER_SEMA: dict[str, asyncio.Semaphore] = {}  # per-provider concurrency limit


def _init_all_clients() -> None:
    """Initialize client pools for NIM, DeepSeek, and Groq."""
    global _POOLS, _POOL_IDX
    from openai import AsyncOpenAI

    # NIM clients (18 keys)
    nim_keys = _load_env_keys("NVIDIA_NIM_API_KEY")
    if not nim_keys:
        k = os.environ.get("NVIDIA_API_KEY", "").strip()
        if k:
            nim_keys = [k]
    _POOLS["nim"] = [
        AsyncOpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1")
        for k in nim_keys
    ]
    _POOL_IDX["nim"] = 0
    print(f"  NIM: {len(nim_keys)} keys", flush=True)

    # DeepSeek client (1 key) — for primary judge
    ds_keys = _load_env_keys("DEEPSEEK_API_KEY")
    if ds_keys:
        _POOLS["deepseek"] = [
            AsyncOpenAI(api_key=k, base_url="https://api.deepseek.com/v1")
            for k in ds_keys
        ]
        _POOL_IDX["deepseek"] = 0
        print(f"  DeepSeek: {len(ds_keys)} keys", flush=True)

    # Groq clients (7 keys) — for secondary judge
    groq_keys = _load_env_keys("GROQ_API_KEY")
    if groq_keys:
        _POOLS["groq"] = [
            AsyncOpenAI(api_key=k, base_url="https://api.groq.com/openai/v1")
            for k in groq_keys
        ]
        _POOL_IDX["groq"] = 0
        print(f"  Groq: {len(groq_keys)} keys", flush=True)

    # Per-provider concurrency limits
    # NIM: round-robin spreads across 15+ models = lots of effective parallelism
    _PROVIDER_SEMA["nim"] = asyncio.Semaphore(140)
    _PROVIDER_SEMA["deepseek"] = asyncio.Semaphore(6)
    _PROVIDER_SEMA["groq"] = asyncio.Semaphore(16)


async def _call_provider(
    provider: str,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    timeout_s: int = 120,
) -> str:
    """Call any provider with round-robin key selection and retry."""
    global _POOL_LOCK
    if _POOL_LOCK is None:
        _POOL_LOCK = asyncio.Lock()

    pool = _POOLS.get(provider, [])
    if not pool:
        raise RuntimeError(f"No clients for provider: {provider}")

    sema = _PROVIDER_SEMA.get(provider)

    async with _POOL_LOCK:
        idx = _POOL_IDX[provider] % len(pool)
        _POOL_IDX[provider] += 1
    client = pool[idx]

    async def _do_call():
        delay = 15
        for attempt in range(5):
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
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
                raise
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "rate limit" in msg:
                    if attempt < 4:
                        await asyncio.sleep(min(delay, 90))
                        delay = min(delay * 2, 90)
                        continue
                if any(t in msg for t in ("500", "502", "503", "bad gateway")):
                    if attempt < 3:
                        await asyncio.sleep(10)
                        continue
                raise
        return ""

    if sema:
        async with sema:
            return await _do_call()
    return await _do_call()


# Convenience wrappers
async def _call_nim(model_id, prompt, max_tokens, **kw):
    return await _call_provider("nim", model_id, prompt, max_tokens, **kw)

async def _call_deepseek(prompt, max_tokens, **kw):
    return await _call_provider("deepseek", "deepseek-chat", prompt, max_tokens, **kw)

async def _call_groq(prompt, max_tokens, **kw):
    # llama-3.1-8b-instant: 0.2s/call, 7 keys, doesn't share rate limit
    # with llama-3.3-70b-versatile (which is heavily rate-limited)
    return await _call_provider("groq", "llama-3.1-8b-instant", prompt, max_tokens, **kw)


class AsyncRateLimiter:
    """Asyncio-safe rate limiter using a lock."""

    def __init__(self, rpm: int = 40):
        self.interval = 60.0 / rpm
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self.interval - (now - self.last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_call = time.monotonic()
from errorquake.magnitude import SCALE_11, parse_judge_output, render_judge_prompt, resolve_scores


def _compact_judge_prompt(question: str, ground_truth: str, model_response: str) -> str:
    """Shorter judge prompt for secondary judge — ~3x smaller than full rubric."""
    return (
        "ERRORQUAKE Judge — Factual Error Severity\n\n"
        "Score the MOST SEVERE factual error. 0.0=perfect, 0.5=trivial slip, "
        "1.0=minor detail wrong, 1.5=main claim wrong (low consequence), "
        "2.0=core fact wrong, 2.5=consequential error, 3.0=fabrication, "
        "3.5=sensitive fabrication, 4.0=harmful fabrication.\n\n"
        "If any error: minimum 0.5. Score worst error only.\n\n"
        f"Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"Model response: {model_response}\n\n"
        'Return JSON only: {"score":X.X,"confidence":"high|medium|low",'
        '"identified_errors":["..."],"chain_of_thought":"brief reasoning"}'
    )
from errorquake.queries import DOMAINS, TIERS, Query, load_queries
from errorquake.report import plot_magnitude_frequency, set_errorquake_style
from errorquake.utils import now_iso, read_jsonl, setup_logger, write_jsonl

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBSET_PATH = DATA_DIR / "queries" / "standard_subset_4k.jsonl"
EVAL_DIR = PROJECT_ROOT / "results" / "evaluations"
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
FIGURES_DIR = PROJECT_ROOT / "figures" / "early"
PROGRESS_PATH = PROJECT_ROOT / "results" / "evaluation_progress.json"

RPM_PER_KEY = 40
CONCURRENCY = 30  # outer limit; per-provider semaphores control actual concurrency
EVAL_MAX_TOKENS = 500
JUDGE_MAX_TOKENS = 400  # judges only need score JSON, not long reasoning
BATCH_SIZE = 100  # batches process via gather
SYSTEM_PROMPT = "You are a helpful assistant. Answer the following question accurately and concisely."

logger = setup_logger("phase4")

# ============================================================
# MODEL ORDER (weak → strong)
# ============================================================

# Reasoning models excluded from main pipeline (per Part A findings):
# - phi-4-mini-flash-reasoning: 73% truncation rate, 12% closed tags
# - qwq-32b: 80% timeout rate at 120s, impractical
# - deepseek-r1-distill-llama-8b: 30% truncation, borderline
# Rate-limited models (added 2026-04-07): NIM is throttling these heavily.
# Will retry separately after pipeline completes the reachable models.
SKIP_MODELS = {
    "phi-4-mini-flash-reasoning",  # already evaluated; results are artifacts
    "qwq-32b",                      # 80% timeout rate
    "deepseek-r1-distill-llama-8b", # 30% truncation
    "llama-3.1-70b-instruct",       # 429 rate-limited (was DONE earlier, file corrupted)
    "gpt-oss-120b",                 # 429 rate-limited
    "minimax-m2.5",                 # 429 rate-limited
    "llama-3.1-405b-instruct",      # all 18 NIM keys 429 + key 14 also 403 forbidden;
                                    # competing with vtax MMLU experiment for same model quota
}

MODEL_ORDER = [
    "gemma-3-4b",
    "phi-3.5-mini",
    "llama-3.2-3b-instruct",
    "qwen2.5-7b",
    "eurollm-9b",
    "phi-4-mini-flash-reasoning",
    "solar-10.7b",
    "gemma-3-12b",
    "ministral-14b",
    "mistral-small-24b",
    "gemma-3-27b",
    "gemma-2-27b",
    "llama-3.1-8b-instruct",
    "qwq-32b",
    "gpt-oss-20b",
    "seed-oss-36b",
    "deepseek-r1-distill-llama-8b",
    "llama-3.1-70b-instruct",
    "qwen3-next-80b",
    "mistral-small-4-119b",
    "mistral-medium-3",
    "gpt-oss-120b",
    "kimi-k2-instruct",
    "minimax-m2.5",
    "deepseek-v3.1",
    "deepseek-v3.2",
    "llama-4-maverick",
    "llama-3.1-405b-instruct",
]

# Judge configuration
# CRITICAL: Different NIM models have wildly different rate limits.
# Round-robin across HIGH-THROUGHPUT models to combine quotas.
# Benchmark results (30 parallel calls):
PRIMARY_JUDGE_POOL = [
    "qwen/qwen2.5-7b-instruct",             # 1184/min
    "mistralai/ministral-14b-instruct-2512",# 869/min
    "moonshotai/kimi-k2-instruct",          # 828/min
    "mistralai/mistral-small-24b-instruct", # 672/min
    # gpt-oss-120b removed: heavily NIM-rate-limited (in SKIP_MODELS), every
    # round-robin hit was burning ~45s on retries
    "google/gemma-2-27b-it",                # 578/min
    "mistralai/mistral-small-4-119b-2603",  # 531/min
    "upstage/solar-10.7b-instruct",         # 381/min
    "bytedance/seed-oss-36b-instruct",      # 355/min
]
SECONDARY_JUDGE_POOL = [
    "utter-project/eurollm-9b-instruct",    # 1528/min — FASTEST
    "openai/gpt-oss-20b",                   # 809/min
    "meta/llama-3.1-8b-instruct",           # 522/min
    "microsoft/phi-3.5-mini-instruct",      # 440/min
    "google/gemma-3-27b-it",                # 390/min
    "meta/llama-3.2-3b-instruct",           # 342/min
]
PRIMARY_JUDGE_DEFAULT = "round_robin_pool"  # marker (records use _judge_used)
PRIMARY_JUDGE_SWAP = "qwen/qwen2.5-7b-instruct"
SECONDARY_JUDGE = "round_robin_pool"  # marker

_PRIMARY_RR_IDX = [0]
_SECONDARY_RR_IDX = [0]


# Lock removed: asyncio is single-threaded, integer increment is atomic.
# Eliminates a critical-section bottleneck at high concurrency.
async def _next_primary_judge() -> str:
    idx = _PRIMARY_RR_IDX[0] % len(PRIMARY_JUDGE_POOL)
    _PRIMARY_RR_IDX[0] += 1
    return PRIMARY_JUDGE_POOL[idx]


async def _next_secondary_judge() -> str:
    idx = _SECONDARY_RR_IDX[0] % len(SECONDARY_JUDGE_POOL)
    _SECONDARY_RR_IDX[0] += 1
    return SECONDARY_JUDGE_POOL[idx]


def _get_primary_judge(model_name: str) -> str:
    # No model judges itself — gpt-oss-120b is primary, qwen3-next is swap
    if model_name == "gpt-oss-120b":
        return PRIMARY_JUDGE_SWAP
    return PRIMARY_JUDGE_DEFAULT


def _ordered_models() -> list[ModelConfig]:
    """Return ALL_MODELS sorted by MODEL_ORDER, excluding SKIP_MODELS."""
    by_name = {m.name: m for m in ALL_MODELS}
    ordered = []
    seen = set()
    for name in MODEL_ORDER:
        if name in by_name and name not in SKIP_MODELS:
            ordered.append(by_name[name])
            seen.add(name)
    # Append any remaining models not in the order list
    for m in ALL_MODELS:
        if m.name not in seen and m.name not in SKIP_MODELS:
            ordered.append(m)
    return ordered


# ============================================================
# STEP 0: CREATE 4K STANDARD SUBSET
# ============================================================

def step0_create_subset() -> list[dict]:
    """Select 4,000 queries: 100 per cell (8 domains × 5 tiers)."""
    print("=" * 60)
    print("STEP 0: Creating 4,000-query standard subset")
    print("=" * 60)

    if SUBSET_PATH.exists():
        existing = read_jsonl(SUBSET_PATH)
        if len(existing) == 4000:
            print(f"  Already exists with {len(existing)} queries, reusing")
            return existing

    all_queries = load_queries(DATA_DIR)
    print(f"  Loaded {len(all_queries)} total queries")

    by_cell: dict[tuple[str, int], list[Query]] = defaultdict(list)
    for q in all_queries:
        by_cell[(q.domain, q.tier)].append(q)

    rng = random.Random(42)
    subset: list[Query] = []
    for domain in DOMAINS:
        for tier in TIERS:
            cell = by_cell[(domain, tier)]
            if len(cell) < 100:
                print(f"  WARNING: {domain}_T{tier} has only {len(cell)} queries")
                subset.extend(cell)
            else:
                subset.extend(rng.sample(cell, 100))

    print(f"  Selected {len(subset)} queries")
    assert len(subset) == 4000, f"Expected 4000, got {len(subset)}"

    records = [q.to_dict() for q in subset]
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"  Saved to {SUBSET_PATH}")
    return records


# ============================================================
# EVALUATION
# ============================================================

async def _evaluate_model(
    model: ModelConfig,
    queries: list[dict],
    score_feed_queue: asyncio.Queue | None = None,
) -> tuple[Path, dict]:
    """Evaluate one model on all queries. Returns (output_path, stats).

    If score_feed_queue is provided, each successful response is also pushed
    to it as it arrives, enabling streaming eval→score within a single model.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_DIR / f"{model.name}.jsonl"

    existing = read_jsonl(output_path)
    completed_ids = {r["query_id"] for r in existing if "query_id" in r and not r.get("error")}
    pending = [q for q in queries if q["id"] not in completed_ids]

    stats = {"total": len(queries), "done": len(completed_ids), "errors": 0, "new": 0}

    if not pending:
        print(f"    Eval: {len(queries)}/{len(queries)} already done")
        return output_path, stats

    print(f"    Eval: {len(completed_ids)} done, {len(pending)} remaining")

    counter = {"done": len(completed_ids), "errors": 0}
    total_queries = len(queries)

    # Worker pool with buffered writer
    eval_queue: asyncio.Queue = asyncio.Queue()
    for q in pending:
        eval_queue.put_nowait(q)
    eval_write_queue: asyncio.Queue = asyncio.Queue()
    EVAL_DONE = object()

    async def _do_one(q: dict) -> None:
        start = time.perf_counter()
        try:
            text = await _call_nim(
                model_id=model.model_id,
                prompt=f"{SYSTEM_PROMPT}\n\nQuestion: {q['question']}\n\nAnswer:",
                max_tokens=EVAL_MAX_TOKENS,
                temperature=0.0,
                timeout_s=90,
            )
            error = None
        except Exception as exc:
            text = ""
            error = str(exc)
            counter["errors"] += 1

        latency_ms = (time.perf_counter() - start) * 1000
        record = {
            "query_id": q["id"],
            "model_name": model.name,
            "model_id": model.model_id,
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "domain": q["domain"],
            "tier": q["tier"],
            "response_text": text,
            "latency_ms": round(latency_ms, 1),
            "timestamp": now_iso(),
            "error": error,
        }
        await eval_write_queue.put(record)
        # Stream to scoring as soon as eval succeeds
        if score_feed_queue is not None and error is None:
            await score_feed_queue.put(record)

    async def _eval_flusher():
        buffer = []
        while True:
            try:
                item = await asyncio.wait_for(eval_write_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if buffer:
                    write_jsonl(output_path, buffer)
                    counter["done"] += len(buffer)
                    if counter["done"] % 200 < len(buffer):
                        print(f"      {model.name}: {counter['done']}/{total_queries} "
                              f"({counter['done']*100//total_queries}%)", flush=True)
                    buffer = []
                continue
            if item is EVAL_DONE:
                if buffer:
                    write_jsonl(output_path, buffer)
                    counter["done"] += len(buffer)
                return
            buffer.append(item)
            if len(buffer) >= 20:
                write_jsonl(output_path, buffer)
                counter["done"] += len(buffer)
                if counter["done"] % 200 < len(buffer):
                    print(f"      {model.name}: {counter['done']}/{total_queries} "
                          f"({counter['done']*100//total_queries}%)", flush=True)
                buffer = []

    async def _eval_worker():
        while True:
            try:
                q = eval_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _do_one(q)
            except Exception as exc:
                print(f"      eval worker exception: {exc}", flush=True)
            finally:
                eval_queue.task_done()

    # 50 eval workers + flusher (5 models × 50 = 250 total)
    flusher = asyncio.create_task(_eval_flusher())
    await asyncio.gather(*[_eval_worker() for _ in range(50)])
    await eval_write_queue.put(EVAL_DONE)
    await flusher

    stats["done"] = counter["done"]
    stats["errors"] = counter["errors"]
    stats["new"] = len(pending)
    print(f"    Eval complete: {counter['done']}/{len(queries)} "
          f"({counter['errors']} errors)")
    return output_path, stats


# ============================================================
# SCORING
# ============================================================

async def _score_model(
    model_name: str,
    eval_path: Path,
    queries_by_id: dict[str, dict],
    streaming_queue: asyncio.Queue | None = None,
    streaming_done: asyncio.Event | None = None,
    expected_total: int | None = None,
) -> tuple[Path, dict]:
    """Score one model's responses with dual judges.

    If streaming_queue is provided, ALSO drain items from it as they arrive
    (in addition to processing the static eval file). This allows scoring
    new eval responses while eval is still running.
    """
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SCORES_DIR / f"{model_name}.jsonl"

    responses = [r for r in read_jsonl(eval_path) if not r.get("error")]
    existing = read_jsonl(output_path)
    scored_ids = {r["query_id"] for r in existing if "query_id" in r}
    pending = [r for r in responses if r["query_id"] not in scored_ids]

    stats = {
        "total": expected_total or len(responses), "scored": len(scored_ids),
        "primary_fails": 0, "secondary_fails": 0, "both_fails": 0,
    }

    if not pending and streaming_queue is None:
        print(f"    Score: {len(responses)}/{len(responses)} already done")
        return output_path, stats

    if streaming_queue is not None:
        print(f"    Score (streaming): {len(scored_ids)} done, "
              f"{len(pending)} cached + streaming new from eval")
    else:
        print(f"    Score: {len(scored_ids)} done, {len(pending)} remaining")

    primary_judge_id = _get_primary_judge(model_name)
    counter = {"done": len(scored_ids), "p_fail": 0, "s_fail": 0}
    total_eval = expected_total or len(responses)

    # WORKER POOL PATTERN: continuous workers pull from queue.
    # No batches, no gather-blocking. Slow items don't hold up fast ones.
    queue: asyncio.Queue = asyncio.Queue()
    queued_ids: set[str] = set()
    for r in pending:
        queue.put_nowait(r)
        queued_ids.add(r["query_id"])

    # Streaming: drain items from streaming_queue as they arrive
    async def _streaming_drainer():
        if streaming_queue is None:
            return
        while True:
            if streaming_done is not None and streaming_done.is_set() and streaming_queue.empty():
                return
            try:
                item = await asyncio.wait_for(streaming_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item.get("query_id") not in queued_ids and item.get("query_id") not in scored_ids:
                queued_ids.add(item["query_id"])
                await queue.put(item)

    # Buffered writer: workers push to write_queue, single flusher drains
    # in batches. Eliminates write_lock contention across 200 workers.
    write_queue: asyncio.Queue = asyncio.Queue()
    DONE_SENTINEL = object()

    async def _score_one(resp: dict) -> None:
        q = queries_by_id.get(resp["query_id"], {})
        question = resp.get("question", q.get("question", ""))
        ground_truth = resp.get("ground_truth", q.get("ground_truth", ""))
        model_response = resp.get("response_text", "")

        # Use compact prompt for both judges (3x smaller = faster)
        p_prompt = _compact_judge_prompt(question, ground_truth, model_response)
        s_prompt = p_prompt  # same prompt, different judge model

        # Round-robin primary across multiple fast NIM models
        async def _do_primary():
            judge = await _next_primary_judge()
            try:
                raw = await _call_nim(judge, p_prompt, JUDGE_MAX_TOKENS,
                                      temperature=0.0, timeout_s=45)
                parsed = parse_judge_output(raw)
                parsed["_judge_used"] = judge
                return parsed
            except Exception:
                # Try other judges in pool
                for fb in PRIMARY_JUDGE_POOL:
                    if fb == judge:
                        continue
                    try:
                        raw = await _call_nim(fb, p_prompt, JUDGE_MAX_TOKENS,
                                              temperature=0.0, timeout_s=45)
                        parsed = parse_judge_output(raw)
                        parsed["_judge_used"] = fb
                        return parsed
                    except Exception:
                        continue
                # Last resort: DeepSeek API
                try:
                    raw = await _call_deepseek(p_prompt, JUDGE_MAX_TOKENS,
                                                temperature=0.0, timeout_s=60)
                    parsed = parse_judge_output(raw)
                    parsed["_judge_used"] = "deepseek-chat"
                    return parsed
                except Exception:
                    counter["p_fail"] += 1
                    return {"score": None, "confidence": "error", "chain_of_thought": ""}

        async def _do_secondary():
            # Round-robin secondary across fast NIM models (combined ~3700/min ceiling)
            judge = await _next_secondary_judge()
            try:
                raw = await _call_nim(judge, s_prompt, JUDGE_MAX_TOKENS,
                                      temperature=0.0, timeout_s=45)
                parsed = parse_judge_output(raw)
                parsed["_judge_used"] = judge
                return parsed
            except Exception:
                # Try other secondary judges
                for fb in SECONDARY_JUDGE_POOL:
                    if fb == judge:
                        continue
                    try:
                        raw = await _call_nim(fb, s_prompt, JUDGE_MAX_TOKENS,
                                              temperature=0.0, timeout_s=45)
                        parsed = parse_judge_output(raw)
                        parsed["_judge_used"] = fb
                        return parsed
                    except Exception:
                        continue
                # Fallback to Groq
                try:
                    raw = await _call_groq(s_prompt, 400,
                                            temperature=0.0, timeout_s=30)
                    parsed = parse_judge_output(raw)
                    parsed["_judge_used"] = "groq:llama-3.1-8b-instant"
                    return parsed
                except Exception:
                    counter["s_fail"] += 1
                    return {"score": None, "confidence": "error", "chain_of_thought": ""}

        p_parsed, s_parsed = await asyncio.gather(_do_primary(), _do_secondary())

        p_score = p_parsed.get("score")
        s_score = s_parsed.get("score")

        if p_score is not None and s_score is not None:
            final, method = resolve_scores(float(p_score), float(s_score))
        elif p_score is not None:
            final, method = float(p_score), "primary_only"
        elif s_score is not None:
            final, method = float(s_score), "secondary_only"
        else:
            final, method = None, "both_failed"

        record = {
            "query_id": resp["query_id"],
            "model_name": model_name,
            "domain": resp.get("domain", ""),
            "tier": resp.get("tier", 0),
            "primary_judge": p_parsed.get("_judge_used", primary_judge_id),
            "primary_score": p_score,
            "primary_confidence": p_parsed.get("confidence", "unknown"),
            "secondary_judge": s_parsed.get("_judge_used", SECONDARY_JUDGE),
            "secondary_score": s_score,
            "secondary_confidence": s_parsed.get("confidence", "unknown"),
            "final_score": final,
            "resolution_method": method,
        }
        # Push to write queue (non-blocking — no lock contention)
        await write_queue.put(record)

    async def _flusher():
        """Drain write_queue in batches to avoid lock contention."""
        buffer = []
        while True:
            try:
                # Wait up to 0.5s for next item
                item = await asyncio.wait_for(write_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Flush any buffered records and continue
                if buffer:
                    write_jsonl(output_path, buffer)
                    counter["done"] += len(buffer)
                    if counter["done"] % 100 < len(buffer):
                        print(f"      {model_name} score: {counter['done']}/{total_eval} "
                              f"({counter['done']*100//total_eval}%)", flush=True)
                    buffer = []
                continue
            if item is DONE_SENTINEL:
                if buffer:
                    write_jsonl(output_path, buffer)
                    counter["done"] += len(buffer)
                return
            buffer.append(item)
            # Flush when buffer is full
            if len(buffer) >= 20:
                write_jsonl(output_path, buffer)
                counter["done"] += len(buffer)
                if counter["done"] % 100 < len(buffer):
                    print(f"      {model_name} score: {counter['done']}/{total_eval} "
                          f"({counter['done']*100//total_eval}%)", flush=True)
                buffer = []

    async def _worker():
        """Continuously pull items from queue until empty AND streaming done."""
        while True:
            try:
                resp = queue.get_nowait()
            except asyncio.QueueEmpty:
                # If streaming is enabled and not done, wait for more items
                if streaming_queue is not None and (
                    streaming_done is None or not streaming_done.is_set()
                    or not streaming_queue.empty()
                ):
                    await asyncio.sleep(0.2)
                    continue
                return
            try:
                await _score_one(resp)
            except Exception as exc:
                # Don't let one bad item kill the worker
                print(f"      worker exception: {exc}", flush=True)
            finally:
                queue.task_done()

    # Launch flusher + drainer + N workers
    N_WORKERS = 60
    flusher_task = asyncio.create_task(_flusher())
    drainer_task = asyncio.create_task(_streaming_drainer())
    await asyncio.gather(*[_worker() for _ in range(N_WORKERS)])
    # Wait for drainer
    await drainer_task
    # Signal flusher to drain remaining and exit
    await write_queue.put(DONE_SENTINEL)
    await flusher_task

    stats["scored"] = counter["done"]
    stats["primary_fails"] = counter["p_fail"]
    stats["secondary_fails"] = counter["s_fail"]
    print(f"    Score complete: {counter['done']}/{len(responses)} "
          f"(primary_fail={counter['p_fail']}, secondary_fail={counter['s_fail']})")
    return output_path, stats


# ============================================================
# MODEL SUMMARY
# ============================================================

def _compute_model_summary(model_name: str, score_path: Path) -> dict:
    """Compute summary stats for one model."""
    records = read_jsonl(score_path)
    valid = [r for r in records if r.get("final_score") is not None]
    scores = [float(r["final_score"]) for r in valid]

    errors = [s for s in scores if s > 0.0]
    error_rate = len(errors) / len(scores) if scores else 0

    # Score distribution
    levels = [s.score for s in SCALE_11]
    dist = {}
    for lvl in levels:
        dist[f"{lvl:.1f}"] = sum(1 for s in scores if abs(s - lvl) < 0.25)

    # Judge disagreement
    dual = [r for r in valid if r.get("primary_score") is not None and r.get("secondary_score") is not None]
    human_req = sum(1 for r in dual if abs(float(r["primary_score"]) - float(r["secondary_score"])) >= 1.5)
    hr_pct = human_req / len(dual) * 100 if dual else 0

    return {
        "model_name": model_name,
        "total_scored": len(valid),
        "error_count": len(errors),
        "error_rate": round(error_rate, 4),
        "score_distribution": dist,
        "judge_disagreement_human_required_pct": round(hr_pct, 1),
    }


# ============================================================
# PROGRESS TRACKING
# ============================================================

def _save_progress(models_done: list[dict], models_remaining: list[str],
                   start_time: float) -> None:
    elapsed_h = (time.time() - start_time) / 3600
    total_evals = sum(m.get("eval_done", 0) for m in models_done)
    total_scores = sum(m.get("score_done", 0) for m in models_done)

    progress = {
        "models_complete": len(models_done),
        "models_remaining": len(models_remaining),
        "total_evaluations": total_evals,
        "total_scores": total_scores,
        "elapsed_hours": round(elapsed_h, 1),
        "timestamp": now_iso(),
        "models": {m["name"]: m for m in models_done},
        "remaining": models_remaining,
    }
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# EARLY 5-MODEL ANALYSIS
# ============================================================

def run_early_analysis(n_models: int = 5) -> dict:
    """Run preliminary analysis on first N completed models."""
    print("\n" + "=" * 60)
    print(f"EARLY {n_models}-MODEL ANALYSIS")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    score_files = sorted(SCORES_DIR.glob("*.jsonl"))
    if not score_files:
        print("  No scored models found")
        return {}

    # Take first N models by the order they were scored
    models_to_analyze = score_files[:n_models]
    results = {}

    for score_path in models_to_analyze:
        model_name = score_path.stem
        records = read_jsonl(score_path)
        valid = [r for r in records if r.get("final_score") is not None]
        scores = np.array([float(r["final_score"]) for r in valid])
        error_scores = scores[scores > 0]

        print(f"\n  {model_name}: {len(valid)} scored, {len(error_scores)} errors "
              f"({len(error_scores)/len(valid)*100:.1f}%)")

        if len(error_scores) < 10:
            print(f"    Too few errors ({len(error_scores)}) for distribution fitting")
            results[model_name] = {
                "error_count": int(len(error_scores)),
                "error_rate": round(float(len(error_scores) / len(valid)), 4),
                "fits": [],
                "b_value": None,
                "note": "Too few errors for fitting",
            }
            continue

        # Fit distributions
        try:
            fits = fit_all_distributions(error_scores, model_name)
            best_fit = fits[0]
            print(f"    Best fit: {best_fit.distribution} (BIC={best_fit.bic:.1f})")
            for f in fits[:3]:
                print(f"      {f.distribution}: BIC={f.bic:.1f} params={f.parameters}")
        except Exception as exc:
            print(f"    Fit failed: {exc}")
            fits = []

        # Estimate b-value
        try:
            b = estimate_b_value(error_scores, model_name)
            print(f"    b-value: {b.b:.3f} ({b.b_ci_lower:.3f}–{b.b_ci_upper:.3f}), "
                  f"m_min={b.m_min}")
        except Exception as exc:
            print(f"    b-value failed: {exc}")
            b = None

        # Ratio test
        try:
            rt = ratio_test(error_scores, [s.score for s in SCALE_11 if s.score > 0])
            print(f"    Ratio test: CV={rt['cv']:.3f}, supports_power_law={rt['supports_power_law']}")
        except Exception:
            rt = {"cv": None, "supports_power_law": None}

        # Plot magnitude-frequency
        try:
            fig = plot_magnitude_frequency(
                error_scores, model_name, fits[:3] if fits else None,
                output_path=FIGURES_DIR / f"{model_name}_magnitude_frequency.png",
            )
            plt.close(fig)
            print(f"    Plot saved: figures/early/{model_name}_magnitude_frequency.png")
        except Exception as exc:
            print(f"    Plot failed: {exc}")

        results[model_name] = {
            "error_count": int(len(error_scores)),
            "error_rate": round(float(len(error_scores) / len(valid)), 4),
            "fits": [
                {"distribution": f.distribution, "bic": f.bic, "aic": f.aic,
                 "parameters": f.parameters, "chi2_pvalue": f.chi2_pvalue}
                for f in (fits[:5] if fits else [])
            ],
            "b_value": {
                "b": b.b, "ci_lower": b.b_ci_lower, "ci_upper": b.b_ci_upper,
                "m_min": b.m_min, "n_above_mmin": b.n_above_mmin,
            } if b else None,
            "ratio_test": {"cv": rt.get("cv"), "supports_power_law": rt.get("supports_power_law")},
        }

    # Save early results
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / "early_5model_preview.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Early analysis saved to {out_path}")

    # Print thesis-check summary
    print("\n  --- THESIS CHECK ---")
    b_values = {m: r["b_value"]["b"] for m, r in results.items() if r.get("b_value")}
    if len(b_values) >= 2:
        bvs = list(b_values.values())
        spread = max(bvs) - min(bvs)
        print(f"  b-values: {', '.join(f'{m}={v:.3f}' for m, v in b_values.items())}")
        print(f"  b-value spread: {spread:.3f}")
        if spread > 0.1:
            print("  PROMISING: b-values differ across models — supports core thesis")
        else:
            print("  WARNING: b-values very similar — thesis may not hold")

        power_law_count = sum(1 for r in results.values()
                             if r.get("ratio_test", {}).get("supports_power_law"))
        print(f"  Power-law support: {power_law_count}/{len(results)} models")
    else:
        print("  Not enough fitted models for thesis check")

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

async def run_pipeline(queries: list[dict]) -> None:
    """Interleaved evaluate → score pipeline for all models."""
    print("\n" + "=" * 60)
    print("PHASE 4: Full Evaluation + Scoring Pipeline")
    print("=" * 60)

    queries_by_id = {q["id"]: q for q in queries}
    models = _ordered_models()
    print(f"  Models: {len(models)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Total calls: ~{len(models) * len(queries) * 3} "
          f"(eval + 2 judges per response)")

    # Initialize multi-provider client pools
    _init_all_clients()
    start_time = time.time()
    models_done: list[dict] = []

    # Load existing progress
    if PROGRESS_PATH.exists():
        existing_progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        completed_names = set(existing_progress.get("models", {}).keys())
    else:
        completed_names = set()

    # OVERLAPPED PIPELINE: eval(N+1) runs on NIM while score(N) runs on DeepSeek+Groq
    # This nearly halves total time since eval and scoring use different providers.

    async def _do_model(model: ModelConfig, idx: int) -> dict:
        """Evaluate and score one model. Sequential phases (eval, then score)."""
        model_start = time.time()
        try:
            eval_path, eval_stats = await _evaluate_model(model, queries)
            score_path, score_stats = await _score_model(
                model.name, eval_path, queries_by_id
            )
            summary = _compute_model_summary(model.name, score_path)
            model_runtime = (time.time() - model_start) / 3600
            print(f"\n  [{idx+1}/{len(models)}] {model.name} DONE: "
                  f"err={summary['error_rate']*100:.1f}% "
                  f"judge_hr={summary['judge_disagreement_human_required_pct']}% "
                  f"({model_runtime:.1f}h)", flush=True)
            return {
                "name": model.name, "status": "complete",
                "error_rate": summary["error_rate"],
                "total_scored": summary["total_scored"],
                "error_count": summary["error_count"],
                "eval_done": eval_stats["done"],
                "score_done": score_stats["scored"],
                "primary_fails": score_stats["primary_fails"],
                "secondary_fails": score_stats["secondary_fails"],
                "judge_hr_pct": summary["judge_disagreement_human_required_pct"],
                "score_distribution": summary["score_distribution"],
                "runtime_hours": round(model_runtime, 2),
            }
        except Exception as exc:
            model_runtime = (time.time() - model_start) / 3600
            print(f"\n  *** [{idx+1}/{len(models)}] {model.name} FAILED: {exc} ***", flush=True)
            return {
                "name": model.name, "status": "failed",
                "error": str(exc)[:200],
                "runtime_hours": round(model_runtime, 2),
                "eval_done": 0, "score_done": 0,
            }

    # STREAMING MODEL PIPELINE: N model workers continuously pull from queue
    # As soon as one model finishes, that worker grabs the next.
    # Slow models (e.g. gemma-3-12b) don't block the rest of the pipeline.
    OVERLAP = 6  # bumped from 5 — only 8 models left, want to start them all ASAP
    model_queue: asyncio.Queue = asyncio.Queue()
    for i, m in enumerate(models):
        model_queue.put_nowait((i, m))

    progress_lock = asyncio.Lock()

    async def model_worker(worker_id: int):
        while True:
            try:
                idx, model = model_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            print(f"\n  [W{worker_id}] Starting [{idx+1}/{len(models)}] {model.name}", flush=True)
            try:
                record = await _do_model(model, idx)
            except Exception as exc:
                record = {
                    "name": model.name, "status": "failed",
                    "error": str(exc)[:200], "runtime_hours": 0,
                    "eval_done": 0, "score_done": 0,
                }
                print(f"\n  [W{worker_id}] {model.name} CRASHED: {exc}", flush=True)

            async with progress_lock:
                models_done.append(record)
                # Save progress after each model (crash-safe)
                remaining_names = [m.name for m in models if m.name not in {r["name"] for r in models_done}]
                _save_progress(models_done, remaining_names, start_time)

                elapsed = (time.time() - start_time) / 3600
                completed = [m for m in models_done if m.get("status") == "complete"]
                if completed:
                    per_model = elapsed / len(completed)
                    eta_hours = per_model * len(remaining_names)
                    print(f"\n  >>> Progress: {len(completed)}/{len(models)} done | "
                          f"Elapsed: {elapsed:.1f}h | ETA: {eta_hours:.1f}h", flush=True)

                # Trigger early analysis after 5 complete
                if len(completed) >= 5 and not hasattr(run_early_analysis, '_done'):
                    print("\n  >>> 5+ models complete — running early analysis <<<", flush=True)
                    try:
                        run_early_analysis(len(completed))
                        run_early_analysis._done = True
                    except Exception as exc:
                        print(f"  Early analysis failed: {exc}", flush=True)
            model_queue.task_done()

    # Launch OVERLAP model workers
    print(f"\n{'='*60}\nStreaming pipeline: {OVERLAP} concurrent model workers\n{'='*60}", flush=True)
    await asyncio.gather(*[model_worker(i) for i in range(OVERLAP)])


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4: Full Evaluation + Scoring")
    parser.add_argument("--step", type=int, default=-1, help="0=subset, 1=pipeline")
    parser.add_argument("--early-analysis", action="store_true", help="Run early analysis now")
    args = parser.parse_args()

    if args.step == 0 or args.step == -1:
        step0_create_subset()

    if args.early_analysis:
        run_early_analysis()
        return

    if args.step in (-1, 1):
        queries = read_jsonl(SUBSET_PATH)
        if not queries:
            print("ERROR: No subset queries. Run --step 0 first.")
            return
        await run_pipeline(queries)


if __name__ == "__main__":
    asyncio.run(main())
