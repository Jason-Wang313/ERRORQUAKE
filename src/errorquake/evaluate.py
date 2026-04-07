"""Async NIM-backed LLM evaluation client with JSONL checkpointing."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from tqdm.auto import tqdm

from errorquake.queries import Query
from errorquake.utils import ProjectConfig, get_completed_ids, now_iso, setup_logger, write_jsonl


@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str
    api_key_env: str
    max_tokens: int = 500
    temperature: float = 0.0
    rate_limit_rpm: int = 60
    system_prompt: str = (
        "You are a helpful assistant. Answer the following question accurately and concisely."
    )

    def __post_init__(self) -> None:
        required = {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "api_key_env": self.api_key_env,
        }
        missing = [field for field, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Missing required model config fields: {', '.join(missing)}")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.rate_limit_rpm <= 0:
            raise ValueError("rate_limit_rpm must be > 0")


@dataclass
class ModelResponse:
    query_id: str
    model_name: str
    model_id: str
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    timestamp: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelResponse":
        return cls(**payload)


class RateLimiter:
    """Simple sliding-window async rate limiter."""

    def __init__(
        self,
        rpm: int,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.rpm = rpm
        self.clock = clock or time.monotonic
        self.sleep_fn = sleep_fn or asyncio.sleep
        self.events: deque[float] = deque()

    async def acquire(self) -> None:
        while True:
            now = self.clock()
            while self.events and now - self.events[0] >= 60:
                self.events.popleft()
            if len(self.events) < self.rpm:
                self.events.append(now)
                return
            wait_for = 60 - (now - self.events[0])
            await self.sleep_fn(max(wait_for, 0))


def _get_api_key(config: ModelConfig) -> str:
    env_names = [config.api_key_env]
    if config.api_key_env == "NVIDIA_API_KEY":
        env_names.append("NVIDIA_NIM_API_KEY")
    for env_name in env_names:
        api_key = os.environ.get(env_name, "").strip()
        if api_key:
            return api_key
    raise RuntimeError(f"Missing API key: {config.api_key_env}")


def _usage_tuple(usage: Any) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    prompt_tokens = int(
        getattr(usage, "input_tokens", 0)
        or getattr(usage, "prompt_tokens", 0)
        or usage.get("input_tokens", 0)
        or usage.get("prompt_tokens", 0)
    )
    completion_tokens = int(
        getattr(usage, "output_tokens", 0)
        or getattr(usage, "completion_tokens", 0)
        or usage.get("output_tokens", 0)
        or usage.get("completion_tokens", 0)
    )
    return prompt_tokens, completion_tokens


def _build_nim_messages(question: str, system_prompt: str) -> list[dict[str, str]]:
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question accurately and concisely.\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            ),
        }
    ]


async def call_nim(config: ModelConfig, question: str) -> ModelResponse:
    """NVIDIA NIM via OpenAI-compatible endpoint."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=_get_api_key(config),
        base_url="https://integrate.api.nvidia.com/v1",
    )
    messages = _build_nim_messages(question, config.system_prompt)
    start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=config.model_id,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    except Exception as exc:
        if config.system_prompt and "system role not supported" in str(exc).lower():
            response = await client.chat.completions.create(
                model=config.model_id,
                messages=_build_nim_messages(question, ""),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        else:
            raise
    latency_ms = (time.perf_counter() - start) * 1000
    message = response.choices[0].message
    prompt_tokens, completion_tokens = _usage_tuple(getattr(response, "usage", None))
    return ModelResponse(
        query_id="",
        model_name=config.name,
        model_id=config.model_id,
        response_text=message.content or "",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        timestamp=now_iso(),
    )


PROVIDER_ADAPTERS: dict[str, Callable[[ModelConfig, str], Awaitable[ModelResponse]]] = {
    "nim": call_nim,
}


ALL_MODELS: list[ModelConfig] = [
    # Meta / Llama
    ModelConfig("llama-4-maverick", "nim", "meta/llama-4-maverick-17b-128e-instruct", "NVIDIA_API_KEY"),
    ModelConfig("llama-3.1-405b-instruct", "nim", "meta/llama-3.1-405b-instruct", "NVIDIA_API_KEY"),
    ModelConfig("llama-3.1-70b-instruct", "nim", "meta/llama-3.1-70b-instruct", "NVIDIA_API_KEY"),
    ModelConfig("llama-3.1-8b-instruct", "nim", "meta/llama-3.1-8b-instruct", "NVIDIA_API_KEY"),
    ModelConfig("llama-3.2-3b-instruct", "nim", "meta/llama-3.2-3b-instruct", "NVIDIA_API_KEY"),

    # DeepSeek
    ModelConfig("deepseek-v3.2", "nim", "deepseek-ai/deepseek-v3.2", "NVIDIA_API_KEY"),
    ModelConfig("deepseek-v3.1", "nim", "deepseek-ai/deepseek-v3.1", "NVIDIA_API_KEY"),
    ModelConfig("deepseek-r1-distill-llama-8b", "nim", "deepseek-ai/deepseek-r1-distill-llama-8b", "NVIDIA_API_KEY"),

    # Mistral
    ModelConfig("mistral-small-4-119b", "nim", "mistralai/mistral-small-4-119b-2603", "NVIDIA_API_KEY"),
    ModelConfig("mistral-small-24b", "nim", "mistralai/mistral-small-24b-instruct", "NVIDIA_API_KEY"),
    ModelConfig("mistral-medium-3", "nim", "mistralai/mistral-medium-3-instruct", "NVIDIA_API_KEY"),
    ModelConfig("ministral-14b", "nim", "mistralai/ministral-14b-instruct-2512", "NVIDIA_API_KEY"),

    # Qwen
    ModelConfig("qwen3-next-80b", "nim", "qwen/qwen3-next-80b-a3b-instruct", "NVIDIA_API_KEY"),
    ModelConfig("qwen2.5-7b", "nim", "qwen/qwen2.5-7b-instruct", "NVIDIA_API_KEY"),
    ModelConfig("qwq-32b", "nim", "qwen/qwq-32b", "NVIDIA_API_KEY"),

    # Gemma
    ModelConfig("gemma-2-27b", "nim", "google/gemma-2-27b-it", "NVIDIA_API_KEY"),
    ModelConfig("gemma-3-27b", "nim", "google/gemma-3-27b-it", "NVIDIA_API_KEY"),
    ModelConfig("gemma-3-12b", "nim", "google/gemma-3-12b-it", "NVIDIA_API_KEY"),
    ModelConfig("gemma-3-4b", "nim", "google/gemma-3-4b-it", "NVIDIA_API_KEY"),

    # Phi
    ModelConfig("phi-4-mini-flash-reasoning", "nim", "microsoft/phi-4-mini-flash-reasoning", "NVIDIA_API_KEY"),
    ModelConfig("phi-3.5-mini", "nim", "microsoft/phi-3.5-mini-instruct", "NVIDIA_API_KEY"),

    # Other text-generation models
    ModelConfig("kimi-k2-instruct", "nim", "moonshotai/kimi-k2-instruct", "NVIDIA_API_KEY"),
    ModelConfig("minimax-m2.5", "nim", "minimaxai/minimax-m2.5", "NVIDIA_API_KEY"),
    ModelConfig("gpt-oss-20b", "nim", "openai/gpt-oss-20b", "NVIDIA_API_KEY"),
    ModelConfig("gpt-oss-120b", "nim", "openai/gpt-oss-120b", "NVIDIA_API_KEY"),
    ModelConfig("solar-10.7b", "nim", "upstage/solar-10.7b-instruct", "NVIDIA_API_KEY"),
    ModelConfig("seed-oss-36b", "nim", "bytedance/seed-oss-36b-instruct", "NVIDIA_API_KEY"),
    ModelConfig("eurollm-9b", "nim", "utter-project/eurollm-9b-instruct", "NVIDIA_API_KEY"),
]


def get_model_catalog(models: list[ModelConfig] | None = None) -> list[ModelConfig]:
    return list(models or ALL_MODELS)


async def _probe_model_reachability(model: ModelConfig) -> tuple[bool, str, str]:
    probe_model = ModelConfig(
        **{**asdict(model), "max_tokens": min(model.max_tokens, 32), "system_prompt": ""}
    )
    last_status = "error"
    last_detail = "Unknown error."
    for attempt in range(3):
        try:
            await asyncio.wait_for(
                PROVIDER_ADAPTERS[model.provider](probe_model, "Reply with the word OK."),
                timeout=60,
            )
            return True, "reachable", model.provider
        except asyncio.TimeoutError:
            last_status = "timeout"
            last_detail = "Probe exceeded 60 seconds."
        except Exception as exc:  # pragma: no cover
            message = str(exc)
            lower = message.lower()
            if "credit balance" in lower or "billing" in lower or "plans & billing" in lower:
                return False, "billing_error", message
            if "404" in lower or "not found" in lower or "does not exist" in lower:
                return False, "model_not_found", message
            if "401" in lower or "403" in lower or "auth" in lower or "api key" in lower:
                return False, "auth_error", message
            if "429" in lower or "rate limit" in lower:
                last_status = "rate_limited"
                last_detail = message
            else:
                return False, "error", message

        if attempt < 2:
            await asyncio.sleep(2)
    return False, last_status, last_detail


async def verify_model_access(
    models: list[ModelConfig] | None = None,
    attempt_live: bool = False,
) -> dict[str, dict[str, str | bool]]:
    report: dict[str, dict[str, str | bool]] = {}
    for model in get_model_catalog(models):
        try:
            _get_api_key(model)
        except RuntimeError:
            report[model.name] = {"ok": False, "status": "missing_api_key", "detail": model.api_key_env}
            continue
        if not attempt_live:
            report[model.name] = {"ok": True, "status": "configured", "detail": model.api_key_env}
            continue
        ok, status, detail = await _probe_model_reachability(model)
        report[model.name] = {"ok": ok, "status": status, "detail": detail}
    return report


class EvaluationEngine:
    """
    Core evaluation loop with checkpointing, concurrency control, and resume support.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = setup_logger("errorquake.evaluate")

    async def _call_with_retry(
        self,
        model: ModelConfig,
        query: Query,
        adapter: Callable[[ModelConfig, str], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        delay = 30.0
        for attempt in range(5):
            try:
                response = await adapter(model, query.question)
                response.query_id = query.id
                response.model_name = model.name
                response.model_id = model.model_id
                if not response.timestamp:
                    response.timestamp = now_iso()
                return response
            except Exception as exc:
                message = str(exc)
                retryable = "429" in message or "rate limit" in message.lower()
                if retryable and attempt < 4:
                    await asyncio.sleep(min(delay, 300.0))
                    delay = min(delay * 2, 300.0)
                    continue
                return ModelResponse(
                    query_id=query.id,
                    model_name=model.name,
                    model_id=model.model_id,
                    response_text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=0.0,
                    timestamp=now_iso(),
                    error=message,
                )

    async def evaluate_model(
        self,
        model: ModelConfig,
        queries: list[Query],
        output_dir: Path,
        concurrency: int = 10,
    ) -> Path:
        """
        Evaluate one model against all queries.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model.name}.jsonl"
        completed_ids = get_completed_ids(output_path)
        pending = [query for query in queries if query.id not in completed_ids]
        if not pending:
            return output_path

        semaphore = asyncio.Semaphore(concurrency)
        limiter = RateLimiter(model.rate_limit_rpm)
        adapter = PROVIDER_ADAPTERS[model.provider]
        write_lock = asyncio.Lock()
        progress = tqdm(
            total=len(pending),
            desc=f"{model.name}",
            disable=not sys.stderr.isatty(),
        )

        async def run_one(query: Query) -> None:
            async with semaphore:
                await limiter.acquire()
                result = await self._call_with_retry(model, query, adapter)
                async with write_lock:
                    write_jsonl(output_path, [result.to_dict()])
                    progress.update(1)

        await asyncio.gather(*(run_one(query) for query in pending))
        progress.close()
        return output_path

    async def evaluate_batch(
        self,
        models: list[ModelConfig],
        queries: list[Query],
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Evaluate multiple models sequentially.
        """
        results: dict[str, Path] = {}
        for model in models:
            results[model.name] = await self.evaluate_model(
                model,
                queries,
                output_dir,
                concurrency=self.config.eval_concurrency,
            )
        return results
