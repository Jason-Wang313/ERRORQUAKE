from __future__ import annotations

from pathlib import Path

import pytest

from errorquake.evaluate import (
    ALL_MODELS,
    EvaluationEngine,
    ModelConfig,
    ModelResponse,
    PROVIDER_ADAPTERS,
    RateLimiter,
    verify_model_access,
)
from errorquake.utils import ProjectConfig, read_jsonl, write_jsonl


def test_model_config_validation() -> None:
    with pytest.raises(ValueError):
        ModelConfig("", "nim", "meta/llama-3.1-8b-instruct", "NVIDIA_API_KEY")


def test_model_catalog_shape() -> None:
    assert len(ALL_MODELS) == 28
    assert len({model.name for model in ALL_MODELS}) == 28
    assert {model.provider for model in ALL_MODELS} == {"nim"}
    assert set(PROVIDER_ADAPTERS) == {"nim"}


def test_all_nim_models_use_nvidia_api_key() -> None:
    nim_models = [model for model in ALL_MODELS if model.provider == "nim"]
    assert nim_models
    assert all(model.api_key_env == "NVIDIA_API_KEY" for model in nim_models)


def test_no_duplicate_model_names() -> None:
    names = [model.name for model in ALL_MODELS]
    assert len(names) == len(set(names))


@pytest.mark.asyncio()
async def test_evaluation_checkpoint_resume(tmp_path: Path, sample_queries) -> None:
    model = ModelConfig("fake-model", "nim", "fake", "FAKE_API_KEY")
    output_dir = tmp_path / "evaluations"
    output_path = output_dir / "fake-model.jsonl"
    output_dir.mkdir()
    write_jsonl(
        output_path,
        [
            {
                "query_id": sample_queries[0].id,
                "model_name": model.name,
                "model_id": model.model_id,
                "response_text": "existing",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_ms": 1.0,
                "timestamp": "now",
                "error": None,
            }
        ],
    )

    async def fake_adapter(config: ModelConfig, question: str) -> ModelResponse:
        return ModelResponse(
            query_id="",
            model_name=config.name,
            model_id=config.model_id,
            response_text=f"answer to {question}",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
            timestamp="now",
        )

    original_adapter = PROVIDER_ADAPTERS["nim"]
    PROVIDER_ADAPTERS["nim"] = fake_adapter
    engine = EvaluationEngine(ProjectConfig(eval_concurrency=2))
    try:
        path = await engine.evaluate_model(model, sample_queries, output_dir, concurrency=2)
    finally:
        PROVIDER_ADAPTERS["nim"] = original_adapter
    records = read_jsonl(path)
    assert len(records) == 3
    assert len({record["query_id"] for record in records}) == 3


@pytest.mark.asyncio()
async def test_rate_limiter_respects_rpm() -> None:
    now = 0.0

    def clock() -> float:
        return now

    async def fake_sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    limiter = RateLimiter(rpm=2, clock=clock, sleep_fn=fake_sleep)
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()
    assert now >= 60.0


@pytest.mark.asyncio()
async def test_verify_model_access_reports_missing_keys() -> None:
    report = await verify_model_access([ModelConfig("fake", "nim", "fake", "MISSING_KEY")])
    assert report["fake"]["status"] == "missing_api_key"
