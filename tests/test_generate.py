from __future__ import annotations

from pathlib import Path

import pytest

from errorquake.generate import (
    RateLimiter,
    _build_batch_prompt,
    _generate_cell_candidates,
    assign_final_ids,
    generate_domain_queries,
    heuristic_match,
    parse_generation_response,
    parse_verification_batch_response,
    select_final,
    verify_queries_batch,
)
from errorquake.utils import ProjectConfig, read_jsonl


def _candidate(
    question: str,
    *,
    verified: bool = True,
    subtopic: str = "topic",
    ground_truth: str = "Alpha Bravo",
) -> dict[str, object]:
    return {
        "id": question,
        "domain": "BIO",
        "tier": 1,
        "question": question,
        "ground_truth": ground_truth,
        "sources": ["s1", "s2"],
        "difficulty_rationale": "test",
        "subtopic": subtopic,
        "verified": verified,
    }


def test_batch_prompt_construction_includes_tier_and_diversity() -> None:
    prompt = _build_batch_prompt(
        "system rules",
        "domain rules",
        3,
        25,
        ["anatomy", "anatomy", "genetics"],
    )
    assert "system rules" in prompt
    assert "domain rules" in prompt
    assert "Generate exactly 25 questions at difficulty tier T3." in prompt
    assert "anatomy(2)" in prompt
    assert "Respond with a JSON array only." in prompt


def test_parse_generation_response_handles_fences_truncation_and_nested_arrays() -> None:
    fenced = """```json
    [{"question":"Q1?","ground_truth":"A1","sources":["s1","s2"],"difficulty_rationale":"r","subtopic":"bio"}]
    ```"""
    assert len(parse_generation_response(fenced)) == 1

    truncated = (
        '[{"question":"Q1?","ground_truth":"A1","sources":["s1","s2"],'
        '"difficulty_rationale":"r","subtopic":"bio"},'
    )
    salvaged = parse_generation_response(truncated)
    assert len(salvaged) == 1
    assert salvaged[0]["question"] == "Q1?"

    wrapped = (
        '{"queries":[['
        '{"question":"Q2?","ground_truth":"A2","sources":["s1","s2"],'
        '"difficulty_rationale":"r","subtopic":"geo"}'
        ']]}'
    )
    flattened = parse_generation_response(wrapped)
    assert len(flattened) == 1
    assert flattened[0]["subtopic"] == "geo"


def test_parse_verification_batch_response_handles_fences() -> None:
    fenced = """```json
    [{"id":"RAW_BIO_T1_00001","answer":"The Fifth Amendment protects against self-incrimination."}]
    ```"""
    parsed = parse_verification_batch_response(fenced)
    assert parsed["RAW_BIO_T1_00001"].startswith("The Fifth Amendment")


def test_heuristic_match_good_and_bad_pairs() -> None:
    assert heuristic_match(
        "The Fifth Amendment protects against self-incrimination in the United States.",
        "The Fifth Amendment protects people against self-incrimination in the United States.",
    )
    assert not heuristic_match(
        "The Fifth Amendment protects against self-incrimination in the United States.",
        "The Sixth Amendment governs criminal jury trials.",
    )


@pytest.mark.asyncio()
async def test_generate_domain_queries_respects_batch_sizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_dir = tmp_path / "prompts" / "query_generation"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "system.txt").write_text("system", encoding="utf-8")
    (prompts_dir / "bio.txt").write_text("bio", encoding="utf-8")

    seen_batch_sizes: list[int] = []

    async def fake_generate_batch(**kwargs):
        seen_batch_sizes.append(kwargs["batch_size"])
        return [
            {
                "question": f"Q{index}?",
                "ground_truth": f"Answer {index}.",
                "sources": ["s1", "s2"],
                "difficulty_rationale": "test",
                "subtopic": "topic",
            }
            for index in range(kwargs["batch_size"])
        ]

    monkeypatch.setattr("errorquake.generate._generate_batch", fake_generate_batch)
    records = await generate_domain_queries("BIO", 1, 26, tmp_path / "prompts", oversample_factor=2)
    assert len(records) == 52
    assert seen_batch_sizes == [25, 25, 2]


def test_select_final_prefers_verified_queries() -> None:
    candidates = [
        _candidate(f"Question {index}?", verified=index % 2 == 0, subtopic=f"s{index}")
        for index in range(6)
    ]
    final = select_final(candidates, target_per_cell=3)
    assert len(final) == 3
    assert sum(1 for item in final if item["verified"]) >= 2


def test_select_final_enforces_diversity_cap_when_enough_candidates() -> None:
    candidates = [
        _candidate("A?", subtopic="alpha"),
        _candidate("B?", subtopic="alpha"),
        _candidate("C?", subtopic="beta"),
        _candidate("D?", subtopic="gamma"),
        _candidate("E?", subtopic="delta"),
        _candidate("F?", subtopic="epsilon"),
    ]
    final = select_final(candidates, target_per_cell=5, diversity_cap_ratio=0.20)
    counts: dict[str, int] = {}
    for item in final:
        key = str(item["subtopic"])
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) <= 1


def test_assign_final_ids_format() -> None:
    records = assign_final_ids(
        "BIO",
        3,
        [
            _candidate("Question 1?", ground_truth="Ground truth one."),
            _candidate("Question 2?", ground_truth="Ground truth two."),
        ],
    )
    assert [record["id"] for record in records] == ["BIO_T3_0001", "BIO_T3_0002"]


@pytest.mark.asyncio()
async def test_verify_queries_batch_falls_back_for_missing_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_nim_text(**kwargs):
        return '[{"id":"RAW_BIO_T1_00001","answer":"Alpha Bravo"}]'

    async def fake_verify_query(query, **kwargs):
        return {
            **query,
            "verified": True,
            "verification_answer": "Fallback answer",
            "verification_match": True,
            "verification_model": "qwen/qwen3-next-80b-a3b-instruct",
        }

    monkeypatch.setattr("errorquake.generate._call_nim_text", fake_call_nim_text)
    monkeypatch.setattr("errorquake.generate.verify_query", fake_verify_query)

    results = await verify_queries_batch(
        [
            {
                **_candidate("Question 1?", ground_truth="Alpha Bravo"),
                "id": "RAW_BIO_T1_00001",
            },
            {
                **_candidate("Question 2?", ground_truth="Fallback answer"),
                "id": "RAW_BIO_T1_00002",
            },
        ]
    )
    assert len(results) == 2
    assert any(item["verification_answer"] == "Fallback answer" for item in results)


@pytest.mark.asyncio()
async def test_generation_checkpoint_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts_dir = tmp_path / "prompts" / "query_generation"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "system.txt").write_text("system", encoding="utf-8")
    (prompts_dir / "bio.txt").write_text("bio", encoding="utf-8")

    raw_dir = tmp_path / "queries" / "raw"
    errors_path = tmp_path / "queries" / "generation_errors.jsonl"
    progress_path = raw_dir / "generation_progress.json"

    class DummyLogger:
        def info(self, *args, **kwargs) -> None:
            return None

    config = ProjectConfig(
        prompts_dir=tmp_path / "prompts",
        queries_per_cell=2,
        oversample_factor=2,
        generation_batch_size=1,
    )

    calls = {"count": 0}

    async def flaky_generate_batch(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return [
                {
                    "question": "First question?",
                    "ground_truth": "First answer.",
                    "sources": ["s1", "s2"],
                    "difficulty_rationale": "r",
                    "subtopic": "alpha",
                }
            ]
        raise RuntimeError("crash")

    monkeypatch.setattr("errorquake.generate._generate_batch", flaky_generate_batch)

    with pytest.raises(RuntimeError):
        await _generate_cell_candidates(
            domain="BIO",
            tier=1,
            prompts_dir=tmp_path / "prompts",
            raw_dir=raw_dir,
            errors_path=errors_path,
            progress_path=progress_path,
            config=config,
            rate_limiter=RateLimiter(rpm=1000),
            resume=False,
            logger=DummyLogger(),
        )

    partial = read_jsonl(raw_dir / "BIO_T1.jsonl")
    assert len(partial) == 1

    async def stable_generate_batch(**kwargs):
        idx = kwargs["batch_size"]
        return [
            {
                "question": f"Recovered {kwargs['tier']} {idx}?",
                "ground_truth": "Recovered answer.",
                "sources": ["s1", "s2"],
                "difficulty_rationale": "r",
                "subtopic": "beta",
            }
        ]

    monkeypatch.setattr("errorquake.generate._generate_batch", stable_generate_batch)
    recovered, _ = await _generate_cell_candidates(
        domain="BIO",
        tier=1,
        prompts_dir=tmp_path / "prompts",
        raw_dir=raw_dir,
        errors_path=errors_path,
        progress_path=progress_path,
        config=config,
        rate_limiter=RateLimiter(rpm=1000),
        resume=True,
        logger=DummyLogger(),
    )
    assert len(recovered) == 4


@pytest.mark.asyncio()
async def test_rate_limiter_spacing() -> None:
    now = 0.0

    def clock() -> float:
        return now

    async def fake_sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    limiter = RateLimiter(rpm=30, clock=clock, sleep_fn=fake_sleep)
    await limiter.acquire()
    await limiter.acquire()
    assert now >= 2.0
