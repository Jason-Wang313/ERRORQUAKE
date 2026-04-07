from __future__ import annotations

from pathlib import Path

import pytest

from errorquake.score import ScoreResult, ScoringPipeline
from errorquake.utils import read_jsonl, write_jsonl


def test_score_result_round_trip() -> None:
    result = ScoreResult(
        query_id="Q1",
        model_name="m",
        primary_score=1.0,
        primary_confidence="high",
        primary_chain_of_thought="a",
        primary_identified_errors=["x"],
        secondary_score=1.5,
        secondary_confidence="low",
        secondary_chain_of_thought="b",
        secondary_identified_errors=["y"],
        final_score=1.25,
        resolution_method="average",
        timestamp="now",
    )
    assert ScoreResult.from_dict(result.to_dict()) == result


@pytest.mark.asyncio()
async def test_scoring_pipeline_and_human_review_queue(
    tmp_path: Path,
    sample_queries,
) -> None:
    responses_path = tmp_path / "responses.jsonl"
    output_path = tmp_path / "scores.jsonl"
    write_jsonl(
        responses_path,
        [
            {"query_id": sample_queries[0].id, "model_name": "m", "response_text": "Mars"},
            {"query_id": sample_queries[1].id, "model_name": "m", "response_text": "Sixth Amendment"},
            {"query_id": sample_queries[2].id, "model_name": "m", "response_text": "Intel"},
        ],
    )

    primary_scores = iter([0.0, 1.0, 2.0])
    secondary_scores = iter([0.4, 1.7, 3.6])

    async def primary(prompt, query, response):
        return {
            "score": next(primary_scores),
            "confidence": "high",
            "chain_of_thought": "p",
            "identified_errors": [],
        }

    async def secondary(prompt, query, response):
        return {
            "score": next(secondary_scores),
            "confidence": "medium",
            "chain_of_thought": "s",
            "identified_errors": [],
        }

    pipeline = ScoringPipeline(
        config=__import__("errorquake.utils", fromlist=["ProjectConfig"]).ProjectConfig(),
        primary_judge_fn=primary,
        secondary_judge_fn=secondary,
    )
    summary = await pipeline.score_responses(responses_path, sample_queries, output_path)
    assert summary["scored"] == 3
    assert summary["human_required"] == 1
    assert len(pipeline.get_human_review_queue(output_path)) == 1
    assert len(read_jsonl(output_path)) == 3

