"""Dual-judge scoring pipeline with disagreement resolution."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from errorquake.magnitude import get_scale, parse_judge_output, render_judge_prompt, resolve_scores
from errorquake.queries import Query
from errorquake.utils import ProjectConfig, get_completed_ids, now_iso, read_jsonl, write_jsonl


@dataclass
class ScoreResult:
    query_id: str
    model_name: str
    primary_score: float
    primary_confidence: str
    primary_chain_of_thought: str
    primary_identified_errors: list[str]
    secondary_score: float
    secondary_confidence: str
    secondary_chain_of_thought: str
    secondary_identified_errors: list[str]
    final_score: float
    resolution_method: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoreResult":
        return cls(**payload)


JudgeCallable = Callable[[str, Query, dict[str, Any]], Awaitable[dict[str, Any] | str]]


async def _default_judge(_: str, query: Query, response: dict[str, Any]) -> dict[str, Any]:
    ground_truth = query.ground_truth.strip().casefold()
    candidate = str(response.get("response_text", "")).strip().casefold()
    if ground_truth and ground_truth in candidate:
        score = 0.0
        errors: list[str] = []
    else:
        score = 2.0
        errors = ["Ground truth not reflected in the response."]
    return {
        "score": score,
        "confidence": "medium",
        "chain_of_thought": "Offline heuristic judge used for phase-0 infrastructure testing.",
        "identified_errors": errors,
    }


class ScoringPipeline:
    """
    Scores model responses using two LLM judges.
    """

    def __init__(
        self,
        config: ProjectConfig,
        primary_judge_fn: JudgeCallable | None = None,
        secondary_judge_fn: JudgeCallable | None = None,
    ) -> None:
        self.config = config
        self.primary_judge_fn = primary_judge_fn or _default_judge
        self.secondary_judge_fn = secondary_judge_fn or _default_judge

    async def _normalise_judge_result(
        self,
        judge_fn: JudgeCallable,
        prompt: str,
        query: Query,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        raw = await judge_fn(prompt, query, response)
        if isinstance(raw, str):
            return parse_judge_output(raw)
        return raw

    async def score_responses(
        self,
        responses_path: Path,
        queries: list[Query],
        output_path: Path,
        scale_name: str = "11-point",
        concurrency: int = 10,
    ) -> dict[str, Any]:
        """
        Score all responses for a model.
        """
        query_lookup = {query.id: query for query in queries}
        scale = get_scale(scale_name)
        completed_ids = get_completed_ids(output_path)
        responses = [
            record
            for record in read_jsonl(responses_path)
            if record.get("query_id") not in completed_ids
        ]

        semaphore = asyncio.Semaphore(concurrency)
        write_lock = asyncio.Lock()
        summary = {
            "total": len(read_jsonl(responses_path)),
            "scored": 0,
            "human_required": 0,
            "disagreement_stats": {"primary": 0, "average": 0, "human_required": 0},
        }

        async def score_one(response: dict[str, Any]) -> None:
            query_id = str(response.get("query_id"))
            query = query_lookup.get(query_id)
            if query is None:
                return
            primary_prompt = render_judge_prompt(
                scale=scale,
                question=query.question,
                ground_truth=query.ground_truth,
                model_response=str(response.get("response_text", "")),
                judge_role="primary",
            )
            secondary_prompt = render_judge_prompt(
                scale=scale,
                question=query.question,
                ground_truth=query.ground_truth,
                model_response=str(response.get("response_text", "")),
                judge_role="secondary",
            )

            async with semaphore:
                primary_result, secondary_result = await asyncio.gather(
                    self._normalise_judge_result(
                        self.primary_judge_fn,
                        primary_prompt,
                        query,
                        response,
                    ),
                    self._normalise_judge_result(
                        self.secondary_judge_fn,
                        secondary_prompt,
                        query,
                        response,
                    ),
                )

            final_score, resolution_method = resolve_scores(
                float(primary_result["score"]),
                float(secondary_result["score"]),
            )
            score_result = ScoreResult(
                query_id=query_id,
                model_name=str(response.get("model_name", "")),
                primary_score=float(primary_result["score"]),
                primary_confidence=str(primary_result.get("confidence", "unknown")),
                primary_chain_of_thought=str(primary_result.get("chain_of_thought", "")),
                primary_identified_errors=list(primary_result.get("identified_errors", [])),
                secondary_score=float(secondary_result["score"]),
                secondary_confidence=str(secondary_result.get("confidence", "unknown")),
                secondary_chain_of_thought=str(secondary_result.get("chain_of_thought", "")),
                secondary_identified_errors=list(secondary_result.get("identified_errors", [])),
                final_score=final_score,
                resolution_method=resolution_method,
                timestamp=now_iso(),
            )
            async with write_lock:
                write_jsonl(output_path, [score_result.to_dict()])
                summary["scored"] += 1
                if resolution_method == "human_required":
                    summary["human_required"] += 1
                summary["disagreement_stats"][resolution_method] += 1

        await asyncio.gather(*(score_one(response) for response in responses))
        return summary

    def get_human_review_queue(self, scores_path: Path) -> list[dict[str, Any]]:
        """Extract items flagged for human adjudication."""
        return [
            record
            for record in read_jsonl(scores_path)
            if record.get("resolution_method") == "human_required"
        ]
