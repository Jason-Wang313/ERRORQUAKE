from __future__ import annotations

from errorquake.magnitude import SCALE_5, SCALE_7, SCALE_11, parse_judge_output, render_judge_prompt, render_rubric, resolve_scores


def test_scales_are_sorted_and_nested() -> None:
    scores_11 = [level.score for level in SCALE_11]
    scores_7 = [level.score for level in SCALE_7]
    scores_5 = [level.score for level in SCALE_5]
    assert scores_11 == sorted(scores_11)
    assert set(scores_7).issubset(scores_11)
    assert set(scores_5).issubset(scores_11)


def test_anchors_cover_expected_structure() -> None:
    domains = []
    for level in SCALE_11:
        assert len(level.anchors) == 3
        domains.extend(anchor["domain"] for anchor in level.anchors)
    for domain in {"BIO", "LAW", "HIST", "GEO", "SCI", "TECH", "FIN", "CULT"}:
        assert domains.count(domain) >= 2


def test_render_rubric_contains_all_scores() -> None:
    rubric = render_rubric(SCALE_11)
    assert rubric
    for score in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        assert f"{score:.1f}" in rubric


def test_render_judge_prompt_contains_item_fields() -> None:
    prompt = render_judge_prompt(SCALE_11, "Q?", "GT", "RESP")
    assert "Q?" in prompt
    assert "GT" in prompt
    assert "RESP" in prompt


def test_parse_judge_output_valid_json() -> None:
    parsed = parse_judge_output(
        '{"score": 1.5, "confidence": "high", "chain_of_thought": "x", "identified_errors": ["y"]}'
    )
    assert parsed["score"] == 1.5
    assert parsed["confidence"] == "high"


def test_parse_judge_output_regex_fallback() -> None:
    parsed = parse_judge_output("score: 2.5 confidence: low")
    assert parsed["score"] == 2.5


def test_resolve_scores_boundaries() -> None:
    assert resolve_scores(1.0, 1.49) == (1.0, "primary")
    assert resolve_scores(1.0, 1.5) == (1.25, "average")
    assert resolve_scores(1.0, 2.49) == (1.745, "average")
    assert resolve_scores(1.0, 2.5) == (1.0, "human_required")

