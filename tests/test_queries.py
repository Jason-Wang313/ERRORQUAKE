from __future__ import annotations

import json
from pathlib import Path

from errorquake.queries import Query, generate_manifest, load_queries, validate_dataset
from errorquake.utils import write_jsonl


def test_query_validate_catches_all_errors() -> None:
    query = Query(
        id="BAD",
        domain="NOPE",
        tier=7,
        question=" ",
        ground_truth=" ",
        sources=["one"],
        difficulty_rationale="bad",
    )
    errors = query.validate()
    assert any("Invalid ID format" in error for error in errors)
    assert any("Invalid domain" in error for error in errors)
    assert any("Invalid tier" in error for error in errors)
    assert "Empty question" in errors
    assert "Empty ground truth" in errors
    assert any("Need ≥2 sources" in error for error in errors)


def test_query_validate_success(sample_query: Query) -> None:
    assert sample_query.validate() == []


def test_load_queries_round_trip_and_manifest(tmp_path: Path, sample_queries: list[Query]) -> None:
    queries_dir = tmp_path / "queries"
    queries_dir.mkdir()
    write_jsonl(queries_dir / "sci.jsonl", [sample_queries[0].to_dict()])
    write_jsonl(queries_dir / "law.jsonl", [sample_queries[1].to_dict()])
    loaded = load_queries(tmp_path)
    assert [query.id for query in loaded] == [sample_queries[1].id, sample_queries[0].id]

    manifest_path = queries_dir / "manifest.json"
    generate_manifest(loaded, manifest_path, created_by="test")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["query_count"] == 2
    assert payload["created_by"] == "test"


def test_validate_dataset_duplicates_and_distribution(sample_queries: list[Query]) -> None:
    duplicated = sample_queries + [sample_queries[0]]
    report = validate_dataset(duplicated)
    assert sample_queries[0].id in report["duplicate_ids"]
    assert report["distribution"]["SCI"][2] == 2
