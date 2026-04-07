from __future__ import annotations

from pathlib import Path

from errorquake.utils import ProjectConfig, get_completed_ids, read_jsonl, write_jsonl


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = [{"id": "A"}, {"query_id": "B", "value": 2}]
    write_jsonl(path, records)
    assert read_jsonl(path) == records


def test_get_completed_ids_handles_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"id": "A"}, {"id": "A"}, {"query_id": "B"}])
    assert get_completed_ids(path) == {"A", "B"}


def test_project_config_save_load(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = ProjectConfig(active_scale="7-point", eval_concurrency=5)
    config.save(path)
    loaded = ProjectConfig.load(path)
    assert loaded.active_scale == "7-point"
    assert loaded.eval_concurrency == 5

