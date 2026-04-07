"""Shared utilities for ERRORQUAKE."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalise_record(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    if isinstance(record, dict):
        return record
    raise TypeError(f"Unsupported record type: {type(record)!r}")


def write_jsonl(path: Path, records: list[dict[str, Any]] | list[Any]) -> None:
    """Append records to a JSONL file. Creates file if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_normalise_record(record), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file. Returns empty list if file doesn't exist.

    Skips corrupt lines (logs warning) so a single bad line from concurrent
    writes doesn't crash the whole pipeline.
    """
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    if bad_lines > 0:
        import logging
        logging.getLogger("errorquake.utils").warning(
            "read_jsonl: skipped %d corrupt lines in %s", bad_lines, path
        )
    return records


def get_completed_ids(path: Path) -> set[str]:
    """Extract all unique IDs from a JSONL checkpoint file for resume."""
    completed: set[str] = set()
    for record in read_jsonl(path):
        for key in ("id", "query_id"):
            value = record.get(key)
            if isinstance(value, str) and value:
                completed.add(value)
                break
    return completed


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime


def setup_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Configure logger with console + file handlers. ISO timestamps."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = _UtcFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


@dataclass
class ProjectConfig:
    """Central configuration for all ERRORQUAKE runs."""

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    prompts_dir: Path = Path("prompts")
    figures_dir: Path = Path("figures")

    active_scale: str = "11-point"

    generation_model: str = "meta/llama-4-maverick-17b-128e-instruct"
    generation_provider: str = "nim"
    generation_batch_size: int = 25
    generation_rpm: int = 35
    generation_max_tokens: int = 4000
    generation_timeout_s: int = 120
    oversample_factor: int = 2
    queries_per_cell: int = 250
    reserve_per_domain: int = 2000

    eval_temperature: float = 0.0
    eval_max_tokens: int = 500
    eval_concurrency: int = 10

    verification_model: str = "qwen/qwen3-next-80b-a3b-instruct"
    verification_batch_size: int = 5
    verification_rpm: int = 40
    verification_concurrency: int = 8
    verification_max_tokens: int = 1400
    verification_timeout_s: int = 90

    primary_judge: str = "qwen/qwen3-next-80b-a3b-instruct"
    secondary_judge: str = "meta/llama-4-maverick-17b-128e-instruct"
    self_score_swap_judge: str = "deepseek-ai/deepseek-v3.2"
    disagreement_threshold_average: float = 0.5
    disagreement_threshold_human: float = 1.5

    min_errors_for_fitting: int = 50
    adaptive_difficulty_threshold: float = 0.15

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        kwargs: dict[str, Any] = {}
        for field in fields(cls):
            value = payload.get(field.name, getattr(cls(), field.name))
            if field.type is Path or isinstance(getattr(cls(), field.name), Path):
                value = Path(value)
            kwargs[field.name] = value
        return cls(**kwargs)

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Path):
                value = str(value)
            payload[field.name] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()
