"""Query schema, validation, and data loading."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from errorquake.utils import read_jsonl

DOMAINS = ["BIO", "LAW", "HIST", "GEO", "SCI", "TECH", "FIN", "CULT"]
TIERS = [1, 2, 3, 4, 5]
ID_PATTERN = re.compile(r"^(BIO|LAW|HIST|GEO|SCI|TECH|FIN|CULT)_T[1-5]_\d{4}$")


@dataclass
class Query:
    id: str
    domain: str
    tier: int
    question: str
    ground_truth: str
    sources: list[str]
    difficulty_rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list = valid."""
        errors: list[str] = []
        if not ID_PATTERN.match(self.id):
            errors.append(f"Invalid ID format: {self.id}")
        if self.domain not in DOMAINS:
            errors.append(f"Invalid domain: {self.domain}")
        if self.tier not in TIERS:
            errors.append(f"Invalid tier: {self.tier}")
        if not self.question.strip():
            errors.append("Empty question")
        if not self.ground_truth.strip():
            errors.append("Empty ground truth")
        if len(self.sources) < 2:
            errors.append(f"Need ≥2 sources, got {len(self.sources)}")
        return errors

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Query":
        return cls(
            id=payload["id"],
            domain=payload["domain"],
            tier=int(payload["tier"]),
            question=payload["question"],
            ground_truth=payload["ground_truth"],
            sources=list(payload.get("sources", [])),
            difficulty_rationale=payload.get("difficulty_rationale", ""),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_dataset_dir(data_dir: Path, leaf: str) -> Path:
    nested = data_dir / leaf
    if nested.exists():
        return nested
    return data_dir


def _iter_query_files(base_dir: Path) -> list[Path]:
    return sorted(path for path in base_dir.glob("*.jsonl") if path.is_file())


def _load_query_dir(base_dir: Path, domains: list[str] | None = None) -> list[Query]:
    domain_set = set(domains or DOMAINS)
    queries: list[Query] = []
    for path in _iter_query_files(base_dir):
        stem = path.stem.split("_")[0].upper()
        if stem not in domain_set:
            continue
        for record in read_jsonl(path):
            queries.append(Query.from_dict(record))
    return queries


def load_queries(data_dir: Path, domains: list[str] | None = None) -> list[Query]:
    """Load queries from JSONL files. Optionally filter by domain."""
    return _load_query_dir(_resolve_dataset_dir(data_dir, "queries"), domains=domains)


def load_reserve(data_dir: Path, domains: list[str] | None = None) -> list[Query]:
    """Load hard reserve queries from reserve directory."""
    return _load_query_dir(_resolve_dataset_dir(data_dir, "reserve"), domains=domains)


def validate_dataset(queries: list[Query]) -> dict[str, Any]:
    """
    Full dataset validation.
    """
    distribution = {domain: {tier: 0 for tier in TIERS} for domain in DOMAINS}
    errors: list[str] = []
    seen_ids = Counter(query.id for query in queries)

    valid = 0
    invalid = 0
    for query in queries:
        query_errors = query.validate()
        if query.domain in distribution and query.tier in distribution[query.domain]:
            distribution[query.domain][query.tier] += 1
        if query_errors:
            invalid += 1
            errors.extend(f"{query.id}: {item}" for item in query_errors)
        else:
            valid += 1

    duplicate_ids = sorted(query_id for query_id, count in seen_ids.items() if count > 1)
    for duplicate_id in duplicate_ids:
        errors.append(f"Duplicate ID: {duplicate_id}")

    return {
        "total": len(queries),
        "valid": valid,
        "invalid": invalid,
        "errors": errors,
        "distribution": distribution,
        "duplicate_ids": duplicate_ids,
    }


def generate_manifest(queries: list[Query], output_path: Path, **extra: Any) -> None:
    """Write manifest.json with dataset statistics."""
    validation = validate_dataset(queries)
    payload = {
        "query_count": len(queries),
        "domains": sorted({query.domain for query in queries}),
        "tiers": sorted({query.tier for query in queries}),
        "validation": validation,
    }
    payload.update(extra)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

