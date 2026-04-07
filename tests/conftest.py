from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def sample_query():
    from errorquake.queries import Query

    return Query(
        id="SCI_T2_0001",
        domain="SCI",
        tier=2,
        question="What planet is known as the Red Planet?",
        ground_truth="Mars",
        sources=["https://science.nasa.gov", "https://www.britannica.com"],
        difficulty_rationale="Common astronomy recall item.",
    )


@pytest.fixture()
def sample_queries(sample_query):
    from errorquake.queries import Query

    queries = [sample_query]
    queries.append(
        Query(
            id="LAW_T3_0002",
            domain="LAW",
            tier=3,
            question="Which amendment protects against self-incrimination?",
            ground_truth="The Fifth Amendment",
            sources=["https://constitution.congress.gov", "https://www.oyez.org"],
            difficulty_rationale="Canonical constitutional law fact.",
        )
    )
    queries.append(
        Query(
            id="TECH_T1_0003",
            domain="TECH",
            tier=1,
            question="Which company created CUDA?",
            ground_truth="NVIDIA",
            sources=["https://developer.nvidia.com", "https://www.britannica.com"],
            difficulty_rationale="Widely known technical attribution.",
        )
    )
    return queries
