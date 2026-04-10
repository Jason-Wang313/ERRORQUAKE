"""v6 scale-up runner.

Reuses the existing run_phase4.py pipeline (worker pool, judge round-robin,
buffered writers, multi-provider fallback) but redirects paths to the v6
supplement directories. The 4K data in results/evaluations and
results/scores stays untouched until the A5 merge step.

Output:
  results/evaluations_v6_supplement/{model}.jsonl  — model responses on 6K
  results/scores_v6_supplement/{model}.jsonl       — dual-judge scores on 6K
  results/evaluation_progress_v6.json              — live progress
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "scripts"))

import run_phase4 as p4
from errorquake.utils import read_jsonl


def main() -> None:
    # Redirect ALL pipeline paths to v6 supplement locations BEFORE
    # importing/calling pipeline functions. The pipeline reads these as
    # module-level globals at runtime, so reassignment is sufficient.
    p4.SUBSET_PATH = ROOT / "data" / "queries" / "v6_supplement_6k.jsonl"
    p4.EVAL_DIR = ROOT / "results" / "evaluations_v6_supplement"
    p4.SCORES_DIR = ROOT / "results" / "scores_v6_supplement"
    p4.PROGRESS_PATH = ROOT / "results" / "evaluation_progress_v6.json"

    p4.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    p4.SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # Load the 6K supplement queries
    queries = read_jsonl(p4.SUBSET_PATH)
    if not queries:
        print(f"ERROR: no queries at {p4.SUBSET_PATH}")
        return
    print(f"v6 scale-up: {len(queries)} new queries to evaluate")
    print(f"  EVAL_DIR={p4.EVAL_DIR}")
    print(f"  SCORES_DIR={p4.SCORES_DIR}")
    print(f"  PROGRESS={p4.PROGRESS_PATH}")
    print()

    # Run the existing phase4 pipeline (eval -> score, per-model checkpointed)
    asyncio.run(p4.run_pipeline(queries))


if __name__ == "__main__":
    main()
