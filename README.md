# ERRORQUAKE

ERRORQUAKE is a research toolkit for measuring the magnitude-frequency profile
of factual errors produced by language models.

Phase 0 builds the infrastructure only:

- package layout and CLI entry points
- prompt assets and scoring rubric drafts
- query, evaluation, scoring, analysis, synthetic, and reporting modules
- test suite and CI configuration

## Quick start

```bash
pip install -e ".[dev]"
ruff check src/ tests/
pytest -x -v
```

## Notes

- Query-generation prompts are drafts and require researcher review before use.
- The current evaluation catalog is a NIM-only 28-model benchmark slate.
- `scripts/run_evaluation.py --verify-access` performs live NVIDIA NIM reachability checks.
