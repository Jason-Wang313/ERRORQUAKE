# ERRORQUAKE-10K Reproduction Guide

## Setup

```bash
git clone https://github.com/anonymous/errorquake.git
cd errorquake
pip install -e ".[dev]"
```

## Fast verification

These commands check the local artifact without any API access:

```bash
ruff check src tests
pytest -q
python scripts/spot_check.py
python scripts/count_abstract.py
powershell -ExecutionPolicy Bypass -File paper/build_submission.ps1
```

## Saved analysis artifacts used by the submission

The submission-critical claims are checked directly against the saved
10K and human-validation JSON artifacts in:

- `results/analysis/v7_4k_vs_10k.json`
- `results/analysis/v10_full_human.json`
- `results/analysis/oral_upgrade/oral_upgrade_analyses.json`
- `data/human_audit/expanded_study/analysis_report.json`

`python scripts/spot_check.py` is the supported verification entry
point for these claims.

Some legacy `run_exp*.py` scripts in the repo regenerate earlier 4K or
judge-only analyses and are kept for project history, but they are not
the primary source of truth for the current submission numbers.

## Regenerating figures

```bash
python scripts/make_figures.py
```

## Building the submission PDF

```bash
powershell -ExecutionPolicy Bypass -File paper/build_submission.ps1
```

## Full evaluation pipeline

The repository also contains scripts for query generation, model
evaluation, and LLM-judge scoring. Those workflows require third-party
provider credentials and may be subject to rate limits or model
availability changes, so they are not required for reproducing the
submission from saved artifacts.

## Released artifact contents

- Saved analysis outputs: `results/analysis/`
- Released benchmark metadata: `data/release/`
- Human-validation protocol: `data/human_audit/expanded_study/protocol.md`
- Paper source and figures: `paper/`

## Licenses

- Code: MIT
- Released benchmark data/metadata: CC-BY-4.0
