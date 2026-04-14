# ERRORQUAKE

ERRORQUAKE is the submission artifact for the ERRORQUAKE benchmark and
paper on error-severity distributions in open-weight LLMs. The repo
contains the paper source, released benchmark metadata, saved analysis
outputs, and the scripts used to regenerate the reported tables and
figures from local artifacts.

## What is included

- `paper/`: NeurIPS 2026 paper source, figures, checklist, and a
  PowerShell build script that does not rely on `latexmk`.
- `results/analysis/`: saved JSON outputs for the main experiments and
  robustness analyses.
- `data/release/`: released benchmark metadata, Croissant file, and
  datasheet.
- `scripts/`: analysis and reporting scripts. Core reproduction scripts
  now resolve paths relative to the repo root.

## Quick start

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
python scripts/spot_check.py
python scripts/count_abstract.py
powershell -ExecutionPolicy Bypass -File paper/build_submission.ps1
```

## Reproducing the paper from saved artifacts

The main reported results can be regenerated from the checked-in data
and JSON files without any API calls. See `REPRODUCE.md` for the
command list.

## Notes

- Reproducing the analysis from saved artifacts does not require model
  access.
- Re-running the full evaluation pipeline from scratch depends on
  third-party hosted models, provider credentials, and rate limits, so
  it is not required for artifact verification.

## Licenses

- Code: MIT
- Released benchmark data/metadata: CC-BY-4.0
