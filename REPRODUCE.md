# ERRORQUAKE-10K — Reproduction Guide

## Quick Start

```bash
git clone https://github.com/anonymous/errorquake.git
cd errorquake
pip install -e .
```

## Requirements

- Python >= 3.9
- numpy, scipy, pandas, matplotlib, seaborn
- NVIDIA NIM API key (for re-running evaluations; not needed for analysis)

## Reproducing Analysis from Existing Data

All scored data is included in `results/`. No API calls needed.

```bash
# Experiment 1: Distribution characterization
python scripts/run_exp1.py

# Experiment 2: Matched-accuracy discriminator
python scripts/run_exp2.py
python scripts/run_exp2_robustness.py

# Experiment 3: Micro-error prediction
python scripts/run_exp3.py

# Experiment 4: Domain variation
python scripts/run_exp4.py

# Experiment 5: Scaling analysis
python scripts/run_exp5.py

# Sensitivity analyses
python scripts/run_sensitivity.py
python scripts/run_s5_mmin.py

# Judge agreement
python scripts/run_icc.py
python scripts/run_judge_agreement.py

# Generate all figures
python scripts/make_figures.py

# Oral-caliber upgrade analyses (new)
python scripts/run_oral_upgrade_analyses.py
python scripts/run_multiplicative_model_test.py
```

## Reproducing from Scratch (Requires API)

### Step 1: Generate queries
```bash
python scripts/run_generation.py --domains all --tiers all --n-per-cell 250
```

### Step 2: Evaluate models
```bash
python scripts/run_evaluation.py --models all --queries data/queries/standard_subset_4k.jsonl
```

### Step 3: Score responses
```bash
python scripts/run_scoring.py --models all
```

### Step 4: Run analysis
```bash
python scripts/run_analysis.py
```

## Data Format

### Evaluation records (`results/evaluations_10k/*.jsonl`)
```json
{
  "query_id": "BIO_T1_0036",
  "model_name": "deepseek-v3.2",
  "question": "...",
  "ground_truth": "...",
  "response_text": "...",
  "domain": "BIO",
  "tier": 1
}
```

### Score records (`results/scores_10k/*.jsonl`)
```json
{
  "query_id": "BIO_T1_0036",
  "model_name": "deepseek-v3.2",
  "primary_score": 1.0,
  "secondary_score": 1.5,
  "final_score": 1.25,
  "resolution_method": "average"
}
```

## Tests

```bash
pytest tests/ -v
```

## License

Code: Apache 2.0
Data: CC-BY-4.0
