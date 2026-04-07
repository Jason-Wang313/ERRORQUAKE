# ERRORQUAKE — Phase 2a: Synthetic Pipeline Validation (Experiment 0)

## Purpose

Before spending days on real model evaluation, verify the scoring +
fitting pipeline can recover KNOWN distributions from synthetic data.
If it can't distinguish a power law from an exponential from uniform,
nothing downstream is trustworthy.

**This is a GO/NO-GO gate.** If the pipeline fails to recover known
distributions, we debug before spending any more API budget.

---

## Architecture at 40 RPM

The bottleneck is judge scoring. Strategy: minimize synthetic dataset
size to the smallest n that still validates recovery, while keeping
enough statistical power for distribution fitting.

**n = 500 per distribution × 3 distributions = 1,500 synthetic items.**

500 items per distribution is sufficient for:
- Discrete MLE fitting with 6-8 bins (need ~30+ per bin minimum)
- Chi-squared GOF with reasonable df
- BIC comparison across 5 distribution families
- b-value MLE with bootstrap CI

**API call budget:**
| Step | Calls | Time at 40 RPM |
|------|-------|----------------|
| Generate synthetic responses | 1,500 | ~38 min |
| Primary judge scoring | 1,500 | ~38 min |
| Secondary judge scoring | 1,500 | ~38 min |
| **Total** | **4,500** | **~1.9 hours** |

Analysis and fitting are local (zero API calls).

---

## Step-by-Step Pipeline

### Step 1: Generate Synthetic Score Distributions (LOCAL — no API)

Create three arrays of 500 target severity scores, drawn from known
distributions and discretized to the 11-point scale:

```python
def generate_synthetic_scores(distribution, n=500, scale_points=None):
    """
    Generate target scores from a known distribution, discretized to
    the nearest valid scale point.
    
    scale_points default: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    Distributions:
    
    (a) Power law, b=1.0:
        Draw from P(M=m) ∝ m^(-1.0) for m in scale_points[1:]
        (exclude 0.0 — errors only, not perfect responses)
        This should produce many 0.5s, fewer 1.0s, rare 4.0s.
    
    (b) Exponential, λ=0.8:
        Draw from P(M=m) ∝ exp(-0.8 * m) for m in scale_points[1:]
        Similar shape to power law but different tail behavior.
    
    (c) Uniform:
        Equal probability across all scale_points[1:]
        ~62 items per non-zero bin.
    """
```

Save to `data/synthetic/target_scores_{distribution}.json`.

**Verification:** Print the bin counts for each distribution. Power law
should have ~60% of mass in the 0.5-1.0 range. Uniform should have
~12.5% per bin. If these look wrong, the generation code has a bug.

### Step 2: Select Source Queries (LOCAL — no API)

For each synthetic item, select a random query from the Phase 1 dataset
(`data/queries/*.jsonl`). This gives us a real question + ground truth
to anchor the synthetic response generation.

Stratify selection: match higher target severity scores to higher-tier
queries when possible (T4-T5 queries are more likely to elicit natural
errors). This isn't critical but improves realism:

```
Target 0.5-1.0  → draw from T1-T2 queries
Target 1.5-2.5  → draw from T2-T3 queries  
Target 3.0-4.0  → draw from T3-T5 queries
```

### Step 3: Generate Synthetic Responses (1,500 NIM calls)

For each (query, target_score) pair, ask DeepSeek-V3.2 to generate a
response containing an error at the target severity level:

```
Prompt:
You are generating a test response for a scoring calibration experiment.

Question: {question}
Correct answer: {ground_truth}

Generate a response to this question that contains a factual error at
severity level {target_score} on the following scale:

{condensed_rubric — just score/label/definition, no anchors}

Your response should:
- Sound like a natural LLM response (not obviously fake)
- Contain exactly one primary error at the target severity
- For score 0.5: introduce a tiny imprecision (wrong rounding, slightly off date)
- For score 1.0-2.0: get a secondary detail wrong while keeping the core right (1.0) or get the core answer wrong (1.5-2.0)
- For score 3.0+: fabricate a fact, entity, or source

Respond with ONLY the simulated LLM response. No meta-commentary.
```

**Model:** `deepseek-ai/deepseek-v3.2` via NIM.
**Rate limit:** 35 RPM, sequential. ~43 minutes for 1,500 calls.
**Max tokens:** 300 per response.
**Temperature:** 0.7 (want variety in error types, not deterministic).

**Checkpoint:** Save each response immediately to
`data/synthetic/responses_{distribution}.jsonl` with fields:
```json
{
    "synthetic_id": "SYN_PL_0001",
    "target_score": 1.5,
    "source_query_id": "BIO_T2_0042",
    "question": "...",
    "ground_truth": "...",
    "synthetic_response": "...",
    "generation_model": "deepseek-ai/deepseek-v3.2"
}
```

### Step 4: Score with Primary Judge (1,500 NIM calls)

Score each synthetic response using the primary judge:

**Model:** `meta/llama-3.1-405b-instruct` via NIM

Use `render_judge_prompt()` from magnitude.py — the full prompt with
step-by-step procedure, all 27 anchors, and strict JSON output format.

**Rate limit:** 35 RPM, sequential. ~43 minutes.
**Max tokens:** 1000 (need space for chain_of_thought in JSON).

**Checkpoint:** Save to `data/synthetic/scores_primary_{distribution}.jsonl`:
```json
{
    "synthetic_id": "SYN_PL_0001",
    "target_score": 1.5,
    "primary_score": 1.5,
    "primary_confidence": "high",
    "primary_chain_of_thought": "...",
    "primary_identified_errors": ["..."]
}
```

### Step 5: Score with Secondary Judge (1,500 NIM calls)

Same process with the secondary judge:

**Model:** `qwen/qwen3-next-80b` via NIM

**Checkpoint:** Save to `data/synthetic/scores_secondary_{distribution}.jsonl`.

### Step 6: Resolve Scores and Fit Distributions (LOCAL — no API)

For each synthetic item:
1. Apply `resolve_scores(primary, secondary)` from magnitude.py
2. Record the final resolved score

Then for each of the 3 distributions:
1. Collect the 500 final resolved scores
2. Run `fit_all_distributions()` from analyze.py (5-way fit)
3. Run `ratio_test()` from analyze.py
4. Run `estimate_b_value()` from analyze.py (for power law set only)

### Step 7: Validate Recovery (LOCAL — no API)

For each distribution, check:

```python
def validate_recovery(true_dist, true_params, fits, b_estimate):
    """
    Pre-registered criteria:
    
    1. FAMILY RECOVERY: Does the best-BIC fit match the true family?
       - Power law data → best fit should be power_law or truncated_power_law
       - Exponential data → best fit should be exponential
       - Uniform data → no single family should dominate (all BICs close),
         OR chi-squared GOF should reject all parametric fits
    
    2. PARAMETER RECOVERY (power law only):
       Does the recovered b-value fall within ±0.3 of true b=1.0?
       (Relaxed from ±0.2 in original spec because we're using 500 items
       instead of 3000, so there's more sampling noise.)
    
    3. JUDGE CALIBRATION: 
       Spearman correlation between target_score and final_score ≥ 0.70.
       This measures whether judges can actually discriminate severity.
       If this fails, the SCALE is broken, not just the fitting.
    
    Returns:
    {
        "distribution": str,
        "family_recovered": bool,
        "true_family": str,
        "best_fit_family": str,
        "param_recovered": bool | None,
        "true_b": float | None,
        "recovered_b": float | None,
        "judge_correlation": float,  # Spearman ρ (target vs final)
        "judge_calibration_pass": bool,  # ρ ≥ 0.70
        "verdict": "PASS" | "FAIL" | "MARGINAL",
        "details": str
    }
    """
```

**VERDICT LOGIC:**

| Family recovered | Param recovered | Judge calibration | Verdict |
|---|---|---|---|
| Yes | Yes (or N/A) | ≥ 0.70 | **PASS** — proceed to Phase 3 |
| Yes | No | ≥ 0.70 | **MARGINAL** — fitting works, b estimate noisy. Acceptable. |
| No | — | ≥ 0.70 | **FAIL** — fitting pipeline broken. Debug before proceeding. |
| — | — | < 0.70 | **FAIL** — judges can't discriminate severity. Try different judge models or collapse scale. |

**The judge calibration check is the most important gate.** If ρ < 0.70
between target and judged scores, the judges are too noisy to produce
meaningful distributions regardless of how good the fitting code is.
This would trigger the cascade: try 7-point scale, then 5-level.

---

## Output Report

Save the full report to `results/analysis/experiment_0_report.json`:

```json
{
    "experiment": "Experiment 0: Synthetic Pipeline Validation",
    "date": "2026-04-XX",
    "n_per_distribution": 500,
    "primary_judge": "meta/llama-3.1-405b-instruct",
    "secondary_judge": "qwen/qwen3-next-80b",
    "scale": "11-point",
    "results": {
        "power_law": {
            "true_b": 1.0,
            "recovered_b": ...,
            "best_fit_family": "...",
            "bic_ranking": [...],
            "judge_target_correlation": ...,
            "verdict": "PASS|FAIL|MARGINAL"
        },
        "exponential": { ... },
        "uniform": { ... }
    },
    "overall_verdict": "GO|NO-GO",
    "score_disagreement_stats": {
        "primary_only": ...,
        "averaged": ...,
        "human_required": ...
    },
    "judge_confusion_matrix": {
        "target_vs_primary": [[...], ...],
        "target_vs_final": [[...], ...]
    }
}
```

Also generate the Figure 11 draft (3-panel synthetic validation plot):
- Panel A: Power law — true vs recovered magnitude-frequency
- Panel B: Exponential — true vs recovered
- Panel C: Uniform — true vs recovered

Save to `figures/fig11_synthetic_validation.pdf` and `.png`.

---

## Additional Diagnostics (LOCAL — no API)

Beyond the GO/NO-GO, compute and report:

1. **Score disagreement rate:** What fraction of items had |primary - secondary| ≥ 1.5?
   If > 20%, the judges are poorly calibrated to each other.

2. **Per-level accuracy:** For each target score (0.5, 1.0, ..., 4.0),
   what fraction of judged scores land within ±0.5 of the target?
   This reveals which severity levels the judges struggle with
   (expect 1.0-2.0 to be the hardest — the "muddy middle").

3. **Systematic bias:** Is the mean judged score consistently higher or
   lower than the mean target? If judges systematically underscore,
   the distribution will be shifted and b-values biased.

4. **Confusion matrix:** 9×9 matrix of target_score × final_score.
   Heatmap saved to `figures/judge_confusion_matrix.png`.

---

## Failure Modes and Responses

**If judges can't discriminate severity (ρ < 0.70):**
1. Check which levels are confused (confusion matrix)
2. If the muddy middle (1.0-2.5) is the problem: test 7-point scale
   (collapse 1.5 and 2.5). Re-score the SAME synthetic responses
   by remapping target scores: 1.5→1.0, 2.5→3.0. Refit. Check if
   ρ improves.
3. If ρ still < 0.70 on 7-point: test 5-level scale.
4. If 5-level ρ < 0.55: judges are fundamentally broken. Try swapping
   DeepSeek-V3.2 as primary judge instead of 405B.

**If family recovery fails:**
1. Check if the fitting code has bugs (test with perfectly generated
   discrete distributions, no judge noise — pure unit test)
2. If fitting works on clean data but fails on judged data: the judge
   noise is too high for 500 items. Increase to 1000 and re-run scoring.

**If b recovery is off by more than ±0.3:**
1. This is expected at n=500 — sampling noise is real
2. Marginal pass — note the uncertainty and proceed
3. The real evaluation has 28× more data points per model

---

## Tests

Add to `tests/test_synthetic.py`:

1. `generate_synthetic_scores("power_law", n=1000)` — verify bin
   counts decrease monotonically.
2. `generate_synthetic_scores("exponential", n=1000)` — verify bin
   counts decrease.
3. `generate_synthetic_scores("uniform", n=1000)` — verify all bins
   within 20% of expected count.
4. `validate_recovery` with mock PASS case.
5. `validate_recovery` with mock FAIL case (wrong family).
6. `validate_recovery` with mock FAIL case (low judge correlation).

---

## CLI

Update `scripts/run_synthetic.py`:

```
Usage: python scripts/run_synthetic.py [OPTIONS]

Options:
  --n INT              Items per distribution (default: 500)
  --distributions TEXT  Comma-separated (default: power_law,exponential,uniform)
  --output-dir PATH    Output directory (default: data/synthetic)
  --rpm INT            Rate limit (default: 35)
  --skip-scoring       Generate responses only, don't score
  --score-only         Score existing responses, don't regenerate
  --analyze-only       Fit/validate existing scores, no API calls
  --resume             Resume from checkpoint
```

---

## Completion Criteria

Phase 2a is DONE when:

1. `results/analysis/experiment_0_report.json` exists
2. All three distributions have a verdict
3. Overall verdict is GO, MARGINAL, or NO-GO with clear failure diagnosis
4. `figures/fig11_synthetic_validation.pdf` exists
5. `figures/judge_confusion_matrix.png` exists
6. `pytest -x -v` all green
7. `ruff check src/ tests/` clean

**If GO or MARGINAL:** Proceed to Phase 2b (psychometric pilot) and
Phase 3 (model pilot). Report the results here for review.

**If NO-GO:** Report the failure mode, diagnostics, and confusion
matrix. Do NOT proceed. We debug here before spending more budget.
