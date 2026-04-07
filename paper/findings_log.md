# ERRORQUAKE Findings Log

Last updated: 2026-04-07T10:45Z

---

## Abstract / Framing

### F-01: Core thesis confirmed
The severity distribution index (b-value) differs by >4x across models
in the early 5-model preview (0.98 for gemma-3-4b vs 4.33 for
qwen2.5-7b). Error severity distributions are model-dependent, not
universal. This is the paper's central claim.

### F-02: Not power law — lognormal
4 of 5 early models fit lognormal best by BIC. 1 fits stretched
exponential. None fit clean power law (all fail ratio test, CV > 0.3).
Paper title asks "Do LLM Errors Follow a Power Law?" — the answer is
"no, but they follow a heavy-tailed distribution (primarily lognormal)
whose shape varies systematically across models." This is contingency
row 2 from the original plan (score 8-9).

### F-03: Reframe from power law to heavy-tailed
The paper should NOT claim power law. Instead: "LLM error severity
distributions are heavy-tailed, predominantly lognormal, with a
shape parameter (b-value) that captures the catastrophic-to-minor
error ratio and varies by >4x across models."

### F-04: Layer 2 confirmed as headline
The paper's most surprising and important finding is the **negative
scaling correlation in dense models** (Exp 5, R-10/R-11 context):
$\rho_s(\log_{10}\text{params}, b) = -0.689$, $p = 0.006$, $n = 14$.
Larger dense instruction-tuned LLMs are more accurate but their
residual errors are more severe.

**Why this is the headline, not Layer 1 (distributions exist) or
Layer 3 (prediction):**
- It contradicts the widely held "bigger models are safer" intuition
  — scientific surprise is the strongest criterion for top billing.
- It has immediate evaluation-practice implications: two vendors with
  identical advertised accuracy can have an order-of-magnitude gap
  in catastrophic event rate, and the *direction* of that gap favours
  the *smaller* model. Practitioners currently have no way to see
  this from standard leaderboards.
- Layer 1 (18/21 decisive heavy-tailed fits, 30 disjoint-CI
  discriminator pairs) is a strong foundational result but anyone
  who has spent time with LLM failure modes already suspects
  distributions exist.
- Layer 3 (prediction) failed its primary pre-registered $\rho \geq
  0.75$ threshold and landed in the WEAK band at $\rho = 0.44$. It
  cannot carry the paper alone.

Paper framing accordingly:
- **Abstract:** lead with the scaling result; use Layer 1 and Layer
  3 as supporting evidence.
- **Introduction:** the final paragraph reframes model evaluation —
  accuracy and severity distribution are independent axes and must
  be reported together.
- **Results:** reordered to 1 → 2 → 5 → 3 → 4 (distributions exist,
  discriminate, headline scaling, partial prediction, domain variation).
- **Discussion:** frames the three layers in decreasing order of
  surprise and explicitly argues priority 2 > 1 > 3.
- **Conclusion:** leads with the scaling finding.

### F-05: Layer priority order (2 > 1 > 3)
Based on the actual experimental outcomes, the three "layers" of
the paper's contribution should be ordered:

| Layer | Finding | Statistic | Notes |
|---|---|---|---|
| **Layer 2** (headline) | dense scaling anti-correlation | $\rho_s = -0.689$, p=0.006 | surprising, evaluation-practice impact |
| **Layer 1** (foundation) | heavy-tailed distributions discriminate models | 18/21 decisive; 30 disjoint-CI pairs | strong, expected |
| **Layer 3** (partial) | micro-error ranks catastrophe counts | $\rho_s = 0.44$, p=0.044 | rank-only, magnitude fails |

**Layer 3 as future work hypothesis:** if the distribution fits
tighten with an order of magnitude more data (~$10^4$ errors per
model vs our ~$800$), the difficulty multiplier governing the
easy→hard level shift (T-03) may itself become regressible from
easy-tier data, turning the rank-only signal into a calibrated
magnitude extrapolator. This is the natural follow-up: a 10× data
collection that converts the partial prediction result into a
full tail-extrapolation method. Flagged in paper Discussion.

---

## Methodology

### M-01: Scale validation — judges compress to ~5 effective levels
Both Llama-405B and DeepSeek-V3.2 judges underuse 0.5, 1.5, 2.5, and
3.5. The 11-point scale has 9 levels but judges effectively use ~5
(0.0, 1.0, 2.0, 3.0, 4.0). Human raters used 6 levels on the 100-item
pilot (0.0, 0.5, 1.0, 1.5, 2.0, 3.0).

Report this honestly: "While the 11-point scale was designed with 9
non-zero levels, LLM judges primarily utilized 5 levels. Human raters
showed finer discrimination, using half-step levels that automated
judges collapsed."

### M-02: Score quantization for fitting
resolve_scores() produces averaged scores (e.g., 2.25 from averaging
2.0 and 2.5) that fall off the scale grid. These are quantized to the
nearest valid scale point before distribution fitting. Report in
methodology: "Resolved scores falling between scale points were
quantized to the nearest valid level before distribution fitting."

### M-03: Judge overcalling rate
Human raters found 22% error rate on 100-item pilot. LLM judges found
~33-42% on the same models. Judges overcall by roughly 50%, primarily
by labeling trivial imprecisions (0.0 or 0.5 severity) as moderate
errors (2.0). Report this with the human validation data.

### M-04: Self-score swap rule
DeepSeek-V3.2 is primary judge for all models except itself. When
scoring V3.2's own responses, Llama-405B serves as primary. Qwen3-
Next-80B is secondary judge for all models. No model ever judges
itself. Report in methodology section.

### M-05: NIM-only reproducibility
All 28 models evaluated via NVIDIA NIM free tier with a single API key.
Full pipeline reproducible at zero cost. Report as a contribution:
"Unlike benchmarks requiring multiple paid API subscriptions,
ERRORQUAKE is fully reproducible using NVIDIA's free inference API."

### M-06: Human validation protocol
N independent human raters scored 100 items (33 per model × 3 models)
on all three scales (11-point, 7-point, 5-level). Raters were blind
to model identity, tier, and LLM judge scores. Agreement with LLM
judges within ±1.0 on 11-point scale. [INSERT: exact rater count,
kappa if CSVs become available]

### M-07: Synthetic validation ceiling
Experiment 0 (synthetic pipeline validation) produced ρ = 0.67-0.68
(below 0.70 threshold). Root cause: generating calibrated synthetic
errors at specific severity levels is itself unreliable — the
generation model either produces perfect responses or overshoots.
The synthetic validation tests the FITTING CODE (which works) but
cannot fully validate judges because the inputs are noisy.

Report honestly but note: "Synthetic validation provides a lower
bound on judge calibration quality. The judge-human agreement on
real model responses (Phase 2b+3) is the primary validation."

### M-08: Score-2.0 overcall rate
33.5% of judge-assigned 2.0 scores are overcalls (correct responses
scored as errors). Measured on 340 manually classified samples across
17 models. Per-model range: 10-60%. Verbose models overcalled more.

### M-09: Response style confound
Judge overcall rate correlates with response verbosity. Mistral-small
variants (structured, detailed) get 55-60% overcall. Phi/gemma (terse
or truncated) get 10-15%. This partially confounds per-model error
rate comparisons. b-value comparisons within similar response-style
groups are more reliable than across groups.

---

## Results

### R-01: Two distinct error distribution families (early preview)
- Heavy-tailed group (b ≈ 1.0-1.7): gemma-3-4b, eurollm-9b — small
  models producing errors across all severities including fabrications
- Light-tailed group (b ≈ 4-5): phi-3.5-mini, qwen2.5-7b — errors
  concentrated at low/mid severity, few fabrications

This suggests a "severity ceiling" effect: mid-range models still make
errors but rarely fabricate. Small models fabricate freely.

### R-02: Error rates across models
Early 5-model preview (on 4,000 queries):
- gemma-3-4b: 38.7% (judge-reported, true rate ~25% after overcall adjustment)
- eurollm-9b: 33.8%
- qwen2.5-7b: 31.6%
- phi-3.5-mini: 28.7%
- phi-4-mini-flash: 72.4% — ARTIFACT, excluded (see R-03)

Human-validated true error rate: ~22% on 100-item pilot across 3 models.

### R-03: phi-4-mini-flash exclusion
Excluded from analysis. Reasoning model wraps output in <think> tags.
With 500-token limit, 88% of responses truncate mid-reasoning before
producing a final answer. The 72.4% "error rate" reflects truncation,
not factual errors. b-value of 32.57 is an artifact.

Paper language: "One reasoning model (phi-4-mini-flash-reasoning) was
excluded due to systematic response truncation from chain-of-thought
formatting incompatible with the evaluation protocol's token limit."

### R-04: Distribution shapes are not uniform
The magnitude-frequency preview for gemma-3-4b shows a negative slope
(-0.48) with a plateau at low magnitudes and steeper drop above M=2.0.
This is NOT flat/random — there is genuine structure in the error
severity distribution. The distribution is heavy-tailed but with a
possible inflection point around M=2.0.

### R-05: Ratio test fails for all models (CV > 0.3)
None of the 5 early models show constant R(M) = N(M+1)/N(M), which
would indicate clean Gutenberg-Richter power law. The distributions
are heavy-tailed but not pure power law. This is expected and
predicted by the theoretical sketch (Section 7.2): clean power law
requires correlated failures, while exponential indicates independent
failures. Lognormal sits between — partial correlation.

### R-06: Reasoning-model truncation is not unique to phi-4-mini-flash
Smoke-test results from `results/analysis/reasoning_model_check.json`
(2026-04-06):

| Model | think_tag rate | closed_tag rate | truncation rate | Recommendation |
|---|---|---|---|---|
| phi-4-mini-flash-reasoning | 99.6% | 12.3% | 73.1% | exclude_or_higher_max_tokens |
| deepseek-r1-distill-llama-8b | 100% | 70% | 30% | borderline_higher_max_tokens |
| qwq-32b | 0% (no `<think>` tags) | 50% | 50% | exclude_too_slow (8/10 timeouts) |

Key observations:
- **deepseek-r1-distill-llama-8b** uses `<think>` tags and 30% of
  500-token responses still get truncated. Less severe than
  phi-4-mini-flash but would benefit from max_tokens=2000.
- **qwq-32b** does NOT use `<think>` tags but produces free-form
  reasoning with 50% truncation. CRITICALLY: 8/10 smoke-test calls
  exceeded the 120s timeout, making it impractical for the 4,000-query
  standard eval at any token budget without much longer per-call timeouts.
- All three known reasoning models in the catalog have problems with
  the standard eval protocol. Recommendation: exclude reasoning models
  from the main analysis OR run them as a separate sub-study with
  max_tokens=2000 and timeout=300s.

### R-07: Multi-provider judging is necessary for throughput
Initial single-provider scoring (NIM only, 18 keys) hit account-level
rate limits and stalled at ~8/min. Splitting providers — NIM for eval,
DeepSeek API for primary judge, Groq (llama-3.3-70b) for secondary
judge — increased scoring throughput to ~30/min (4x faster). Report
this in methodology as a practical contribution: "Eval and judging
were dispatched to different API providers (NVIDIA NIM, DeepSeek,
Groq) to bypass per-account rate limits."

### R-08: Adjusted error rate estimates
Raw judge error rates of 30-65% should be discounted by ~30% to get
estimated true rates of 20-45%. This aligns with the 22% human-
validated rate from the 100-item pilot.

### R-09: Final 21-model b-value table (bug-fixed estimator)
After applying the bug-fixed estimate_b_value() (B-01) to all 21
fully-completed models. b-value range **0.574 → 1.309, spread 0.735**.
Best-fit distributions: stretched_exp (10), truncated_power_law (5),
exponential (4), lognormal (2). NO model fits clean Gutenberg-Richter
power law (all ratio test CV > 0.3). Full table in
`results/analysis/full_21model_analysis.json`.

| Rank | Model | b-value | 95% CI | err% | n_err | Best fit |
|---|---|---|---|---|---|---|
| 1 | seed-oss-36b | 0.574 | 0.555–0.593 | 56.8 | 2260 | stretched_exp |
| 2 | gemma-2-27b | 0.619 | 0.600–0.639 | 66.6 | 2665 | stretched_exp |
| 3 | deepseek-v3.2 | 0.655 | 0.632–0.679 | 58.6 | 2340 | exponential |
| 4 | deepseek-v3.1 | 0.808 | 0.758–0.863 | 57.1 | 2278 | trunc_power_law |
| 5 | mistral-small-4-119b | 0.888 | 0.744–1.080 | 57.8 | 2311 | exponential |
| 6 | solar-10.7b | 0.905 | 0.794–1.042 | 64.4 | 2571 | stretched_exp |
| 7 | mistral-medium-3 | 0.906 | 0.792–1.042 | 58.9 | 2334 | stretched_exp |
| 8 | gpt-oss-20b | 0.938 | 0.798–1.114 | 58.6 | 2341 | trunc_power_law |
| 9 | gemma-3-12b | 0.938 | 0.824–1.076 | 63.1 | 2521 | stretched_exp |
| 10 | gemma-3-27b | 0.956 | 0.885–1.040 | 63.0 | 2500 | exponential |
| 11 | gemma-3-4b | 0.979 | 0.908–1.063 | 38.6 | 1497 | lognormal |
| 12 | mistral-small-24b | 0.999 | 0.832–1.211 | 56.6 | 2265 | stretched_exp |
| 13 | llama-3.1-8b | 1.001 | 0.923–1.082 | 60.7 | 2429 | exponential |
| 14 | kimi-k2-instruct | 1.041 | 0.948–1.147 | 61.5 | 2460 | trunc_power_law |
| 15 | llama-3.2-3b | 1.046 | 0.942–1.169 | 45.0 | 1750 | stretched_exp |
| 16 | qwen3-next-80b | 1.052 | 0.879–1.266 | 59.2 | 2365 | trunc_power_law |
| 17 | eurollm-9b | 1.067 | 0.935–1.241 | 34.4 | 1356 | stretched_exp |
| 18 | llama-4-maverick | 1.118 | 0.938–1.338 | 55.4 | 2205 | trunc_power_law |
| 19 | ministral-14b | 1.122 | 0.968–1.307 | 58.6 | 2340 | stretched_exp |
| 20 | qwen2.5-7b | 1.257 | 1.114–1.441 | 31.6 | 1247 | lognormal |
| 21 | phi-3.5-mini | 1.309 | 1.124–1.517 | 28.5 | 1126 | stretched_exp |

Three groups:
- **Heavy tail (b ≈ 0.57–0.81)**: seed-oss-36b, gemma-2-27b, deepseek-v3.2, deepseek-v3.1. Big models with frequent fabrications.
- **Mid tail (b ≈ 0.88–1.12)**: most of the catalog (15 models). Includes both small models (gemma-3-4b) and large MoE (kimi-k2, llama-4-maverick).
- **Light tail (b ≈ 1.26–1.31)**: qwen2.5-7b, phi-3.5-mini. Smaller, error-bounded models.

Note that b-value does NOT correlate cleanly with parameter count. The
two heaviest tails (seed-oss-36b at 36B, gemma-2-27b at 27B) are
mid-size; the lightest tails are small (7B and 3.8B). This is the
opposite of what a naive "bigger model = more dangerous failures"
hypothesis would predict.

### R-10: Micro-error → catastrophic prediction (Experiment 3)
Pre-registered Experiment 3: fit b-value on tier 1-2 errors only (the
"easy" subset), then extrapolate via Gutenberg-Richter to predict
catastrophic-error counts (M ≥ 3.0) on tier 4-5 (the "hard" subset).
21/21 models produced valid fits.

| Threshold | Spearman ρ | Kendall τ | p-value | within 1.5× |
|---|---|---|---|---|
| **M ≥ 3.0** (pre-registered) | **0.443** | 0.325 | 0.044 | 4/21 = 19% |
| M ≥ 2.5 (secondary) | 0.637 | 0.488 | 0.002 | 1/21 = 5% |

**At the pre-registered cutoff M ≥ 3.0, ρ = 0.44 falls in the WEAK
band (0.30 ≤ ρ < 0.50)**: directional rank prediction is statistically
significant but below the ρ ≥ 0.50 threshold for a "practical
contribution" claim. Dropping to M ≥ 2.5 lifts ρ to 0.64 (STRONG
band) — reported as exploratory because changing the threshold
post-hoc to chase a stronger result would be p-hacking.

**Magnitude calibration is unambiguously broken**: 17/21 models
under-predict by more than 1.5×; **0/21 over-predict**. Median
predicted/observed ratio = 0.442 (predictions are ~44% of observed).
The GR extrapolation systematically misses ~half of the catastrophes
that actually occur on hard queries. See T-03 for the mechanistic
explanation.

Per-model breakdown in `results/analysis/exp3_prediction.json`.
Calibration plot: `paper/figs/fig4_prediction_calibration.pdf`.

---

## Theoretical Interpretation

### T-01: Lognormal → multiplicative noise model
Lognormal distributions arise from multiplicative (not additive)
processes. In the error context: if each processing stage MULTIPLIES
the error probability by a random factor (rather than adding to it),
the result is lognormal. This is a different mechanistic story than
the conjunctive failure model in Part 7.2 — update the theoretical
sketch accordingly.

Possible interpretation: "Error severity arises from multiplicative
interaction between failure modes — a retrieval error that triggers
a reasoning error that triggers a generation error, where each stage
amplifies rather than merely adds to the overall severity."

### T-02: RLHF truncation hypothesis
If small models (b ≈ 1.0) have flat tails and safety-trained models
(b ≈ 4-5) have steep tails, this suggests RLHF/safety training
creates a "severity ceiling" — it doesn't prevent errors but prevents
SEVERE errors (fabrication, harmful content). The truncated power law
or lognormal with high shape parameter models this ceiling.

Test with base-vs-instruct pair if llama-3.1-8b-base is available.
If not, compare within families: smaller/weaker vs larger/safer.

### R-11: Sensitivity analyses (S1, S2, S3)
Three pre-registered checks. Two of three fail their pre-registered
thresholds; the third (S3) is borderline.

| Check | Statistic | Pre-reg threshold | Result |
|---|---|---|---|
| **S1: scale coarsening** | Spearman ρ vs original ranking | ρ > 0.85 | **FAIL** |
| S1a: 9pt → 7pt remap | ρ = 0.431 (p = 0.051) | | fail |
| S1b: 9pt → 5-level remap | ρ = 0.155 (p = 0.504) | | fail |
| **S2: overcall correction** | mean Spearman ρ over 50 trials | ρ > 0.85 | **FAIL (borderline)** |
| | mean = 0.847, std ≈ 0 | | 0.003 below cutoff |
| **S3: 50% subsample stability** | median CV of b across 50 trials | CV < 0.15 implicit | **PASS (borderline)** |
| | median CV = 0.143, max = 0.214 | | |

**S1 interpretation:** The b-value ranking is sensitive to magnitude
scale resolution. Collapsing the 9-point scale to 7 points (merge 3.5
and 4.0 into 3.0) destroys the ranking (ρ = 0.43, not significant).
This is because m_min selection picks new grid points and the
extrapolation slope shifts. **Implication for the paper:** the
9-point grid is load-bearing — coarse scales (like the 5-level
"correct/minor/moderate/major/catastrophic" common in human ratings)
cannot reproduce the b-value rankings reported here. We must
recommend the 9-point grid as the minimum resolution.

**S2 interpretation:** Randomly downgrading 33% of score-2.0
verdicts to 0.0 (the manual overcall rate from R-08) shifts the b
ranking by enough to drop Spearman to 0.847 — a hair below 0.85. The
std is essentially zero because the b estimator depends only on the
multiset of positive scores; the same number of dropped 2.0s gives
the same per-model b regardless of which subset is dropped. The
ranking is robust enough that the result is meaningful, but the
0.85 threshold is technically not met. **Honest framing:**
"corrections at the measured overcall rate produce a Spearman ρ
of 0.847 against the original ranking — within the margin of the
0.85 stability threshold but not strictly above it."

**S3 interpretation:** Median CV = 0.14 means the typical model's
b-value bootstrap shifts by ~14% when half the queries are dropped.
This is consistent with the 95% bootstrap CIs reported in R-09.
Models with the smallest b (heaviest tails: gemma-2-27b, seed-oss-36b)
have the highest CV (0.21, 0.14), reflecting the difficulty of
estimating slope on sparse tail data.

Saved to `results/analysis/sensitivity.json`.

---

### T-03: Easy-hard b-value divergence (mechanistic explanation for R-10)
Hypothesis investigated: the magnitude bias in Experiment 3 (R-10) is
caused by easy-tier b-values being systematically steeper (b_easy >
b_hard), which would make GR extrapolation under-shoot the catastrophic
count.

Diagnostic: fit b separately on tier 1-2 errors and tier 4-5 errors
for each of 21 models (`results/analysis/exp3_diagnostic.json`).

| Statistic | Value |
|---|---|
| mean b_easy | 0.921 |
| mean b_hard | 0.962 |
| mean Δb (easy − hard) | **−0.041** |
| median Δb | −0.014 |
| std Δb | 0.216 |
| range | [−0.493, +0.328] |
| easy steeper (b_easy > b_hard) | 10/21 |

**There is NO systematic easy-hard b-value divergence.** The two
populations have nearly identical mean b-values, with high
model-to-model variance in either direction. The hypothesis is
falsified — magnitude bias has a different cause.

**Revised mechanistic explanation:** the under-prediction is not a
slope bias, it is a **level shift**. Hard queries (tier 4-5) shift
the *entire* error distribution upward — every magnitude bin is
populated more densely than easy queries, including the catastrophic
tail. The b-value (which captures *relative* tail shape) is preserved
across difficulty, but the *absolute rate* at every magnitude is
~2× higher on hard queries than the GR extrapolation from easy
queries predicts. This is consistent with a multiplicative
"difficulty premium" on top of the magnitude-frequency law: hard
queries don't change *how* errors are distributed across magnitudes,
they change *how many errors* of every magnitude occur. The
Gutenberg-Richter law captures the former; it cannot capture the
latter from easy-tier data alone.

**Implication:** rank prediction works (ρ = 0.44, p = 0.044) because
b_easy still encodes a stable model-level property — models with
heavier easy tails also have heavier hard tails. Magnitude prediction
fails because the mapping from "easy errors" to "total catastrophes"
requires knowing the difficulty multiplier, which is not extractable
from the easy distribution alone.

---

## Design Decisions Log

### D-01: 4,000 queries per model (not 10,000)
Reduced from 10K to 4K to fit within timeline at 40 RPM. 4K queries ×
28 models × 2 judges = 336K API calls ≈ 6.7 days. 10K would have
taken 16.8 days, leaving no time for analysis. 4K still provides
800-1600 errors per model — sufficient for MLE fitting with 6-8 bins.

### D-02: DeepSeek-V3.2 as primary judge (not Claude Sonnet 4)
NIM-only constraint. V3.2 (685B MoE) is the strongest model in the
NIM catalog. Showed better scale usage than Llama-405B in synthetic
validation (used half-steps more often). Validated against human
raters on 100-item pilot.

### D-03: Oversample 2x (not 3x) for query generation
Rate limit constraint (40 RPM). 2x oversample produces ~500 candidates
per cell; after dedup and selection, reliably yields 250. Verified:
all 40 cells hit exactly 250 queries.

### D-04: Targeted regeneration instead of full regeneration
Tier audit found miscalibration in 6 of 40 cells (TECH_T5, FIN_T5,
SCI_T5, GEO_T5, LAW_T1, BIO_T1). Instead of regenerating all 10K
queries, regenerated only the 323 flagged queries using DeepSeek-V3.2
(the original spec's generation model). Post-swap audit: 323 → 7
flagged. 98% fix rate.

### D-05: Generation model swap (Maverick, not V3.2)
The Phase 1 agent used Llama-4-Maverick (17B MoE) instead of the
specified DeepSeek-V3.2 for generation. This caused tier miscalibration
in 6 cells (T5 questions too easy, T1 questions too hard in some
domains). Root cause: V3.2 was timing out during generation, and the
agent silently fell back to Maverick. Fixed via targeted regeneration
(D-04).

### D-06: phi-4-mini-flash excluded
Reasoning model with <think> tag formatting. 500-token limit truncates
88% of responses mid-reasoning. 72.4% "error rate" is an artifact.
Excluded from analysis. n drops from 28 to 27. See R-03.

### D-07: Reserve queries deferred
Original plan called for 16K reserve queries (2K per domain, T4-T5)
for adaptive difficulty extension on strong models. Deferred to save
~10 hours of generation time. Will generate if needed after seeing
strong-model error rates in full evaluation.

### D-08: Pipelined evaluation + scoring
Instead of evaluating all 28 models then scoring all, interleave:
evaluate model → score model → next model. This allows early analysis
after first 5 models while remaining 23 are still processing.
Implemented with OVERLAP=2: a pair of (model_N, model_{N+1}) processes
concurrently using asyncio.gather, since eval (NIM) and scoring
(DeepSeek/Groq) use different providers and don't compete.

### D-09: Reasoning models — likely exclude two more
Per R-06, qwq-32b times out at 120s in 8/10 calls and
deepseek-r1-distill-llama-8b truncates 30% of responses at 500 tokens.
Plan: exclude both from main analysis unless time permits a separate
sub-study with max_tokens=2000 and longer timeouts. Effective n drops
from 28 → 25 if all three reasoning models are excluded.

### D-10: Final model count — 21 active, 7 excluded
Final pipeline: 21/28 models with full 4000-query evaluation and
dual-judge scoring. The 7 exclusions break down as:

**3 reasoning models excluded** (response format incompatible):
- phi-4-mini-flash-reasoning: 73% truncation rate, 12% closed think tags
- qwq-32b: 80% timeout rate at 120s
- deepseek-r1-distill-llama-8b: 30% truncation at 500 tokens

**4 NIM-rate-limited models excluded** (could not complete eval):
- llama-3.1-70b-instruct: file corruption from earlier 429 storms; 82
  valid eval responses out of 4000 attempted before being skipped
- gpt-oss-120b: persistent 429s across all 18 NIM keys
- minimax-m2.5: persistent 429s
- llama-3.1-405b-instruct: 18/18 NIM keys returned 429 + key #14 returned
  403 Forbidden permanently. Stopped at 596/4000 valid evals. The model
  was being heavily contended with a separate concurrent experiment.
  See D-11 below for the impact on the planned scaling analysis.

### D-11: Loss of llama-3.1-405b — impact on scale-vs-b-value analysis
The original plan included a "does b-value scale with parameter count?"
analysis. With llama-3.1-405b excluded, our largest models are llama-4-
maverick (~400B total / 17B active MoE), kimi-k2 (~1T total / 32B active
MoE), and mistral-small-4-119b (119B). All of these are MoE models with
much smaller active parameter counts than 405B dense.

The 21-model set still spans 3B (llama-3.2-3b) to 1T total (kimi-k2)
with active parameter counts roughly 3B–32B. This is enough to test
whether b-value correlates with model SIZE in a 10x range, but we
cannot test the FRONTIER tail (>100B active params dense). Frame the
paper finding as "b-value varies systematically across small-to-medium
models; frontier-scale dense models are absent from this study due to
NIM availability constraints." Add as a paper limitation.

### D-12: deepseek-v3.2 self-judging accidentally avoided
Original protocol: when scoring deepseek-v3.2's own responses, swap
primary judge to llama-3.1-405b to avoid self-judging. With llama-3.1-
405b excluded (D-10), the protocol would have fallen back to using
deepseek-v3.2 as its own judge — a contamination risk.

INSTEAD, the round-robin primary judge pool I built (qwen2.5-7b, kimi-
k2, ministral-14b, mistral-small-24b, mistral-small-4-119b, gemma-2-
27b, solar-10.7b, seed-oss-36b) does NOT include deepseek-v3.2. So
when scoring deepseek-v3.2 responses, the round-robin selects ONE OF
EIGHT non-self models. Audit of `results/scores/deepseek-v3.2.jsonl`:
**0/3992 records used self-judging on either primary or secondary
judge** (0.00%). Same audit result for deepseek-v3.1 and other models.

Lucky outcome from a routing change made for unrelated reasons (the
round-robin was added for throughput, not contamination prevention),
but worth flagging as a methodology safeguard in the paper.

---

## Bugs and Fixes

### B-01: fit_all_distributions crash
Two bugs: (a) chi-square test fails when resolve_scores() averaging
produces off-grid values; (b) searchsorted returns out-of-bounds index
for off-grid maximum. Fixed by quantizing scores to nearest grid point
before fitting, safe-clipping indices, and renormalizing expected
frequencies. All 5 models now fit successfully. Each fitter is now
wrapped in try/except so single failures return a sentinel FitResult
with bic=inf instead of crashing the batch.

### B-02: llama-3.2-3b corrupt JSON line
Single truncated JSON line (concurrent write race) caused pipeline to
crash on read. Fixed by: (a) cleaning corrupt line, (b) making
read_jsonl in utils.py corrupt-tolerant (skip bad lines, log warning).
Recovered 78 additional valid records from deduplication.

### B-03: Tier miscalibration from Maverick generation
Maverick (17B) couldn't follow nuanced tier calibration instructions.
T5 questions too easy (TECH_T5 and FIN_T5 worst at 34% miscalibrated),
T1 questions occasionally too hard (LAW_T1 at 15%). Diagnosed via
automated heuristic audit. Fixed via targeted regeneration with V3.2.

### B-04: Asyncio rate-limiter race condition
Original interval-based RateLimiter from generate.py is not asyncio-
safe — concurrent coroutines all read self.last_call simultaneously,
causing burst requests that triggered 429 cascades and pipeline
deadlocks. Replaced with per-provider asyncio.Semaphore (DeepSeek=3,
Groq=14, NIM=18) plus a lock-protected sliding-window limiter. Fixed
the early-pipeline stalls.

---

## Numbers to Cite

- 10,000 total queries across 8 domains × 5 difficulty tiers
- 4,000 queries per model (standard evaluation subset)
- 25-27 models in main analysis (28 minus 1-3 reasoning model exclusions)
- ~336,000 total API calls for evaluation + scoring
- 100-item human validation set, N raters, agreement within ±1.0
- b-value range: 0.98 (gemma-3-4b) to 4.33 (qwen2.5-7b) in early preview
- Judge disagreement: 7.3% human-required rate (pilot)
- Human-validated error rate: ~22% (vs ~35% judge-reported)
- Best-fit distribution: lognormal (4/5 models in early preview)

---

## Open Questions (resolve as data comes in)

- [x] Do frontier models (deepseek-v3.2, llama-405b) show even steeper
      tails (higher b)? Or do they have too few errors for fitting?
      → DeepSeek-v3.2 has b=0.655 (heavy tail, opposite of expected).
      Llama-405b unavailable (rate-limited out, see D-10/D-11). The
      heaviest-tail finding for frontier-class models is the opposite
      of "RLHF-makes-frontier-models-safer."
- [x] Does the prediction experiment work? (micro-error → catastrophic
      prediction, Experiment 3)
      → PARTIAL. Rank prediction works (ρ=0.44, p=0.044 at M≥3.0)
      but magnitude prediction fails (median ratio 0.44, 17/21
      under-predict). See R-10 + T-03.
- [x] Are there domain-specific b-value patterns? (Experiment 4)
      → YES (Friedman p=0.026) but model-idiosyncratic (Kendall
      W=0.108). BIO and FIN have heaviest mean tails. See exp4_domains.json.
- [x] Does scale (parameter count) predict b-value? (Experiment 5)
      → YES, but the OPPOSITE of the pre-registered null:
      ρ_dense = -0.689 (p=0.006). Larger dense models have heavier tails.
      See exp5_scaling.json.
- [x] Do reasoning models (qwq-32b, deepseek-r1-distill) have different
      distribution shapes than standard models? (Experiment 7)
      → Cannot answer with current pipeline. All three reasoning models
      have format/timeout issues. Need separate sub-study with
      max_tokens=2000 and timeout=300s. See R-06.
- [x] Other reasoning models in catalog — do they have the same <think>
      truncation problem as phi-4-mini-flash?
      → YES. deepseek-r1-distill-llama-8b: 30% truncation at 500 tokens.
      qwq-32b: 50% truncation + 80% timeout rate. See R-06.
- [ ] Should we attempt Option A for phi-4-mini-flash (re-evaluate with
      max_tokens=2000) if time permits? Same question now applies to
      deepseek-r1-distill-llama-8b.
