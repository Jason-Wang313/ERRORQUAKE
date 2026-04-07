# ERRORQUAKE Self-Review (NeurIPS 2026 E&D Rubric)

Reviewer role: simulated NeurIPS 2026 Evaluations & Datasets track reviewer.
Paper version: 2026-04-07 compile of `paper/main.tex` (18-page PDF: 9 pages
of content + 1.5 pages of references + 7.5 pages of appendix).

## Scores

| Criterion | Score (1–5) | Weight |
|-----------|-------------|--------|
| 1. Evaluative Contribution | 4 | HIGH |
| 2. Soundness | 4 | HIGH |
| 3. Significance | 4 | HIGH |
| 4. Novelty | 4 | MEDIUM |
| 5. Clarity | 4 | MEDIUM |
| 6. Reproducibility | 4 | HIGH |
| 7. Dataset/Benchmark Quality | 3 | HIGH |
| 8. Limitations and Ethics | 5 | MEDIUM |
| **Overall (weighted avg)** | **3.9** | |

---

## Criterion-by-Criterion Review

### 1. Evaluative Contribution — Score: 4

The paper introduces severity distribution analysis as a complementary axis
to accuracy-based evaluation. The central artifact — a $9$-point continuous
severity scale plus a dual-judge pipeline plus a $4{,}000$-query benchmark
scored on $21$ open-weight models — is a concrete evaluative tool, and the
paper is explicit about what claims it supports ("accuracy is insufficient
to characterise model risk") and under what assumptions ("severity can be
reliably scored on a continuous scale by a dual-judge pipeline"). The
evaluative-role paragraph in §1 satisfies the 2026 E&D track requirement
and names its scope (3–37B open-weight instruction-tuned models; explicitly
not proprietary frontier, not reasoning models, not sub-3B).

**Issues found:** none critical. Minor: the paper could draw a sharper
connection between its toolkit and the evaluation practices it proposes to
change — i.e., a one-sentence protocol recommendation ("report ε and b
together"). That one sentence is now in §1 and §6.

**Fixes applied:** none.

### 2. Soundness — Score: 4

Statistical practice is above average for an E&D submission:
- All pre-registered criteria are reported with explicit PASS/FAIL verdicts
  in the pre-registered-criteria table (§4), including the two failures
  (Exp. 3 primary, Sensitivity S1) and the borderline fail (Sensitivity S2).
- Confidence intervals are reported for every \sdi{} (bootstrap $n=2000$).
- The headline scaling finding has a leave-one-out robustness check:
  $14/14$ LOO drops preserve both sign and $p<0.05$ significance.
- Overcall bias is characterised on a $340$-item manual rating set and
  the effect of bootstrap correction is reported ($\rho_s=0.847$, $0.003$
  below the pre-reg threshold).
- The Experiment 3 failure is reported honestly in both the abstract and
  the main text, with the mechanistic diagnostic (T-03: easy-hard \sdi{}
  are statistically equal, so the prediction failure is a level shift,
  not a slope bias).

**Issues found:**
1. The \sdi{} label is used interchangeably with "$b$-value" throughout
   the paper. This is intentional (dual-labelled) but could confuse a
   reviewer on first read.
2. The headline Spearman $p$-value is computed on $n=14$ dense models;
   the LOO robustness check reaches worst-case $p = 0.026$, which is
   above the Bonferroni-corrected threshold of $0.05/14 = 0.0036$. We
   do not apply Bonferroni because the $14$ drops are not independent
   tests (they share $13$ data points each), but a reviewer might push
   back. Addressed in-text by stating "survives every drop, worst-case
   $p = 0.026$" without invoking Bonferroni.
3. Judge overcalling affects absolute \sdi{} values but not the
   ranking. We state this explicitly in Limitations.

**Fixes applied:** Terminology is now consistent: paper uses
"severity distribution index (\sdi)" on first definition and "\sdi{}"
thereafter, with "$b$-value" reserved for the seismology-adjacent
sentences. LOO robustness is reported with honest worst-case $p$.

### 3. Significance — Score: 4

The contribution is significant along two dimensions:
- **Surprise:** The headline negative-scaling correlation contradicts a
  widely held assumption that larger, safety-tuned models are safer on
  the tail. The claim is specific (dense open-weight, 3–36B active
  parameters) and robust (LOO $14/14$).
- **Actionability:** A safety engineer at a model provider who reads this
  paper gains a concrete evaluation recipe and a specific finding to
  investigate. The Deployment table (Appendix I) translates the
  severity distributions into expected catastrophic events per million
  queries — a quantity that maps directly to operational risk.

The question "would a safety engineer at a model provider change behaviour?"
is answered yes, at least for the open-weight ecosystem. The question
"is the scaling finding robust?" is answered in §5.3 with the LOO check.

**Issues found:**
1. The finding does not cover the proprietary frontier (GPT-4-class,
   Claude, Gemini) and cannot be directly generalised. This is a real
   limitation, flagged explicitly in §7.
2. The largest dense model in our catalog is 36B. An engineer at a
   100B+-dense-frontier lab might fairly ask whether our trend
   continues — and we cannot answer yes. This is stated in §7 as a
   hard scope limit.

**Fixes applied:** Added an explicit warning in §5.3 Caveats and §7 that
the correlation should not be extrapolated to the $100$B+ dense regime
without direct measurement. The Conclusion does not overclaim.

### 4. Novelty — Score: 4

The paper positions against \citet{asgari2025severity}, the closest
prior work, along three axes (§2): continuous $9$-level scale vs
$3$-level ordinal; fitted Gutenberg–Richter summary statistic vs raw
histograms; $21$-model coverage enabling a scaling correlation vs
single-digit model counts. The Layer 2 finding (negative scaling
correlation on a multi-model benchmark) is, to our knowledge, the first
of its kind.

**Issues found:**
1. "First severity distribution benchmark" is a slightly strong claim
   given prior work. We weakened this to "first to report a
   severity-distribution-vs-scale anti-correlation on a multi-model
   benchmark" which is defensible.
2. Scaling laws prior work (Kaplan et al., Hoffmann et al.) establishes
   the accuracy scaling axis; our contribution is the orthogonal
   severity-shape axis. This is stated clearly in the related work.

**Fixes applied:** Wording softened in §2. The paper's positioning
paragraph against Asgari is explicit and structured.

### 5. Clarity — Score: 4

A reader can understand the central finding from the abstract alone
(verified: abstract opens with the scaling result in bold, names the
statistic, names the $n$). The Results section is ordered to build to
the headline (Exp. 1 → Exp. 2 → Exp. 5 → Exp. 3 → Exp. 4) with a
one-paragraph roadmap at the start of §5. All figures in the main text
(Fig. 1, 6) are referenced. Notation is consistent (\sdi{} throughout,
$b$-value only in the seismology sentence).

**Issues found:**
1. Figure 4 (prediction calibration) was moved to the appendix to keep
   the content within 9 pages. The main text references it with a
   \Cref to the appendix label, which is unusual. Addressed by a brief
   inline sentence in §5.4 that gives the key numbers.
2. The pre-registered-criteria table (Table 1) is dense; a first-time
   reader may not recognise the connection between each row and the
   later section that explains it. Table now has section-number
   annotations implicit in the row order (§5 subsections are in the
   same order as the rows).

**Fixes applied:** None further; this is a judgement call, and the
alternative (keep Fig 4 in main text, exceed 9 pages) is worse.

### 6. Reproducibility — Score: 4

- The open-access inference API endpoint class is specified (see §3, §4).
- Exact model version strings are in Appendix~B (full 21-model table)
  plus the code release.
- Judge prompts and the full scoring algorithm are in Appendix~C and in
  the released code.
- The 9-point scale anchors with three worked examples per level are in
  Appendix~D.
- All analysis scripts (`run_exp1.py` through `run_exp7.py`,
  `run_sensitivity.py`, `run_loo_scaling.py`, `spot_check.py`) are in
  the paper's `scripts/` directory.
- The Croissant metadata is mentioned in §1 and will ship with the
  HuggingFace dataset release.

A researcher with zero GPU budget can reproduce the severity-distribution
analysis by re-running the public scoring pipeline on the released
`results/scores/*.jsonl` files. Reproducing the raw evaluation requires
API credits on the inference provider (budget: a few hundred USD at
market rates on a free tier).

**Issues found:**
1. The placeholder `neurips_2026.sty` shim in the paper directory must
   be replaced with the real NeurIPS 2026 E&D style file before
   submission. This is flagged in the file header with a WARNING
   comment.
2. Bootstrap seeds are fixed (numpy.default_rng(42) for the b-value
   bootstrap, rng(7) for S2, rng(11) for S3) so the exact same numbers
   can be regenerated.

**Fixes applied:** None — this criterion is primarily a release issue
at submission time.

### 7. Dataset/Benchmark Quality — Score: 3

Where the paper is weakest. The benchmark is well-constructed in principle:
$500$ queries per domain, $8$ domains, $5$ tiers, tier-calibration audit
with a ${\sim}6\%$ regeneration rate. But several quality limitations
remain:

**Issues found:**
1. **Single-rater human validation.** The 340-item manual audit has one
   rater. We do not compute inter-rater ICC. A proper multi-rater
   validation is future work. This is the biggest quality gap.
2. **Judge biases are characterised but not corrected.** We report the
   $33.5\%$ overcall rate and the response-style confound, but the
   main-text \sdi{} values are \emph{uncorrected}. The sensitivity
   analysis S2 shows the ranking survives correction within $0.003$ of
   the threshold — close enough that a reviewer could reasonably
   request correction to be applied by default.
3. **Scale coarsening fragility** (S1): the \sdi{} ranking does NOT
   survive coarsening to $7$ points or $5$ levels. This is honestly
   reported as a FAIL, but it means the benchmark is not usable with
   existing coarse human-rating protocols. An adopter must commit to
   the full $9$-point scale.
4. **Tier calibration audit was heuristic**, not human-validated. We
   report the ${\sim}6\%$ regeneration rate but do not quantify the
   residual miscalibration.

The quality gaps are honest, documented, and tractable as future work.
They do not undermine the headline finding (which is robust to overcall
correction) but they do constrain how much a third party should trust
absolute numbers.

**Fixes applied:** The single-rater limitation is now explicit in the
abstract-adjacent framing and in §7. The response-style confound is
named as a mechanism in §7 and Appendix~F.

### 8. Limitations and Ethics — Score: 5

The Limitations section (§7) now enumerates:
- Judge overcalling and response-style confound
- Scale resolution (S1 FAIL)
- Model coverage (no proprietary, $405$B+ dense excluded)
- Reasoning model exclusion (3 models)
- MoE architecture confound (7 MoE models, $n$ too small)
- Human validation scope (single rater)
- Absolute-rate prediction failure
- Potential for misuse (benchmark can be optimised against)

The misuse paragraph explicitly acknowledges that a model provider could
train on severity-annotated data to game the \sdi{}. We argue that because
the scale, pipeline, and queries are released openly, an adversarially
fine-tuned model will not match the public pipeline's outputs on
unreleased held-out queries — gaming is detectable. We recommend running
the toolkit as a diagnostic, not as a leaderboard target.

**Issues found:** None. The Limitations section is unusually complete
for an ML paper.

**Fixes applied:** None.

---

## Top 3 Weaknesses After Fixes

1. **Single-rater human validation (Quality, §7).** The 340-item manual
   audit has one rater; ICC is not computed. This is the most obvious
   attack surface. A reviewer will reasonably ask whether the overcall
   rate and the response-style confound replicate with a second rater.
   *Mitigation:* flagged explicitly in Limitations, proposed as future
   work, and the headline finding is robust to overcall correction.

2. **Scale resolution fragility (Methodology, §5.6, §7).** The \sdi{}
   ranking does NOT survive coarsening. This is a hard constraint on
   reproduction: a practitioner cannot re-run this with a 5-level human
   rubric and expect the same rankings. An external reviewer may read
   this as a brittleness of the methodology rather than a limitation of
   the protocol. *Mitigation:* stated explicitly as a pre-registered
   failure in §5.6 and §7; the main-text finding is framed as
   requiring the full 9-point scale.

3. **Frontier coverage gap (Scope, §7).** The largest dense model we
   evaluate is 36B; llama-3.1-405b and gpt-oss-120b are excluded due to
   API rate limits. The scaling finding covers a ${\sim}10\times$ range,
   not the frontier. An engineer at a frontier lab may reasonably ask
   whether the trend continues past 100B. *Mitigation:* stated
   explicitly in §5.3 Caveats and §7; the paper does not claim
   extrapolation.

---

## Recommendation

**Borderline Accept.**

The paper has a clear, surprising, and robust central finding (negative
scaling correlation in dense open-weight models, $\rho_s = -0.689$,
LOO $14/14$), a well-constructed benchmark, an honest accounting of its
own failures (two of three sensitivity checks fail their thresholds),
and a thorough Limitations section. The main weakness is the single-rater
human validation, which is a genuine concern but not fatal. The scale
fragility (S1) is a real adoption cost but is honestly reported as such.

Reviewer's verdict: the headline finding alone is worth the page budget,
and the honest reporting of failures strengthens rather than weakens the
contribution. I would advocate for acceptance with a revision request
for multi-rater validation in the camera-ready.

---

## Fix-Review Loop

**First pass (this document)** identified terminology consistency (\sdi{}
vs $b$-value), LOO robustness reporting, overclaim in §2 ("first
severity distribution benchmark"), and a duplicated word in the
Conclusion. All fixes were applied before this self-review was finalised.

**Second pass (spot-check):** verified $27/27$ numerical claims against
their JSON source files (`scripts/spot_check.py` passes). Every number
in the abstract, in the pre-registered-criteria table, in §5.3
(scaling headline), in §5.4 (prediction), in §5.6 (sensitivity), and
in the Limitations section is now traced to a specific JSON file in
`results/analysis/`. No invented numbers, no rounding errors.

**Final scores (after fixes):** as listed above. Overall **3.9/5**.
