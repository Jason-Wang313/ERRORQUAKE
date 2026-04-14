# Severity Mechanism Taxonomy — Pilot Notes (15-item sample)

## Emerging Categories from Score >= 3.0 Responses

### 1. NUMERICAL DISTORTION (5/15 items)
Correct entity type, wrong number. Magnitude off by 1.5x to 100x.
- BIO_T5_0239: Thyroid weight 25.7g vs 18.6g (1.4x off)
- BIO_T5_0207: Odds ratio 28.0 vs 9.72 (2.9x off), **with fabricated CI bounds**
- LAW_T5_0091: Treaty count 1,225 vs 215 (5.7x off)
- LAW_T5_0070: Treaty count 250,000 vs 2,500 (100x off)
- GEO_T5_0095: Population 17M vs 12.7M (1.3x off), **with fabricated "2024 census" source**

Pattern: The model retrieves the correct entity/concept but generates a plausible-sounding but wrong number. Often accompanied by fabricated precision (fake CIs, fake source years).

### 2. ENTITY SUBSTITUTION (3/15)
Wrong entity of the correct semantic type.
- HIST_T4_0250: Battle of Brody vs Battle of Prokhorovka (both are WWII tank battles)
- SCI_T3_0236: "Libration" vs "tidal locking" (both are orbital mechanics terms)
- TECH_T2_0171: Grace Hopper vs Ada Lovelace (both are CS pioneers — debatable)

Pattern: The model retrieves a semantically adjacent entity from the correct category. The substituted entity is always real and contextually plausible.

### 3. CONFIDENT CONFABULATION (4/15)
Entirely fabricated details about a topic, stated with high confidence.
- HIST_T3_0224: Fabricated exploration details about a nonexistent "Rumiantsev Mission"
- LAW_T5_0071: Fabricated "14 Articles" for the Model Penal Code (MPC has sections, not articles)
- LAW_T5_0123: Fabricated ruling structure (claims 5-4 with no partial concurrences)
- GEO_T5_0095 (overlap with numerical): Fabricated "2024 census" source

Pattern: The model generates plausible-sounding but entirely invented facts. Often includes false precision (specific numbers, dates, structural details) that increase apparent credibility.

### 4. DENIAL / DEFLECTION (1/15)
Claims the asked-about fact doesn't exist or isn't relevant.
- LAW_T5_0243: "The specific dollar amount was not a subject of the case" (it was — $1,646)

Pattern: The model avoids answering by denying the premise rather than admitting uncertainty. This is particularly dangerous because it sounds authoritative.

### 5. POSSIBLE OVERCALL (2/15)
Responses that appear correct or debatable but received high severity scores.
- LAW_T5_0095: Gives the correct answer (Korematsu v. United States) — scored 3.0 (possible overcall)
- SCI_T4_0224: Begins a correct explanation of the Lagrangian but appears truncated — may be a formatting issue rather than a factual error

## Observations for Taxonomy Design

1. **Numerical distortion dominates at high severity** — 5/15 items are wrong numbers. This aligns with the "fabricated precision" pattern from the overcall audit.

2. **Confident confabulation is the most dangerous category** — these are responses where a non-expert reader would have no way to detect the error without independent verification.

3. **Entity substitution produces "near-miss" errors** — the substituted entity is always real and semantically close, making detection harder.

4. **Domain matters:** LAW has the most high-severity items (6/15 in this sample, matching the full distribution where LAW has 1,140 items). Legal questions about specific counts, dates, and case details are particularly vulnerable.

5. **Tier 5 (hardest) dominates** — 10/15 items are T5, consistent with the Exp 3 finding that hard queries shift the entire severity distribution upward.

## Proposed Full Taxonomy (6 top-level, ~16 subcategories)

### A. RETRIEVAL ERRORS (correct concept, wrong fact)
- A1. Entity substitution (wrong real entity of same type)
- A2. Temporal misattribution (right event, wrong date/year)
- A3. Geographic misattribution (right event, wrong place)
- A4. Numerical distortion (right entity, wrong number)

### B. REASONING ERRORS (correct facts, wrong inference)
- B1. Causal inversion (reverses cause and effect)
- B2. Scope overgeneralization (applies a specific fact too broadly)
- B3. Logical error (invalid deduction from correct premises)

### C. GENERATION ERRORS (fabricated content)
- C1. Entity fabrication (invents a nonexistent entity)
- C2. Citation fabrication (invents a paper, statistic, or source)
- C3. Detail confabulation (fabricates specific details of a real topic)
- C4. False precision (adds fabricated numbers/dates to increase apparent credibility)

### D. METACOGNITIVE ERRORS (wrong relationship to own knowledge)
- D1. Denial/deflection (claims fact doesn't exist when it does)
- D2. Overconfident assertion (states uncertain claim with false certainty)
- D3. Hedged fabrication (appears cautious but content is wrong)

### E. AMPLIFICATION ERRORS (kernel of truth distorted)
- E1. Partial truth inflated (correct start, wrong elaboration)
- E2. Analogical overshoot (applies pattern from similar domain incorrectly)

### F. TRUNCATION / FORMAT ERRORS
- F1. Incomplete response (correct but cuts off before answering)
- F2. Format compliance failure (answers in wrong format)

## Next Steps
1. Code 100 more items from the severity range 1.0-4.0 to validate and refine categories
2. Check if mechanism distribution differs by model size (the key hypothesis)
3. Build LLM classifier based on the finalized taxonomy
4. Human-validate 200 items
