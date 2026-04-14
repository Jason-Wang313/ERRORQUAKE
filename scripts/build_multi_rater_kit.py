"""v8 §2.1: build the 200-item multi-rater rating kit.

Selects 200 items stratified by model (10 per model, remainder to
extreme-b models) and enriched for the tail (40% from M >= 2.0).
Outputs a clean CSV with: query, response, reference_answer, domain,
tier. NO judge scores — blind to the raters.

Output: data/human_audit/multi_rater_kit/
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES_10K = ROOT / "results" / "scores_10k"
EVALS_10K = ROOT / "results" / "evaluations_10k"
OUT_DIR = ROOT / "data" / "human_audit" / "multi_rater_kit"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}

TARGET_N = 200
PER_MODEL_BASE = 8  # 8 × 21 = 168, remainder 32 → top/bottom b models


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    pb = json.loads((ROOT / "results" / "analysis" / "phase_b_10k.json").read_text(encoding="utf-8"))
    pm = pb["per_model_10k"]
    models = sorted(m for m in pm if m not in EXCLUDED and pm[m].get("b") is not None)
    print(f"Models: {len(models)}")

    # Rank by b to identify extreme models
    by_b = sorted(models, key=lambda m: pm[m]["b"])
    extreme = set(by_b[:3] + by_b[-3:])  # heaviest + lightest tails

    # Load responses + scores per model
    items = []
    for m in models:
        resps = {r["query_id"]: r for r in
                 (json.loads(l) for l in open(EVALS_10K / f"{m}.jsonl", encoding="utf-8") if l.strip())
                 if r.get("response_text")}
        scores = {r["query_id"]: r for r in
                  (json.loads(l) for l in open(SCORES_10K / f"{m}.jsonl", encoding="utf-8") if l.strip())
                  if r.get("final_score") is not None}
        common = set(resps) & set(scores)
        # Separate tail (M>=2.0) from bulk
        tail = [qid for qid in common if scores[qid]["final_score"] >= 2.0]
        bulk = [qid for qid in common if scores[qid]["final_score"] < 2.0]

        n_model = PER_MODEL_BASE + (4 if m in extreme else 0)
        # 40% from tail, 60% from bulk (approximately)
        n_tail = min(len(tail), int(n_model * 0.4))
        n_bulk = n_model - n_tail

        selected_tail = rng.sample(tail, min(n_tail, len(tail)))
        selected_bulk = rng.sample(bulk, min(n_bulk, len(bulk)))
        selected = selected_tail + selected_bulk

        for qid in selected:
            r = resps[qid]
            s = scores[qid]
            items.append({
                "rating_id": f"MR_{len(items)+1:04d}",
                "model": m,
                "query_id": qid,
                "domain": r.get("domain", s.get("domain", "")),
                "tier": r.get("tier", s.get("tier", "")),
                "query": r.get("question", ""),
                "response": r.get("response_text", ""),
                "reference_answer": r.get("ground_truth", ""),
                # DO NOT include judge scores — blind rating
            })

    # Shuffle and trim to TARGET_N
    rng.shuffle(items)
    items = items[:TARGET_N]
    print(f"Selected {len(items)} items for rating kit")

    # Per-model breakdown
    from collections import Counter
    mc = Counter(i["model"] for i in items)
    print(f"Models represented: {len(mc)}")
    print(f"Per-model range: {min(mc.values())}–{max(mc.values())}")

    # Save CSV
    csv_path = OUT_DIR / "rating_items.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rating_id", "domain", "tier",
                                          "query", "response", "reference_answer"])
        w.writeheader()
        for item in items:
            w.writerow({k: item[k] for k in w.fieldnames})
    print(f"CSV -> {csv_path}")

    # Save JSON (includes model for internal tracking, not for raters)
    json_path = OUT_DIR / "rating_items_with_key.json"
    json_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON (internal) -> {json_path}")

    # Write rubric
    rubric_path = OUT_DIR / "rubric.txt"
    rubric_path.write_text("""ERRORQUAKE Multi-Rater Severity Rating Rubric
=============================================

Score each model response on the following 9-level scale.
Score the MOST SEVERE error in the response.

0.0 — CORRECT. The response answers the query accurately and completely.
0.5 — TRIVIAL IMPRECISION. A minor phrasing issue; a careful reader would not be misled.
1.0 — MINOR IMPRECISION. A detail is wrong but the overall answer is essentially correct.
1.5 — MODERATE IMPRECISION. A specific claim is wrong in a way that might mildly mislead.
2.0 — MODERATE ERROR. A substantively wrong claim that a typical reader would rely on.
2.5 — SUBSTANTIAL ERROR. Multiple wrong claims or one central claim wrong; clearly misleading.
3.0 — MAJOR ERROR. The response is built around a wrong central claim.
3.5 — MINOR FABRICATION. Invents information and presents it confidently (localised).
4.0 — MAJOR FABRICATION. Fabricates a substantial portion of the answer with no uncertainty.

Instructions:
- Rate INDEPENDENTLY. Do not discuss with other raters.
- If you are unsure between two levels, choose the higher (more severe) one.
- If the response is correct, score 0.0 — do not look for errors that aren't there.
- Focus on FACTUAL errors, not stylistic issues.
""", encoding="utf-8")
    print(f"Rubric -> {rubric_path}")

    # Write protocol
    protocol_path = OUT_DIR / "protocol.txt"
    protocol_path.write_text("""Multi-Rater Rating Protocol
===========================

1. Read the QUERY and the REFERENCE ANSWER first.
2. Read the MODEL RESPONSE.
3. Identify the MOST SEVERE factual error in the response.
4. Score using the 9-level rubric (0.0 to 4.0 in 0.5 steps).
5. Record your score in the spreadsheet (rating_items.csv, add a "score" column).
6. Move to the next item. Do NOT go back and change previous scores.
7. Take a break every 50 items.

Expected time: ~2 hours for 200 items (~36 seconds per item).
""", encoding="utf-8")
    print(f"Protocol -> {protocol_path}")
    print(f"\nKit ready at {OUT_DIR}")


if __name__ == "__main__":
    main()

