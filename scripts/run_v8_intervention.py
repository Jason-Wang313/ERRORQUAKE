"""v8 jawdrop §5: intervention experiment.

Test 3 prompt-engineering interventions on seed-oss-36b (heaviest tail
at 10K) over 1000 queries. Does prompting shift the severity
distribution?

Interventions:
  A — uncertainty prompting ("say I'm not sure")
  B — self-verification ("verify then correct")
  C — severity-aware ("prioritize accuracy of central claims")

Output:
  results/evaluations_v8_intervention/{intervention}_{model}.jsonl
  results/scores_v8_intervention/{intervention}_{model}.jsonl
  results/analysis/v8_intervention.json
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

QUERIES = ROOT / "data" / "queries" / "v6_supplement_6k.jsonl"
EVAL_DIR = ROOT / "results" / "evaluations_v8_intervention"
SCORES_DIR = ROOT / "results" / "scores_v8_intervention"
OUT = ROOT / "results" / "analysis" / "v8_intervention.json"

TARGET_MODEL = "seed-oss-36b"
TARGET_NIM_ID = "bytedance/seed-oss-36b-instruct"
N_QUERIES = 1000

INTERVENTIONS = {
    "baseline": "",
    "uncertainty": (
        "Answer the following question. If you are not confident in your "
        "answer, say 'I am not sure' rather than guessing.\n\n"
    ),
    "self_verify": (
        "Answer the following question, then verify your answer by checking "
        "it against your knowledge. If your verification finds an error, "
        "correct it.\n\n"
    ),
    "severity_aware": (
        "Answer the following question. It is much worse to confidently "
        "state something false than to give a slightly imprecise answer. "
        "Prioritize accuracy of your central claims over completeness.\n\n"
    ),
}


def load_queries(n: int) -> list[dict]:
    qs = []
    for line in open(QUERIES, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            qs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    # Deterministic sample
    rng = random.Random(42)
    return rng.sample(qs, min(n, len(qs)))


async def run_intervention(intervention_name: str, prefix: str,
                           queries: list[dict]) -> None:
    """Evaluate + score one intervention on one model."""
    import run_phase4 as p4

    # Override paths
    p4.EVAL_DIR = EVAL_DIR
    p4.SCORES_DIR = SCORES_DIR
    p4.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    p4.SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the system prompt to include intervention prefix
    original_prompt = p4.SYSTEM_PROMPT
    p4.SYSTEM_PROMPT = prefix + original_prompt if prefix else original_prompt

    # Create a fake ModelConfig
    from errorquake.evaluate import ModelConfig
    model = ModelConfig(
        name=f"{intervention_name}_{TARGET_MODEL}",
        provider="nim",
        model_id=TARGET_NIM_ID,
        api_key_env="NVIDIA_API_KEY",
    )

    print(f"\n  [{intervention_name}] Starting eval+score on {len(queries)} queries...")
    p4._init_all_clients()

    queries_dicts = [{"id": q["id"], "question": q["question"],
                      "ground_truth": q["ground_truth"],
                      "domain": q["domain"], "tier": q["tier"]}
                     for q in queries]
    queries_by_id = {q["id"]: q for q in queries_dicts}

    eval_path, eval_stats = await p4._evaluate_model(model, queries_dicts)
    score_path, score_stats = await p4._score_model(
        model.name, eval_path, queries_by_id)

    # Compute b
    from errorquake.analyze import estimate_b_value
    import numpy as np
    recs = [json.loads(l) for l in open(score_path, encoding="utf-8") if l.strip()]
    scores = np.array([r.get("final_score") for r in recs
                       if r.get("final_score") is not None], dtype=float)
    pos = scores[scores > 0]
    eps = float((scores > 0).mean())
    try:
        bv = estimate_b_value(pos, model_name=f"{intervention_name}_{TARGET_MODEL}")
        b = float(bv.b)
        ci = (float(bv.b_ci_lower), float(bv.b_ci_upper))
    except Exception:
        b = None
        ci = (None, None)

    print(f"  [{intervention_name}] DONE: eps={eps:.3f}, b={b:.3f if b else 'N/A'}, "
          f"n_scored={len(recs)}")

    # Restore
    p4.SYSTEM_PROMPT = original_prompt

    return {
        "intervention": intervention_name,
        "n_scored": len(recs),
        "eps": eps,
        "b": b,
        "ci_lo": ci[0],
        "ci_hi": ci[1],
        "n_pos": int(pos.size),
    }


async def main() -> None:
    print("=" * 70)
    print("v8 INTERVENTION EXPERIMENT")
    print("=" * 70)

    queries = load_queries(N_QUERIES)
    print(f"Loaded {len(queries)} queries")

    results = []
    for name, prefix in INTERVENTIONS.items():
        r = await run_intervention(name, prefix, queries)
        results.append(r)

    # Summary
    print()
    print("=" * 70)
    print("INTERVENTION RESULTS")
    print("=" * 70)
    print(f"{'intervention':<16} {'eps':>7} {'b':>7} {'95% CI':>20} {'Δb vs baseline':>15}")
    baseline_b = next((r["b"] for r in results if r["intervention"] == "baseline"), None)
    for r in results:
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r["ci_lo"] is not None else "N/A"
        delta = f"{r['b'] - baseline_b:+.3f}" if (r["b"] and baseline_b) else "N/A"
        print(f"{r['intervention']:<16} {r['eps']:>7.3f} {(r['b'] or 0):>7.3f} {ci:>20} {delta:>15}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
