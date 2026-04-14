"""Check reasoning models for <think> tag truncation problems."""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from env_paths import load_keys_from_env_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.evaluate import ALL_MODELS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "results" / "evaluations"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
SUBSET_PATH = PROJECT_ROOT / "data" / "queries" / "standard_subset_4k.jsonl"

REASONING_KEYWORDS = ["reasoning", "r1", "qwq", "think"]
MAX_TOKENS = 500  # same as main eval
SAMPLE_SIZE = 50  # for already-evaluated models
SMOKE_SIZE = 10   # for not-yet-evaluated models


def _load_nim_keys() -> list[str]:
    return load_keys_from_env_file("NVIDIA_NIM_API_KEY")


def is_reasoning_model(model) -> bool:
    name = model.name.lower() + " " + model.model_id.lower()
    return any(kw in name for kw in REASONING_KEYWORDS)


def analyze_response(text: str) -> dict:
    """Check if response has reasoning tags / is truncated."""
    has_think_open = "<think>" in text or "<Think>" in text or "<THINK>" in text
    has_think_close = "</think>" in text or "</Think>" in text or "</THINK>" in text
    has_reasoning_tag = "<reasoning>" in text.lower()

    # Truncated if long, no closing tag, no terminating punctuation in last 50 chars
    last_50 = text[-50:].rstrip()
    has_terminator = bool(last_50) and last_50[-1] in ".!?\"」'»)]"
    is_truncated = (
        len(text) > 1500
        and not has_terminator
        and not has_think_close
    )

    # Has answer after </think>?
    has_final_answer = False
    if has_think_close:
        after = text.split("</think>")[-1].strip() if "</think>" in text else ""
        has_final_answer = len(after) > 5

    return {
        "has_think_tag": has_think_open or has_reasoning_tag,
        "has_closing_tag": has_think_close,
        "has_final_answer": has_final_answer,
        "is_truncated": is_truncated,
        "length": len(text),
    }


def analyze_existing(model_name: str) -> dict:
    """Sample 50 responses from existing eval and analyze."""
    path = EVAL_DIR / f"{model_name}.jsonl"
    records = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                if not r.get("error") and r.get("response_text"):
                    records.append(r)
            except json.JSONDecodeError:
                pass

    if not records:
        return {"error": "no valid responses"}

    rng = random.Random(42)
    sample = rng.sample(records, min(SAMPLE_SIZE, len(records)))

    n = len(sample)
    n_think = sum(1 for r in sample if analyze_response(r["response_text"])["has_think_tag"])
    n_closed = sum(1 for r in sample if analyze_response(r["response_text"])["has_closing_tag"])
    n_truncated = sum(1 for r in sample if analyze_response(r["response_text"])["is_truncated"])
    n_final = sum(1 for r in sample if analyze_response(r["response_text"])["has_final_answer"])
    avg_len = sum(len(r["response_text"]) for r in sample) // n

    # Also check ALL records for global rates
    n_total = len(records)
    n_total_think = sum(1 for r in records if analyze_response(r["response_text"])["has_think_tag"])
    n_total_closed = sum(1 for r in records if analyze_response(r["response_text"])["has_closing_tag"])
    n_total_truncated = sum(1 for r in records if analyze_response(r["response_text"])["is_truncated"])

    return {
        "source": "existing_eval",
        "sample_size": n,
        "total_responses": n_total,
        "sample_think_tag_rate": round(n_think / n * 100, 1),
        "sample_closed_tag_rate": round(n_closed / n * 100, 1),
        "sample_truncation_rate": round(n_truncated / n * 100, 1),
        "sample_final_answer_rate": round(n_final / n * 100, 1),
        "sample_avg_length": avg_len,
        "all_think_tag_rate": round(n_total_think / n_total * 100, 1),
        "all_closed_tag_rate": round(n_total_closed / n_total * 100, 1),
        "all_truncation_rate": round(n_total_truncated / n_total * 100, 1),
    }


async def smoke_test(model, queries: list[dict]) -> dict:
    """Run a 10-query smoke test on a model and analyze responses."""
    from openai import AsyncOpenAI

    keys = _load_nim_keys()
    if not keys:
        return {"error": "no NIM keys"}

    client = AsyncOpenAI(api_key=keys[0], base_url="https://integrate.api.nvidia.com/v1")
    rng = random.Random(42)
    sample_queries = rng.sample(queries, SMOKE_SIZE)

    responses = []
    for q in sample_queries:
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model.model_id,
                    messages=[{
                        "role": "user",
                        "content": (
                            "You are a helpful assistant. Answer the following question "
                            "accurately and concisely.\n\n"
                            f"Question: {q['question']}\n\nAnswer:"
                        ),
                    }],
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                ),
                timeout=120,
            )
            text = resp.choices[0].message.content or ""
            responses.append({"query_id": q["id"], "response_text": text})
        except Exception as exc:
            responses.append({"query_id": q["id"], "response_text": "", "error": str(exc)[:120]})
        # Small delay between calls
        await asyncio.sleep(1)

    valid = [r for r in responses if not r.get("error") and r.get("response_text")]
    if not valid:
        return {"source": "smoke_test", "error": "all calls failed",
                "errors": [r.get("error") for r in responses[:3]]}

    n = len(valid)
    n_think = sum(1 for r in valid if analyze_response(r["response_text"])["has_think_tag"])
    n_closed = sum(1 for r in valid if analyze_response(r["response_text"])["has_closing_tag"])
    n_truncated = sum(1 for r in valid if analyze_response(r["response_text"])["is_truncated"])
    n_final = sum(1 for r in valid if analyze_response(r["response_text"])["has_final_answer"])
    avg_len = sum(len(r["response_text"]) for r in valid) // n

    # Save sample responses for inspection
    samples_excerpt = [
        {
            "query_id": r["query_id"],
            "response_excerpt": r["response_text"][:300],
            "length": len(r["response_text"]),
        }
        for r in valid[:3]
    ]

    return {
        "source": "smoke_test",
        "smoke_size": n,
        "successful_calls": n,
        "failed_calls": SMOKE_SIZE - n,
        "think_tag_rate": round(n_think / n * 100, 1),
        "closed_tag_rate": round(n_closed / n * 100, 1),
        "truncation_rate": round(n_truncated / n * 100, 1),
        "final_answer_rate": round(n_final / n * 100, 1),
        "avg_length": avg_len,
        "sample_excerpts": samples_excerpt,
    }


def make_recommendation(stats: dict) -> str:
    """Determine recommendation from analysis stats."""
    if "error" in stats:
        return "needs_investigation"

    # Pick the right field names depending on source
    if stats.get("source") == "existing_eval":
        think_rate = stats.get("all_think_tag_rate", 0)
        closed_rate = stats.get("all_closed_tag_rate", 0)
        trunc_rate = stats.get("all_truncation_rate", 0)
    else:
        think_rate = stats.get("think_tag_rate", 0)
        closed_rate = stats.get("closed_tag_rate", 0)
        trunc_rate = stats.get("truncation_rate", 0)

    # No reasoning tags → ok
    if think_rate < 10:
        return "ok"

    # Has tags AND most close them → ok (model can finish reasoning)
    if think_rate > 50 and closed_rate > 60:
        return "ok"

    # Has tags but most don't close OR high truncation → problem
    if think_rate > 50 and (closed_rate < 30 or trunc_rate > 50):
        return "exclude_or_higher_max_tokens"

    return "needs_investigation"


async def main():
    print("=" * 60)
    print("REASONING MODEL TRUNCATION CHECK")
    print("=" * 60)

    # Identify reasoning models
    reasoning_models = [m for m in ALL_MODELS if is_reasoning_model(m)]
    print(f"\nIdentified {len(reasoning_models)} reasoning models:")
    for m in reasoning_models:
        print(f"  {m.name}  ->  {m.model_id}")

    # Load queries for smoke tests
    queries = []
    for line in open(SUBSET_PATH, encoding="utf-8"):
        line = line.strip()
        if line:
            queries.append(json.loads(line))
    print(f"\nLoaded {len(queries)} subset queries for smoke tests")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {},
    }

    for model in reasoning_models:
        print(f"\n--- {model.name} ---")
        eval_path = EVAL_DIR / f"{model.name}.jsonl"

        if eval_path.exists():
            print("  Analyzing existing eval responses...")
            stats = analyze_existing(model.name)
        else:
            print("  Running 10-query smoke test...")
            stats = await smoke_test(model, queries)

        rec = make_recommendation(stats)
        stats["recommendation"] = rec
        report["models"][model.name] = stats

        print(f"  Recommendation: {rec}")
        for k, v in stats.items():
            if k not in ("sample_excerpts", "recommendation"):
                print(f"    {k}: {v}")

        if "sample_excerpts" in stats:
            print(f"  Sample excerpts:")
            for ex in stats["sample_excerpts"]:
                print(f"    {ex['query_id']} ({ex['length']} chars):")
                print(f"      {ex['response_excerpt'][:200]}")

    # Save report
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS_DIR / "reasoning_model_check.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {out}")

    # Summary
    print("\n=== SUMMARY ===")
    for name, stats in report["models"].items():
        print(f"  {name:35s} -> {stats['recommendation']}")


if __name__ == "__main__":
    asyncio.run(main())
