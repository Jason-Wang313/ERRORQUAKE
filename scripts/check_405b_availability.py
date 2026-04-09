"""Check NIM availability of rate-limited models for Patch 2.

Tests llama-3.1-405b, gpt-oss-120b, minimax-m2.5, llama-3.3-70b
with a single short call per key. Reports availability + first error.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

ENV_PATH = Path("C:/Users/wangz/MIRROR/.env")
OUT = Path("C:/projects/errorquake/results/analysis/availability_check.json")

CANDIDATES = [
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b",
    "minimaxai/minimax-m2.5",
    "meta/llama-3.1-70b-instruct",
]


def load_keys() -> list[str]:
    keys = []
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("NVIDIA_NIM_API_KEY") and "=" in line:
            v = line.split("=", 1)[1].strip()
            if v:
                keys.append(v)
    return keys


async def test_one(client, model: str) -> dict:
    try:
        r = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "say hi"}],
                max_tokens=4,
                temperature=0.0,
            ),
            timeout=20,
        )
        return {"model": model, "ok": True, "response": (r.choices[0].message.content or "")[:30]}
    except Exception as exc:
        return {"model": model, "ok": False, "error": str(exc)[:200]}


async def main() -> None:
    from openai import AsyncOpenAI

    keys = load_keys()
    print(f"Loaded {len(keys)} keys")
    # Try with the first key first; if rate-limited, try a few more
    results = {}
    for model in CANDIDATES:
        print(f"\n--- {model} ---")
        successes = 0
        last_error = None
        for i, k in enumerate(keys[:5]):
            client = AsyncOpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1")
            r = await test_one(client, model)
            if r["ok"]:
                print(f"  key {i}: OK -> {r['response']}")
                successes += 1
                break  # one success is enough
            else:
                short = r["error"][:80]
                print(f"  key {i}: FAIL -> {short}")
                last_error = r["error"]
        results[model] = {
            "available": successes > 0,
            "tested_keys": i + 1,
            "last_error": last_error if successes == 0 else None,
        }

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")
    print()
    print("Summary:")
    for m, r in results.items():
        status = "AVAILABLE" if r["available"] else "rate-limited / unavailable"
        print(f"  {m}: {status}")


if __name__ == "__main__":
    asyncio.run(main())
