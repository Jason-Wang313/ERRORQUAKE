"""A1: NIM throughput test.

Measure sustained calls/sec on (a) the dominant judge model
deepseek-v3.2 and (b) a representative target model llama-3.1-8b.
Reports estimated wall-clock time for the full Phase A under
different concurrency levels.
"""
from __future__ import annotations

import asyncio
import random
import time

from env_paths import get_env_path

ENV_PATH = get_env_path()

CANDIDATES = [
    ("deepseek-v3.2 (judge)", "deepseek-ai/deepseek-v3.2"),
    ("llama-3.1-8b (target)", "meta/llama-3.1-8b-instruct"),
    ("gemma-3-27b (target)", "google/gemma-3-27b-it"),
]

PROMPT = "What is the capital of France?"
MAX_TOKENS = 50
N_CALLS = 30


def load_keys() -> list[str]:
    return [
        line.split("=", 1)[1].strip()
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("NVIDIA_NIM_API_KEY") and "=" in line
        and line.split("=", 1)[1].strip()
    ]


async def one_call(client, model: str) -> tuple[bool, str]:
    try:
        r = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            ),
            timeout=30,
        )
        return True, ""
    except Exception as exc:
        return False, str(exc)[:100]


async def measure_serial(client, model: str, n: int) -> dict:
    """Sustained-rate serial test."""
    start = time.time()
    successes = 0
    errors = []
    for _ in range(n):
        ok, err = await one_call(client, model)
        if ok:
            successes += 1
        else:
            errors.append(err)
        await asyncio.sleep(0.05)
    elapsed = time.time() - start
    return {
        "n": n, "successes": successes, "elapsed_s": elapsed,
        "rate_per_sec": successes / max(elapsed, 0.01),
        "first_errors": errors[:3],
    }


async def measure_concurrent(client_pool: list, model: str, n: int, concurrency: int) -> dict:
    """Round-robin concurrent test."""
    sem = asyncio.Semaphore(concurrency)
    successes = [0]
    errors = []
    start = time.time()

    async def worker(idx):
        client = client_pool[idx % len(client_pool)]
        async with sem:
            ok, err = await one_call(client, model)
            if ok:
                successes[0] += 1
            else:
                errors.append(err)

    await asyncio.gather(*[worker(i) for i in range(n)])
    elapsed = time.time() - start
    return {
        "n": n, "concurrency": concurrency, "successes": successes[0],
        "elapsed_s": elapsed, "rate_per_sec": successes[0] / max(elapsed, 0.01),
        "first_errors": errors[:3],
    }


async def main() -> None:
    from openai import AsyncOpenAI

    keys = load_keys()
    print(f"Loaded {len(keys)} NIM keys")

    pool = [
        AsyncOpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1")
        for k in keys
    ]

    print()
    print("=" * 70)
    print("THROUGHPUT TEST (A1)")
    print("=" * 70)

    for label, model_id in CANDIDATES:
        print(f"\n--- {label} ({model_id}) ---")
        # Serial baseline (single key, sequential)
        ser = await measure_serial(pool[0], model_id, n=N_CALLS)
        print(f"  serial    : {ser['successes']}/{ser['n']} in {ser['elapsed_s']:.1f}s "
              f"= {ser['rate_per_sec']:.2f}/s")
        if ser['first_errors']:
            print(f"    errors: {ser['first_errors']}")

        # Concurrent (using key pool)
        for c in (8, 16):
            con = await measure_concurrent(pool, model_id, n=N_CALLS, concurrency=c)
            print(f"  concur={c:>2}: {con['successes']}/{con['n']} in "
                  f"{con['elapsed_s']:.1f}s = {con['rate_per_sec']:.2f}/s")
            if con['first_errors']:
                print(f"    errors: {con['first_errors']}")

    # Phase A budget estimate
    # Generation: 6000 queries x 21 models = 126K calls
    # Scoring: 6000 x 21 x 2 judges = 252K calls
    # Total = 378K
    print()
    print("=" * 70)
    print("PHASE A BUDGET (378K total NIM calls)")
    print("=" * 70)
    for rate in (2.0, 5.0, 10.0, 20.0):
        hours = 378_000 / rate / 3600
        print(f"  at {rate:>5.1f}/s sustained: {hours:>5.1f} hours")


if __name__ == "__main__":
    asyncio.run(main())
