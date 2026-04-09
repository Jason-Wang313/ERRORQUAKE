"""Generate responses for new (previously rate-limited) models on the
standard 4k subset. Crash-resistant; can be re-run to resume.

Targets (in priority order):
  1. meta/llama-3.1-405b-instruct  (THE one — extends dense range to 405B)
  2. meta/llama-3.3-70b-instruct   (mid-range dense addition)
  3. openai/gpt-oss-120b           (large MoE)

Output: results/evaluations_v4/{stem}.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

ROOT = Path("C:/projects/errorquake")
QUERIES = ROOT / "data" / "queries" / "standard_subset_4k.jsonl"
OUT_DIR = ROOT / "results" / "evaluations_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENV_PATH = Path("C:/Users/wangz/MIRROR/.env")

MODELS = [
    ("llama-3.1-405b-instruct", "meta/llama-3.1-405b-instruct"),
    ("llama-3.3-70b-instruct", "meta/llama-3.3-70b-instruct"),
    ("gpt-oss-120b", "openai/gpt-oss-120b"),
]

CONCURRENCY = 32
NUM_WORKERS = 48
MAX_TOKENS = 500
TIMEOUT_S = 60


def load_keys() -> list[str]:
    keys = []
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("NVIDIA_NIM_API_KEY") and "=" in line:
            v = line.split("=", 1)[1].strip()
            if v:
                keys.append(v)
    return keys


def load_queries() -> list[dict]:
    out = []
    for line in open(QUERIES, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def build_prompt(q: dict) -> list[dict]:
    return [{"role": "user", "content": q["question"]}]


_clients = []
_idx = [0]


def init_clients(keys: list[str]) -> None:
    from openai import AsyncOpenAI
    global _clients
    _clients = [AsyncOpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1") for k in keys]


def next_client():
    c = _clients[_idx[0] % len(_clients)]
    _idx[0] += 1
    return c


async def call_one(model_id: str, q: dict) -> dict:
    client = next_client()
    delay = 0.5
    for attempt in range(4):
        try:
            r = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_id,
                    messages=build_prompt(q),
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                ),
                timeout=TIMEOUT_S,
            )
            return {"response_text": r.choices[0].message.content or "",
                    "error": None}
        except asyncio.TimeoutError:
            if attempt < 3:
                await asyncio.sleep(delay); delay *= 2; continue
            return {"response_text": "", "error": f"timeout_{TIMEOUT_S}s"}
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                if attempt < 3:
                    await asyncio.sleep(delay); delay *= 2; continue
            if "500" in msg or "502" in msg or "503" in msg:
                if attempt < 3:
                    await asyncio.sleep(delay); delay *= 2; continue
            return {"response_text": "", "error": str(exc)[:200]}
    return {"response_text": "", "error": "max_retries"}


async def run_model(stem: str, model_id: str, items: list[dict]) -> None:
    out_path = OUT_DIR / f"{stem}.jsonl"
    completed = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if not r.get("error"):
                    completed.add(r["query_id"])
            except json.JSONDecodeError:
                pass

    pending = [it for it in items if it["id"] not in completed]
    print(f"  [{stem}] {len(completed)} done / {len(pending)} pending")
    if not pending:
        return

    queue: asyncio.Queue = asyncio.Queue()
    for it in pending:
        queue.put_nowait(it)
    write_q: asyncio.Queue = asyncio.Queue()
    DONE = object()

    sem = asyncio.Semaphore(CONCURRENCY)
    counter = {"done": len(completed), "errors": 0}
    total = len(items)
    start = time.time()

    async def process(it):
        async with sem:
            res = await call_one(model_id, it)
        rec = {
            "query_id": it["id"],
            "model_name": stem,
            "model_id": model_id,
            "domain": it.get("domain"),
            "tier": it.get("tier"),
            "question": it["question"],
            "ground_truth": it.get("ground_truth"),
            "response_text": res["response_text"],
            "error": res["error"],
        }
        await write_q.put(rec)

    async def flusher():
        buf = []
        last_print = 0
        while True:
            try:
                item = await asyncio.wait_for(write_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if buf:
                    with out_path.open("a", encoding="utf-8") as f:
                        for r in buf:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    counter["done"] += len(buf)
                    counter["errors"] += sum(1 for r in buf if r.get("error"))
                    if counter["done"] - last_print >= 100:
                        elapsed = (time.time() - start) / 60
                        rate = (counter["done"] - len(completed)) / max(elapsed, 0.01)
                        eta = (total - counter["done"]) / max(rate, 0.1)
                        print(f"  [{stem}] {counter['done']}/{total} "
                              f"err={counter['errors']} rate={rate:.0f}/min eta={eta:.0f}min",
                              flush=True)
                        last_print = counter["done"]
                    buf = []
                continue
            if item is DONE:
                if buf:
                    with out_path.open("a", encoding="utf-8") as f:
                        for r in buf:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                return
            buf.append(item)
            if len(buf) >= 20:
                with out_path.open("a", encoding="utf-8") as f:
                    for r in buf:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                counter["done"] += len(buf)
                counter["errors"] += sum(1 for r in buf if r.get("error"))
                if counter["done"] - last_print >= 100:
                    elapsed = (time.time() - start) / 60
                    rate = (counter["done"] - len(completed)) / max(elapsed, 0.01)
                    eta = (total - counter["done"]) / max(rate, 0.1)
                    print(f"  [{stem}] {counter['done']}/{total} "
                          f"err={counter['errors']} rate={rate:.0f}/min eta={eta:.0f}min",
                          flush=True)
                    last_print = counter["done"]
                buf = []

    async def worker():
        while True:
            try:
                it = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await process(it)
            except Exception as exc:
                print(f"  [{stem}] worker exc: {exc}")
            finally:
                queue.task_done()

    fl = asyncio.create_task(flusher())
    await asyncio.gather(*[worker() for _ in range(NUM_WORKERS)])
    await write_q.put(DONE)
    await fl
    print(f"  [{stem}] DONE: {counter['done']}/{total}  errors={counter['errors']}  "
          f"runtime={(time.time()-start)/60:.0f}min")


async def main() -> None:
    keys = load_keys()
    init_clients(keys)
    print(f"Initialised {len(keys)} NIM clients")
    items = load_queries()
    print(f"Loaded {len(items)} queries")

    for stem, model_id in MODELS:
        print(f"\n=== {stem} ({model_id}) ===")
        await run_model(stem, model_id, items)


if __name__ == "__main__":
    asyncio.run(main())
