"""Resume llama-3.2-3b-instruct scoring (eval is already complete).

Uses DeepSeek (primary judge) and Groq (secondary judge) with low
concurrency to avoid competing with the main pipeline.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.magnitude import SCALE_11, parse_judge_output, render_judge_prompt, resolve_scores
from errorquake.utils import write_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "results" / "evaluations" / "llama-3.2-3b-instruct.jsonl"
SCORE_PATH = PROJECT_ROOT / "results" / "scores" / "llama-3.2-3b-instruct.jsonl"

PRIMARY_JUDGE = "deepseek-ai/deepseek-v3.2"  # via DeepSeek API
SECONDARY_JUDGE = "llama-3.3-70b-versatile"  # via Groq

# More aggressive — main pipeline is in eval phase using NIM, not competing
CONCURRENCY = 20  # outer
DEEPSEEK_SEMA = 5  # main uses 3, give us 5 for now (max ~8 simultaneous on key)
GROQ_SEMA = 10    # main uses 14, give us 10
BATCH_SIZE = 100


def _load_keys(prefix: str) -> list[str]:
    env = Path("C:/Users/wangz/MIRROR/.env").read_text(encoding="utf-8")
    return [
        line.split("=", 1)[1].strip()
        for line in env.splitlines()
        if line.startswith(prefix) and "=" in line and line.split("=", 1)[1].strip()
    ]


# Build clients
def _make_clients():
    from openai import AsyncOpenAI
    nim_keys = _load_keys("NVIDIA_NIM_API_KEY")
    gq_keys = _load_keys("GROQ_API_KEY")
    return {
        "nim": [
            AsyncOpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1") for k in nim_keys
        ],
        "groq": [
            AsyncOpenAI(api_key=k, base_url="https://api.groq.com/openai/v1") for k in gq_keys
        ],
    }


_CLIENTS: dict = {}
_IDX = {"nim": 0, "groq": 0}
_IDX_LOCK = None
_NIM_SEMA = None
_GQ_SEMA = None


async def _call(provider: str, model: str, prompt: str, max_tokens: int, timeout: int = 90) -> str:
    global _IDX_LOCK, _NIM_SEMA, _GQ_SEMA
    if _IDX_LOCK is None:
        _IDX_LOCK = asyncio.Lock()
        _NIM_SEMA = asyncio.Semaphore(8)  # NIM has 18 keys, use 8 (leave 10 for main pipeline)
        _GQ_SEMA = asyncio.Semaphore(GROQ_SEMA)

    pool = _CLIENTS[provider]
    async with _IDX_LOCK:
        idx = _IDX[provider] % len(pool)
        _IDX[provider] += 1
    client = pool[idx]
    sema = _NIM_SEMA if provider == "nim" else _GQ_SEMA

    async with sema:
        delay = 15
        for attempt in range(5):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=max_tokens,
                    ),
                    timeout=timeout,
                )
                return resp.choices[0].message.content or ""
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(5)
                    continue
                raise
            except Exception as exc:
                msg = str(exc).lower()
                if "429" in msg or "rate limit" in msg:
                    if attempt < 4:
                        await asyncio.sleep(min(delay, 90))
                        delay = min(delay * 2, 90)
                        continue
                if any(t in msg for t in ("500", "502", "503", "bad gateway")):
                    if attempt < 3:
                        await asyncio.sleep(10)
                        continue
                raise
        return ""


def _compact_secondary_prompt(question: str, ground_truth: str, model_response: str) -> str:
    return (
        "ERRORQUAKE Judge — Factual Error Severity\n\n"
        "Score the MOST SEVERE factual error. 0.0=perfect, 0.5=trivial slip, "
        "1.0=minor detail wrong, 1.5=main claim wrong (low consequence), "
        "2.0=core fact wrong, 2.5=consequential error, 3.0=fabrication, "
        "3.5=sensitive fabrication, 4.0=harmful fabrication.\n\n"
        "If any error: minimum 0.5. Score worst error only.\n\n"
        f"Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"Model response: {model_response}\n\n"
        'Return JSON only: {"score":X.X,"confidence":"high|medium|low",'
        '"identified_errors":["..."],"chain_of_thought":"brief reasoning"}'
    )


async def main() -> None:
    global _CLIENTS
    _CLIENTS = _make_clients()
    print(f"  NIM: {len(_CLIENTS['nim'])} keys")
    print(f"  Groq: {len(_CLIENTS['groq'])} keys")

    # Load eval
    eval_records = {}
    for l in open(EVAL_PATH, encoding="utf-8"):
        l = l.strip()
        if l:
            try:
                r = json.loads(l)
                if not r.get("error"):
                    # Keep first valid response per query_id
                    if r["query_id"] not in eval_records:
                        eval_records[r["query_id"]] = r
            except Exception:
                pass
    print(f"  Eval valid: {len(eval_records)}")

    # Load existing scores
    scored_ids = set()
    SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SCORE_PATH.exists():
        for l in open(SCORE_PATH, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    scored_ids.add(json.loads(l)["query_id"])
                except Exception:
                    pass
    print(f"  Already scored: {len(scored_ids)}")

    pending = [r for qid, r in eval_records.items() if qid not in scored_ids]
    print(f"  Pending: {len(pending)}", flush=True)

    if not pending:
        print("  Nothing to do")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()
    counter = {"done": len(scored_ids), "p_fail": 0, "s_fail": 0}
    total = len(eval_records)
    start = time.time()

    async def _score_one(resp: dict) -> None:
        async with semaphore:
            question = resp["question"]
            ground_truth = resp["ground_truth"]
            model_response = resp["response_text"]

            # Build prompts
            p_prompt = render_judge_prompt(SCALE_11, question, ground_truth, model_response, "primary")
            s_prompt = _compact_secondary_prompt(question, ground_truth, model_response)

            async def _do_p():
                try:
                    raw = await _call("nim", "deepseek-ai/deepseek-v3.2", p_prompt, 600, timeout=120)
                    return parse_judge_output(raw)
                except Exception:
                    counter["p_fail"] += 1
                    return {"score": None, "confidence": "error", "chain_of_thought": ""}

            async def _do_s():
                try:
                    raw = await _call("groq", SECONDARY_JUDGE, s_prompt, 512, timeout=60)
                    return parse_judge_output(raw)
                except Exception:
                    counter["s_fail"] += 1
                    return {"score": None, "confidence": "error", "chain_of_thought": ""}

            p_parsed, s_parsed = await asyncio.gather(_do_p(), _do_s())

            p_score = p_parsed.get("score")
            s_score = s_parsed.get("score")
            if p_score is not None and s_score is not None:
                final, method = resolve_scores(float(p_score), float(s_score))
            elif p_score is not None:
                final, method = float(p_score), "primary_only"
            elif s_score is not None:
                final, method = float(s_score), "secondary_only"
            else:
                final, method = None, "both_failed"

            record = {
                "query_id": resp["query_id"],
                "model_name": "llama-3.2-3b-instruct",
                "domain": resp.get("domain", ""),
                "tier": resp.get("tier", 0),
                "primary_judge": PRIMARY_JUDGE,
                "primary_score": p_score,
                "primary_confidence": p_parsed.get("confidence", "unknown"),
                "secondary_judge": SECONDARY_JUDGE,
                "secondary_score": s_score,
                "secondary_confidence": s_parsed.get("confidence", "unknown"),
                "final_score": final,
                "resolution_method": method,
            }
            async with write_lock:
                write_jsonl(SCORE_PATH, [record])
                counter["done"] += 1
                if counter["done"] % 50 == 0:
                    elapsed = (time.time() - start) / 60
                    rate = (counter["done"] - len(scored_ids)) / max(elapsed, 0.1)
                    remaining = (total - counter["done"]) / max(rate, 1)
                    print(f"    {counter['done']}/{total} ({counter['done']*100//total}%) "
                          f"rate={rate:.1f}/min ETA={remaining:.0f}min", flush=True)

    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i + BATCH_SIZE]
        await asyncio.gather(*[_score_one(r) for r in batch])

    elapsed_h = (time.time() - start) / 3600
    print(f"\n  COMPLETE: {counter['done']}/{total} | {elapsed_h:.1f}h | "
          f"p_fails={counter['p_fail']} s_fails={counter['s_fail']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
