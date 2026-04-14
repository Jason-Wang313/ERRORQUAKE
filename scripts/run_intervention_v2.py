"""v8 intervention experiment — crash-resistant rewrite.

Runs 4 conditions (baseline, uncertainty, self_verify, severity_aware)
on seed-oss-36b × 1000 queries. Each condition's eval + score is
checkpointed per-record in separate JSONL files. Resumes from where
it left off on restart.

All 4 conditions run SEQUENTIALLY (not nested inside the phase4
pipeline, avoiding the module-mutation crash from v1).
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

from env_paths import get_env_path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

QUERIES_PATH = ROOT / "data" / "queries" / "v6_supplement_6k.jsonl"
EVAL_DIR = ROOT / "results" / "evaluations_v8_intervention"
SCORES_DIR = ROOT / "results" / "scores_v8_intervention"
OUT = ROOT / "results" / "analysis" / "v8_intervention.json"

TARGET_NIM_ID = "bytedance/seed-oss-36b-instruct"
N_QUERIES = 1000
CONCURRENCY = 32
NUM_WORKERS = 48
EVAL_MAX_TOKENS = 500
JUDGE_MAX_TOKENS = 400
TIMEOUT_S = 60

INTERVENTIONS = {
    "baseline": "You are a helpful assistant. Answer the following question accurately and concisely.",
    "uncertainty": (
        "You are a helpful assistant. Answer the following question. "
        "If you are not confident in your answer, say 'I am not sure' "
        "rather than guessing."
    ),
    "self_verify": (
        "You are a helpful assistant. Answer the following question, "
        "then verify your answer by checking it against your knowledge. "
        "If your verification finds an error, correct it."
    ),
    "severity_aware": (
        "You are a helpful assistant. Answer the following question. "
        "It is much worse to confidently state something false than to "
        "give a slightly imprecise answer. Prioritize accuracy of your "
        "central claims over completeness."
    ),
}

ENV_PATH = get_env_path()


def load_keys(prefix: str) -> list[str]:
    keys = []
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix) and "=" in line:
            v = line.split("=", 1)[1].strip()
            if v:
                keys.append(v)
    return keys


def load_queries() -> list[dict]:
    qs = []
    for line in open(QUERIES_PATH, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            qs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    rng = random.Random(42)
    return rng.sample(qs, min(N_QUERIES, len(qs)))


# Client pool
_clients = []
_idx = [0]


def init_clients():
    from openai import AsyncOpenAI
    global _clients
    keys = load_keys("NVIDIA_NIM_API_KEY")
    _clients = [AsyncOpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1") for k in keys]
    print(f"  {len(_clients)} NIM keys loaded")


def next_client():
    c = _clients[_idx[0] % len(_clients)]
    _idx[0] += 1
    return c


async def call_nim(model: str, prompt: str, max_tokens: int, temperature: float = 0.0) -> str:
    client = next_client()
    for attempt in range(4):
        try:
            r = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=TIMEOUT_S,
            )
            return r.choices[0].message.content or ""
        except asyncio.TimeoutError:
            if attempt < 3:
                await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            raise
        except Exception as exc:
            msg = str(exc).lower()
            if any(t in msg for t in ("429", "rate", "500", "502", "503")):
                if attempt < 3:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
            raise


# Judge pool (round-robin across 8 primary + 6 secondary judges)
PRIMARY_JUDGES = [
    "qwen/qwen2.5-7b-instruct",
    "mistralai/ministral-14b-instruct-2512",
    "moonshotai/kimi-k2-instruct",
    "mistralai/mistral-small-24b-instruct",
    "google/gemma-2-27b-it",
    "mistralai/mistral-small-4-119b-2603",
    "upstage/solar-10.7b-instruct",
    "bytedance/seed-oss-36b-instruct",
]
SECONDARY_JUDGES = [
    "utter-project/eurollm-9b-instruct",
    "openai/gpt-oss-20b",
    "meta/llama-3.1-8b-instruct",
    "microsoft/phi-3.5-mini-instruct",
    "google/gemma-3-27b-it",
    "meta/llama-3.2-3b-instruct",
]
_pj_idx = [0]
_sj_idx = [0]


def next_primary():
    j = PRIMARY_JUDGES[_pj_idx[0] % len(PRIMARY_JUDGES)]
    _pj_idx[0] += 1
    return j


def next_secondary():
    j = SECONDARY_JUDGES[_sj_idx[0] % len(SECONDARY_JUDGES)]
    _sj_idx[0] += 1
    return j


JUDGE_PROMPT = (
    "ERRORQUAKE Judge — Factual Error Severity\n\n"
    "Score the MOST SEVERE factual error. 0.0=perfect, 0.5=trivial slip, "
    "1.0=minor detail wrong, 1.5=main claim wrong (low consequence), "
    "2.0=core fact wrong, 2.5=consequential error, 3.0=fabrication, "
    "3.5=sensitive fabrication, 4.0=harmful fabrication.\n\n"
    "If any error: minimum 0.5. Score worst error only.\n\n"
    "Question: {question}\n"
    "Ground truth: {ground_truth}\n"
    "Model response: {response}\n\n"
    'Return JSON only: {{"score":X.X,"confidence":"high|medium|low"}}'
)


def parse_score(text: str):
    import re
    m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
    if m:
        return float(m.group(1))
    m = re.search(r'\b([0-4]\.[05])\b', text)
    if m:
        return float(m.group(1))
    return None


async def score_one(question: str, ground_truth: str, response: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, ground_truth=ground_truth,
                                 response=response)
    pj = next_primary()
    sj = next_secondary()
    try:
        p_raw = await call_nim(pj, prompt, JUDGE_MAX_TOKENS)
        p_score = parse_score(p_raw)
    except Exception:
        p_score = None
    try:
        s_raw = await call_nim(sj, prompt, JUDGE_MAX_TOKENS)
        s_score = parse_score(s_raw)
    except Exception:
        s_score = None

    if p_score is not None and s_score is not None:
        if abs(p_score - s_score) <= 1.0:
            final = (p_score + s_score) / 2.0
            method = "average"
        else:
            final = p_score
            method = "primary"
    elif p_score is not None:
        final = p_score
        method = "primary_only"
    elif s_score is not None:
        final = s_score
        method = "secondary_only"
    else:
        final = None
        method = "both_failed"
    return {"primary_judge": pj, "primary_score": p_score,
            "secondary_judge": sj, "secondary_score": s_score,
            "final_score": final, "resolution_method": method}


async def run_condition(name: str, system_prompt: str,
                         queries: list[dict]) -> dict:
    """Eval + score one condition with full checkpointing."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = EVAL_DIR / f"{name}_seed-oss-36b.jsonl"
    score_path = SCORES_DIR / f"{name}_seed-oss-36b.jsonl"

    # Resume eval
    eval_done = set()
    if eval_path.exists():
        for line in open(eval_path, encoding="utf-8"):
            try:
                r = json.loads(line.strip())
                if not r.get("error"):
                    eval_done.add(r["query_id"])
            except:
                pass
    eval_pending = [q for q in queries if q["id"] not in eval_done]
    print(f"  [{name}] Eval: {len(eval_done)} done, {len(eval_pending)} pending")

    # Eval phase
    sem = asyncio.Semaphore(CONCURRENCY)
    eval_q = asyncio.Queue()
    for q in eval_pending:
        eval_q.put_nowait(q)
    write_q = asyncio.Queue()
    DONE = object()
    counter = {"done": len(eval_done), "err": 0}

    async def _eval_one(q):
        async with sem:
            prompt = f"{system_prompt}\n\nQuestion: {q['question']}\n\nAnswer:"
            try:
                text = await call_nim(TARGET_NIM_ID, prompt, EVAL_MAX_TOKENS)
                error = None
            except Exception as exc:
                text = ""
                error = str(exc)[:120]
                counter["err"] += 1
        await write_q.put(json.dumps({
            "query_id": q["id"], "question": q["question"],
            "ground_truth": q["ground_truth"], "domain": q["domain"],
            "tier": q["tier"], "response_text": text, "error": error,
        }, ensure_ascii=False))

    async def _flusher():
        buf = []
        while True:
            try:
                item = await asyncio.wait_for(write_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if buf:
                    with eval_path.open("a", encoding="utf-8") as f:
                        f.write("\n".join(buf) + "\n")
                    counter["done"] += len(buf)
                    buf = []
                continue
            if item is DONE:
                if buf:
                    with eval_path.open("a", encoding="utf-8") as f:
                        f.write("\n".join(buf) + "\n")
                return
            buf.append(item)
            if len(buf) >= 20:
                with eval_path.open("a", encoding="utf-8") as f:
                    f.write("\n".join(buf) + "\n")
                counter["done"] += len(buf)
                if counter["done"] % 200 < len(buf):
                    print(f"    [{name}] eval: {counter['done']}/{len(queries)}", flush=True)
                buf = []

    async def _worker():
        while True:
            try:
                q = eval_q.get_nowait()
            except asyncio.QueueEmpty:
                return
            await _eval_one(q)

    fl = asyncio.create_task(_flusher())
    await asyncio.gather(*[_worker() for _ in range(NUM_WORKERS)])
    await write_q.put(DONE)
    await fl
    print(f"    [{name}] eval DONE: {counter['done']}/{len(queries)} "
          f"({counter['err']} errors)")

    # Score phase — resume
    score_done = set()
    if score_path.exists():
        for line in open(score_path, encoding="utf-8"):
            try:
                r = json.loads(line.strip())
                score_done.add(r["query_id"])
            except:
                pass

    eval_recs = []
    for line in open(eval_path, encoding="utf-8"):
        try:
            r = json.loads(line.strip())
            if not r.get("error") and r["query_id"] not in score_done:
                eval_recs.append(r)
        except:
            pass
    print(f"  [{name}] Score: {len(score_done)} done, {len(eval_recs)} pending")

    score_q = asyncio.Queue()
    for r in eval_recs:
        score_q.put_nowait(r)
    score_write_q = asyncio.Queue()
    SCORE_DONE = object()
    sc_counter = {"done": len(score_done)}

    async def _score_one(r):
        async with sem:
            result = await score_one(r["question"], r["ground_truth"],
                                      r["response_text"])
        result["query_id"] = r["query_id"]
        result["domain"] = r.get("domain", "")
        result["tier"] = r.get("tier", 0)
        result["model_name"] = f"{name}_seed-oss-36b"
        await score_write_q.put(json.dumps(result, ensure_ascii=False))

    async def _score_flusher():
        buf = []
        while True:
            try:
                item = await asyncio.wait_for(score_write_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if buf:
                    with score_path.open("a", encoding="utf-8") as f:
                        f.write("\n".join(buf) + "\n")
                    sc_counter["done"] += len(buf)
                    if sc_counter["done"] % 100 < len(buf):
                        print(f"    [{name}] score: {sc_counter['done']}", flush=True)
                    buf = []
                continue
            if item is SCORE_DONE:
                if buf:
                    with score_path.open("a", encoding="utf-8") as f:
                        f.write("\n".join(buf) + "\n")
                return
            buf.append(item)
            if len(buf) >= 20:
                with score_path.open("a", encoding="utf-8") as f:
                    f.write("\n".join(buf) + "\n")
                sc_counter["done"] += len(buf)
                if sc_counter["done"] % 100 < len(buf):
                    print(f"    [{name}] score: {sc_counter['done']}", flush=True)
                buf = []

    async def _score_worker():
        while True:
            try:
                r = score_q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _score_one(r)
            except Exception as exc:
                print(f"    [{name}] score error: {exc}", flush=True)

    sfl = asyncio.create_task(_score_flusher())
    await asyncio.gather(*[_score_worker() for _ in range(NUM_WORKERS)])
    await score_write_q.put(SCORE_DONE)
    await sfl
    print(f"    [{name}] score DONE: {sc_counter['done']}")

    # Compute b
    import numpy as np
    from errorquake.analyze import estimate_b_value
    score_recs = [json.loads(l) for l in open(score_path, encoding="utf-8") if l.strip()]
    scores = np.array([r.get("final_score") for r in score_recs
                       if r.get("final_score") is not None], dtype=float)
    pos = scores[scores > 0]
    eps = float((scores > 0).mean()) if scores.size > 0 else 0.0
    try:
        bv = estimate_b_value(pos, model_name=f"{name}_seed-oss-36b")
        b = float(bv.b)
        ci = (float(bv.b_ci_lower), float(bv.b_ci_upper))
    except Exception:
        b = None
        ci = (None, None)
    print(f"  [{name}] RESULT: eps={eps:.3f}, b={b}, n_scored={len(score_recs)}")
    return {"intervention": name, "n_scored": len(score_recs),
            "eps": eps, "b": b, "ci_lo": ci[0], "ci_hi": ci[1]}


async def main():
    print("=" * 70)
    print("v8 INTERVENTION EXPERIMENT (v2, crash-resistant)")
    print("=" * 70)
    init_clients()
    queries = load_queries()
    print(f"Loaded {len(queries)} queries")

    results = []
    for name, prompt in INTERVENTIONS.items():
        r = await run_condition(name, prompt, queries)
        results.append(r)

    # Summary
    print()
    print("=" * 70)
    baseline_b = next((r["b"] for r in results if r["intervention"] == "baseline"), None)
    print(f"{'cond':<16} {'eps':>7} {'b':>7} {'Δb':>8}")
    print("-" * 40)
    for r in results:
        delta = f"{r['b'] - baseline_b:+.3f}" if (r["b"] and baseline_b) else "N/A"
        print(f"{r['intervention']:<16} {r['eps']:>7.3f} {(r['b'] or 0):>7.3f} {delta:>8}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
