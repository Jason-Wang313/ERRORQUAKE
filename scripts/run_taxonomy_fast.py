"""
Fast concurrent taxonomy classification with crash resistance.
Each API key runs independently at its own rate limit.
Every item is checkpointed immediately.
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from collections import deque

import httpx

REPO = Path(__file__).resolve().parent.parent
TAXONOMY_DIR = REPO / "results" / "analysis" / "oral_upgrade" / "taxonomy"
INPUT_PATH = TAXONOMY_DIR / "taxonomy_items_prepared.jsonl"
OUTPUT_PATH = TAXONOMY_DIR / "taxonomy_items_classified.jsonl"
NIM_BASE = "https://integrate.api.nvidia.com/v1"
MODEL_ID = "deepseek-ai/deepseek-v3.2"
RPM_PER_KEY = 10

TAXONOMY_TEXT = """TAXONOMY:
A_RETRIEVAL: Wrong specific fact (A1_entity_substitution, A2_temporal, A3_geographic, A4_numerical_distortion)
B_REASONING: Wrong inference (B1_causal_inversion, B2_scope_overgeneralization, B3_logical_error)
C_GENERATION: Fabricated content (C1_entity_fabrication, C2_citation_fabrication, C3_detail_confabulation, C4_false_precision)
D_METACOGNITIVE: Wrong knowledge state (D1_denial_deflection, D2_overconfident_assertion)
E_AMPLIFICATION: Truth distorted (E1_partial_truth_inflated, E2_analogical_overshoot)
F_FORMAT: Structure issues (F1_incomplete_response, F2_irrelevant_response)"""


def load_keys():
    keys = []
    env_path = Path.home() / "MIRROR" / ".env"
    with open(env_path) as f:
        for line in f:
            if line.startswith("NVIDIA_NIM_API_KEY") and "=" in line:
                k = line.split("=", 1)[1].strip()
                if k: keys.append(k)
    return keys


def build_prompt(item):
    return f"""Classify the error in this LLM response (severity {item['severity_score']}/4.0).

Q: {item['question'][:400]}
GT: {item['ground_truth'][:400]}
Response: {item['response'][:600]}

{TAXONOMY_TEXT}

Output ONLY JSON: {{"primary_category": "X", "primary_subcategory": "X#_name", "confidence": "high/medium/low", "explanation": "..."}}"""


def parse(text):
    text = text.strip()
    if "```" in text:
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m: text = m.group(1).strip()
    try:
        return json.loads(text)
    except:
        m = re.search(r'\{[^{}]*\}', text)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None


def rule_classify(item):
    """Fallback rule-based classification."""
    s = item["severity_score"]
    q = item.get("question", "").lower()
    gt = item.get("ground_truth", "")
    resp = item.get("response", "")

    if not resp or len(resp.strip()) < 20:
        return "F_FORMAT", "F1_incomplete_response"

    gt_nums = re.findall(r'[-+]?\d*\.?\d+', gt or "")
    resp_nums = re.findall(r'[-+]?\d*\.?\d+', resp or "")
    if gt_nums and resp_nums:
        try:
            primary = float(gt_nums[0])
            if primary != 0:
                ratios = [abs(float(rn) - primary) / abs(primary) for rn in resp_nums[:5]]
                if ratios and min(ratios) > 0.15:
                    return "A_RETRIEVAL", "A4_numerical_distortion"
        except: pass

    denial = ["not a subject", "no historical record", "there is no", "does not exist"]
    if any(d in resp.lower() for d in denial) and s >= 2.0:
        return "D_METACOGNITIVE", "D1_denial_deflection"

    if s >= 3.0:
        return "C_GENERATION", "C3_detail_confabulation"
    if s >= 2.0:
        return "E_AMPLIFICATION", "E1_partial_truth_inflated"
    return "A_RETRIEVAL", "A1_entity_substitution"


async def worker(worker_id, api_key, queue, output_lock, output_file, stats):
    """One worker per API key, runs at its own rate limit."""
    client = httpx.AsyncClient(timeout=120)
    timestamps = deque()

    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        # Rate limit for this key
        now = time.monotonic()
        while len(timestamps) >= RPM_PER_KEY:
            oldest = timestamps[0]
            if now - oldest < 60:
                await asyncio.sleep(60 - (now - oldest) + 0.5)
                now = time.monotonic()
            else:
                timestamps.popleft()
        timestamps.append(now)

        prompt = build_prompt(item)
        classified = {**item}
        success = False

        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{NIM_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}],
                          "temperature": 0, "max_tokens": 300},
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    result = parse(text)
                    if result and "primary_category" in result:
                        classified["primary_category"] = result["primary_category"]
                        classified["primary_subcategory"] = result.get("primary_subcategory", "")
                        classified["classification_confidence"] = result.get("confidence", "")
                        classified["classification_explanation"] = result.get("explanation", "")
                        classified["classifier_model"] = "deepseek-v3.2"
                        success = True
                        break
                elif resp.status_code == 429:
                    await asyncio.sleep(3 * (attempt + 1))
                elif resp.status_code == 403:
                    await asyncio.sleep(5)
                else:
                    break
            except Exception:
                await asyncio.sleep(2)

        if not success:
            cat, sub = rule_classify(item)
            classified["primary_category"] = cat
            classified["primary_subcategory"] = sub
            classified["classification_confidence"] = "rule_fallback"
            classified["classifier_model"] = "rule_based"

        # Checkpoint immediately
        async with output_lock:
            output_file.write(json.dumps(classified, ensure_ascii=False) + "\n")
            output_file.flush()

        stats["done"] += 1
        if stats["done"] % 25 == 0:
            print(f"  Progress: {stats['done']}/{stats['total']} ({stats['done']/stats['total']*100:.0f}%)")

    await client.aclose()


async def main():
    keys = load_keys()
    print(f"Loaded {len(keys)} API keys")
    print(f"Model: {MODEL_ID}")
    print(f"RPM per key: {RPM_PER_KEY}, effective: {RPM_PER_KEY * len(keys)}")

    # Load items
    items = []
    with open(INPUT_PATH) as f:
        for line in f:
            items.append(json.loads(line))

    # Load completed
    completed_keys = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                it = json.loads(line)
                completed_keys.add(f"{it['model']}__{it['query_id']}")

    remaining = [it for it in items if f"{it['model']}__{it['query_id']}" not in completed_keys]
    print(f"Total: {len(items)} | Done: {len(completed_keys)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All done!")
        return

    # Build queue
    queue = asyncio.Queue()
    for it in remaining:
        queue.put_nowait(it)

    stats = {"done": 0, "total": len(remaining)}
    output_lock = asyncio.Lock()

    # Open output in append mode
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        # Launch one worker per key
        workers = [
            asyncio.create_task(worker(i, key, queue, output_lock, out_f, stats))
            for i, key in enumerate(keys)
        ]
        await asyncio.gather(*workers)

    print(f"\nDone! Total classified: {len(completed_keys) + stats['done']}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
