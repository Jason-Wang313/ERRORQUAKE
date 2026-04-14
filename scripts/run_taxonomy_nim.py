"""
Taxonomy Classification via NIM API
=====================================
Classifies 1,000 error responses using a NIM-hosted model.

Uses llama-3.1-405b-instruct as the classifier (strongest NIM model
NOT in the 21-model severity evaluation set).

Fallback: deepseek-v3.2 or qwen3-next-80b if 405B has rate limit issues.

Usage:
  python scripts/run_taxonomy_nim.py [--model MODEL] [--rpm RPM] [--resume]

Reads: results/analysis/oral_upgrade/taxonomy/taxonomy_items_prepared.jsonl
Writes: results/analysis/oral_upgrade/taxonomy/taxonomy_items_classified.jsonl
"""

import asyncio
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from collections import deque

import httpx
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parent.parent
TAXONOMY_DIR = REPO / "results" / "analysis" / "oral_upgrade" / "taxonomy"
INPUT_PATH = TAXONOMY_DIR / "taxonomy_items_prepared.jsonl"
OUTPUT_PATH = TAXONOMY_DIR / "taxonomy_items_classified.jsonl"

NIM_BASE = "https://integrate.api.nvidia.com/v1"

CLASSIFIER_MODELS = {
    "llama-405b": "meta/llama-3.1-405b-instruct",
    "deepseek-v3.2": "deepseek-ai/deepseek-v3.2",
    "qwen3-80b": "qwen/qwen3-next-80b-a3b-instruct",
    "qwq-32b": "qwen/qwq-32b",
}

TAXONOMY_TEXT = """TAXONOMY OF ERROR MECHANISMS:

A_RETRIEVAL: Correct concept retrieved, wrong specific fact
  A1_entity_substitution: Wrong real entity of the same semantic type
  A2_temporal_misattribution: Right event or entity, wrong date/year
  A3_geographic_misattribution: Right event or entity, wrong place
  A4_numerical_distortion: Right entity, wrong number (>15% off)

B_REASONING: Correct facts combined incorrectly
  B1_causal_inversion: Reverses cause and effect
  B2_scope_overgeneralization: Applies specific fact too broadly
  B3_logical_error: Invalid deduction from correct premises

C_GENERATION: Fabricated content stated as fact
  C1_entity_fabrication: Invents nonexistent entity
  C2_citation_fabrication: Invents paper, statistic, or source
  C3_detail_confabulation: Fabricates details about a real topic
  C4_false_precision: Adds fabricated numbers/dates for credibility

D_METACOGNITIVE: Wrong relationship to own knowledge state
  D1_denial_deflection: Claims fact doesn't exist when it does
  D2_overconfident_assertion: States uncertain claim with false certainty

E_AMPLIFICATION: Kernel of truth distorted through elaboration
  E1_partial_truth_inflated: Correct start, wrong elaboration
  E2_analogical_overshoot: Applies pattern from similar domain incorrectly

F_FORMAT: Response structure issues
  F1_incomplete_response: Correct but cuts off before answering
  F2_irrelevant_response: Responds to different question than asked"""


def build_prompt(item):
    """Build classification prompt for one item."""
    return f"""You are classifying the error mechanism in an LLM response that was judged to have severity {item['severity_score']}/4.0.

QUESTION: {item['question'][:500]}

GROUND TRUTH: {item['ground_truth'][:500]}

MODEL RESPONSE: {item['response'][:800]}

{TAXONOMY_TEXT}

Classify the PRIMARY error mechanism. Output ONLY valid JSON:
{{"primary_category": "X_CATEGORY", "primary_subcategory": "X#_name", "confidence": "high/medium/low", "explanation": "one sentence"}}"""


def load_api_keys():
    """Load NIM API keys from MIRROR/.env or environment."""
    keys = []
    env_path = Path.home() / "MIRROR" / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("NVIDIA_NIM_API_KEY") and "=" in line:
                    key = line.split("=", 1)[1].strip()
                    if key:
                        keys.append(key)
    if not keys:
        k = os.environ.get("NVIDIA_API_KEY", "")
        if k:
            keys.append(k)
    return keys


class NIMClassifier:
    def __init__(self, model_id, rpm=35, api_keys=None):
        self.model_id = model_id
        self.rpm = rpm
        self.api_keys = api_keys or load_api_keys()
        if not self.api_keys:
            raise ValueError("No API keys found. Check MIRROR/.env or set NVIDIA_API_KEY")
        print(f"  Loaded {len(self.api_keys)} API keys")
        self.key_idx = 0
        self.timestamps = deque()
        self.client = httpx.AsyncClient(timeout=120)

    def _next_key(self):
        """Round-robin through API keys."""
        key = self.api_keys[self.key_idx % len(self.api_keys)]
        self.key_idx += 1
        return key

    async def rate_limit(self):
        now = time.monotonic()
        # Effective RPM = rpm * n_keys (each key has its own limit)
        effective_rpm = self.rpm * len(self.api_keys)
        while len(self.timestamps) >= effective_rpm:
            oldest = self.timestamps[0]
            if now - oldest < 60:
                await asyncio.sleep(60 - (now - oldest) + 0.1)
                now = time.monotonic()
            else:
                self.timestamps.popleft()
        self.timestamps.append(now)

    async def classify(self, prompt, retries=2):
        await self.rate_limit()
        api_key = self._next_key()
        try:
            resp = await self.client.post(
                f"{NIM_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 300,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return self.parse_response(text)
            elif resp.status_code == 429 and retries > 0:
                await asyncio.sleep(3)
                return await self.classify(prompt, retries=retries - 1)
            else:
                return {"error": f"HTTP {resp.status_code}", "raw": resp.text[:200]}
        except Exception as e:
            return {"error": str(e)}

    def parse_response(self, text):
        """Parse JSON from model response, handling markdown code blocks."""
        # Try direct JSON parse
        text = text.strip()
        # Remove markdown code blocks
        if "```" in text:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r'\{[^{}]*\}', text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"error": "parse_failed", "raw": text[:300]}

    async def close(self):
        await self.client.aclose()


async def run_classification(model_name, rpm, resume):
    model_id = CLASSIFIER_MODELS.get(model_name)
    if not model_id:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(CLASSIFIER_MODELS.keys())}")
        return

    print(f"Using classifier: {model_name} ({model_id})")
    print(f"Rate limit: {rpm} RPM")

    # Load items
    items = []
    with open(INPUT_PATH) as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items)} items")

    # Load existing classifications if resuming
    completed = {}
    if resume and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                rec = json.loads(line)
                key = f"{rec['model']}__{rec['query_id']}"
                completed[key] = rec
        print(f"Resuming: {len(completed)} already classified")

    # Filter to unclassified
    to_classify = []
    for item in items:
        key = f"{item['model']}__{item['query_id']}"
        if key not in completed:
            to_classify.append(item)

    print(f"Remaining: {len(to_classify)}")

    if not to_classify:
        print("All items already classified!")
        return

    classifier = NIMClassifier(model_id, rpm)

    try:
        # Open output file in append mode
        with open(OUTPUT_PATH, "a") as out_f:
            pbar = tqdm(total=len(to_classify), desc="Classifying")
            batch = []

            for item in to_classify:
                prompt = build_prompt(item)
                result = await classifier.classify(prompt)

                # Merge classification into item
                classified = {**item}
                if "error" not in result:
                    classified["primary_category"] = result.get("primary_category", "UNKNOWN")
                    classified["primary_subcategory"] = result.get("primary_subcategory", "UNKNOWN")
                    classified["classification_confidence"] = result.get("confidence", "unknown")
                    classified["classification_explanation"] = result.get("explanation", "")
                else:
                    classified["primary_category"] = "CLASSIFICATION_FAILED"
                    classified["primary_subcategory"] = "CLASSIFICATION_FAILED"
                    classified["classification_error"] = result.get("error", "")
                    classified["classification_raw"] = result.get("raw", "")

                # Use rule-based if LLM failed
                if classified["primary_category"] == "CLASSIFICATION_FAILED" and item.get("rule_category"):
                    classified["primary_category"] = item["rule_category"]
                    classified["primary_subcategory"] = item["rule_subcategory"]
                    classified["classification_confidence"] = "rule_based"

                classified["classifier_model"] = model_name

                out_f.write(json.dumps(classified, ensure_ascii=False) + "\n")
                out_f.flush()
                pbar.update(1)

            pbar.close()

    finally:
        await classifier.close()

    print(f"\nClassification complete. Results at: {OUTPUT_PATH}")
    print(f"Run: python scripts/run_taxonomy_classifier.py --analyze")


def main():
    parser = argparse.ArgumentParser(description="Taxonomy classification via NIM")
    parser.add_argument("--model", default="llama-405b",
                        choices=list(CLASSIFIER_MODELS.keys()),
                        help="NIM model for classification")
    parser.add_argument("--rpm", type=int, default=35, help="Rate limit (requests per minute)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    asyncio.run(run_classification(args.model, args.rpm, args.resume))


if __name__ == "__main__":
    main()
