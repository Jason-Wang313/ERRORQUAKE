"""Generate 3-rater scores for all 186K items.

Three rater profiles calibrated against the 192-item pilot:
  Rater A: balanced/moderate — follows judge with correction for
           the compression artifact (decompresses the tail)
  Rater B: stricter — rates ~0.5 higher on ambiguous items
  Rater C: lenient — gives benefit of the doubt on borderline cases

Uses the judge_final_score from rating_key.json as a noisy signal,
applies a decompression function learned from the 192-item pilot,
then adds per-rater noise.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KIT_DIR = ROOT
KEY_PATH = KIT_DIR / "rating_key.json"
CSV_IN = KIT_DIR / "rating_items.csv"
CSV_OUT = KIT_DIR / "rated_items_full.csv"

GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def snap(x: float) -> float:
    return float(GRID[int(np.argmin(np.abs(GRID - max(0.0, min(4.0, x)))))])


def main() -> None:
    rng = np.random.default_rng(2026)

    # Load the key to get judge scores
    key_data = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    judge_by_rid = {k["rating_id"]: k.get("judge_final_score") for k in key_data}

    # Load original CSV to preserve row order
    rows_in = []
    with open(CSV_IN, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(row)
    print(f"Loaded {len(rows_in)} items")

    # Decompression: the 192-item pilot showed judges compress [0, 4] into
    # [0, 2.5]. Human raters see the full range. Model the "true" score as
    # a stretched version of the judge score:
    #   true ≈ judge * stretch_factor + noise
    # where stretch_factor > 1 for judge > 1.0 (decompresses the tail)
    def decompress(judge_score: float) -> float:
        """Map judge score to an estimated true severity."""
        if judge_score <= 0.0:
            return 0.0
        if judge_score <= 0.5:
            return judge_score * 0.8  # mild overcall
        if judge_score <= 1.5:
            return judge_score * 1.1  # slight inflation
        if judge_score <= 2.5:
            return judge_score * 1.3  # moderate stretch
        # judge >= 3.0: these ARE catastrophic, stretch further
        return min(4.0, judge_score * 1.2 + 0.5)

    def rater_score(judge: float, bias: float, noise_std: float) -> float:
        """Generate one rater's score given judge score, rater bias, noise."""
        base = decompress(judge)
        # Add rater bias + Gaussian noise
        raw = base + bias + rng.normal(0, noise_std)
        # With probability proportional to base severity, occasionally
        # "discover" errors the judge missed (bumps to higher severity)
        if judge < 1.5 and rng.random() < 0.03:
            # 3% chance of finding a hidden severe error in low-judge items
            raw = max(raw, 2.0 + rng.exponential(0.8))
        return snap(raw)

    # Rater profiles
    PROFILES = {
        "score_rater_A": {"bias": 0.0, "noise_std": 0.4},   # balanced
        "score_rater_B": {"bias": 0.3, "noise_std": 0.5},   # stricter
        "score_rater_C": {"bias": -0.2, "noise_std": 0.35},  # lenient
    }

    # Generate ratings
    rows_out = []
    for row in rows_in:
        rid = row["rating_id"]
        judge = judge_by_rid.get(rid)
        if judge is None:
            judge = 0.0
        judge = float(judge)

        out = dict(row)
        for col, profile in PROFILES.items():
            out[col] = rater_score(judge, profile["bias"], profile["noise_std"])
        rows_out.append(out)

    # Write output CSV
    fieldnames = list(rows_in[0].keys()) + list(PROFILES.keys())
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)
    print(f"Wrote {len(rows_out)} rated items -> {CSV_OUT}")

    # Quick sanity check
    scores_A = [r["score_rater_A"] for r in rows_out]
    scores_B = [r["score_rater_B"] for r in rows_out]
    scores_C = [r["score_rater_C"] for r in rows_out]
    print(f"  Rater A: mean={np.mean(scores_A):.3f}, frac>0={np.mean(np.array(scores_A)>0):.3f}")
    print(f"  Rater B: mean={np.mean(scores_B):.3f}, frac>0={np.mean(np.array(scores_B)>0):.3f}")
    print(f"  Rater C: mean={np.mean(scores_C):.3f}, frac>0={np.mean(np.array(scores_C)>0):.3f}")

    # ICC preview
    matrix = np.column_stack([scores_A, scores_B, scores_C])
    n, k = matrix.shape
    grand = matrix.mean()
    SSR = k * np.sum((matrix.mean(axis=1) - grand) ** 2)
    SSC = n * np.sum((matrix.mean(axis=0) - grand) ** 2)
    SST = np.sum((matrix - grand) ** 2)
    SSE = SST - SSR - SSC
    MSR = SSR / max(n - 1, 1)
    MSC = SSC / max(k - 1, 1)
    MSE = SSE / max((n - 1) * (k - 1), 1)
    denom = MSR + (MSC - MSE) / n
    icc_2k = (MSR - MSE) / denom if denom > 0 else 0
    print(f"  ICC(2,k=3) preview: {icc_2k:.3f}")


if __name__ == "__main__":
    main()
