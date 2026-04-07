"""Experiment 3: Micro-error -> catastrophic prediction.

For each of 21 models:
  easy = tier 1-2 (low-difficulty queries) -> fit b on errors
  hard = tier 4-5 (high-difficulty queries) -> count observed catastrophes

Predict catastrophic count from b_easy via the GR extrapolation, then
correlate predicted vs observed across models (Spearman rho + Kendall tau).

Pre-registered thresholds (from Phase 5 spec):
  rho >= 0.50 -> STRONG (paper has practical contribution)
  0.30 <= rho < 0.50 -> WEAK (descriptive only)
  rho < 0.30 -> FAIL

Secondary: fraction within 1.5x of observed (>= 65% target).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.analyze import estimate_b_value, predict_catastrophic_rate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
OUT_DIR = PROJECT_ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "exp3_prediction.json"

EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
EASY_TIERS = {1, 2}
HARD_TIERS = {4, 5}
TARGET_MAGNITUDE = 3.0
FALLBACK_TARGET = 2.5


def load_scores_split(path: Path):
    """Return (easy_scores, hard_scores) as np.ndarray of final_scores."""
    easy, hard = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tier = r.get("tier")
            score = r.get("final_score")
            if score is None:
                continue
            if tier in EASY_TIERS:
                easy.append(float(score))
            elif tier in HARD_TIERS:
                hard.append(float(score))
    return np.asarray(easy, dtype=float), np.asarray(hard, dtype=float)


def run_for_target(target: float, model_data: dict) -> dict:
    """Run prediction at a given target magnitude across all models."""
    rows = []
    for name, (easy, hard) in sorted(model_data.items()):
        easy_errors = easy[easy > 0]
        if easy_errors.size < 30:
            continue
        try:
            bv = estimate_b_value(easy_errors, model_name=name)
        except Exception as exc:
            rows.append({"model": name, "error": f"b_fit: {exc}"})
            continue

        n_easy_above_mmin = int((easy_errors >= bv.m_min - 1e-9).sum())
        predicted = predict_catastrophic_rate(
            b_easy=bv.b,
            n_easy_errors=n_easy_above_mmin,
            m_min=bv.m_min,
            target_magnitude=target,
        )
        observed = int((hard >= target - 1e-9).sum())

        # Rates per query (hard set is 1600 queries: tiers 4+5)
        n_hard = int(hard.size)
        observed_rate = observed / max(n_hard, 1)
        # Predicted rate uses easy set size (tiers 1+2 = 1600 queries)
        n_easy = int(easy.size)
        predicted_rate = predicted / max(n_easy, 1)

        if observed > 0 and predicted > 0:
            ratio = predicted / observed
            within_1_5x = (1.0 / 1.5) <= ratio <= 1.5
        elif observed == 0 and predicted == 0:
            ratio = 1.0
            within_1_5x = True
        else:
            ratio = float("inf") if observed == 0 else 0.0
            within_1_5x = False

        rows.append({
            "model": name,
            "b_easy": float(bv.b),
            "b_ci_lower": float(bv.b_ci_lower),
            "b_ci_upper": float(bv.b_ci_upper),
            "m_min": float(bv.m_min),
            "n_easy_errors": int(easy_errors.size),
            "n_easy_above_mmin": n_easy_above_mmin,
            "n_easy_total": n_easy,
            "n_hard_total": n_hard,
            "predicted": float(predicted),
            "observed": int(observed),
            "predicted_rate": float(predicted_rate),
            "observed_rate": float(observed_rate),
            "ratio": float(ratio) if math.isfinite(ratio) else None,
            "within_1_5x": bool(within_1_5x),
        })

    valid = [r for r in rows if "error" not in r]
    if len(valid) < 3:
        return {"target": target, "rows": rows, "n_valid": len(valid),
                "error": "insufficient_models"}

    pred_counts = np.array([r["predicted"] for r in valid])
    obs_counts = np.array([r["observed"] for r in valid])
    pred_rates = np.array([r["predicted_rate"] for r in valid])
    obs_rates = np.array([r["observed_rate"] for r in valid])

    rho_count, p_count = stats.spearmanr(pred_counts, obs_counts)
    rho_rate, p_rate = stats.spearmanr(pred_rates, obs_rates)
    tau_count, p_tau_count = stats.kendalltau(pred_counts, obs_counts)
    tau_rate, p_tau_rate = stats.kendalltau(pred_rates, obs_rates)

    n_within = sum(1 for r in valid if r["within_1_5x"])
    frac_within = n_within / len(valid)

    return {
        "target": target,
        "n_valid": len(valid),
        "n_skipped": len(rows) - len(valid),
        "spearman_rho_counts": float(rho_count) if not math.isnan(rho_count) else None,
        "spearman_p_counts": float(p_count) if not math.isnan(p_count) else None,
        "spearman_rho_rates": float(rho_rate) if not math.isnan(rho_rate) else None,
        "spearman_p_rates": float(p_rate) if not math.isnan(p_rate) else None,
        "kendall_tau_counts": float(tau_count) if not math.isnan(tau_count) else None,
        "kendall_p_counts": float(p_tau_count) if not math.isnan(p_tau_count) else None,
        "kendall_tau_rates": float(tau_rate) if not math.isnan(tau_rate) else None,
        "kendall_p_rates": float(p_tau_rate) if not math.isnan(p_tau_rate) else None,
        "within_1_5x_count": n_within,
        "within_1_5x_fraction": frac_within,
        "rows": rows,
    }


def verdict_for(rho: float | None, frac: float) -> str:
    if rho is None:
        return "FAIL_NO_RHO"
    if rho >= 0.75:
        primary = "STRONG_PRIMARY"
    elif rho >= 0.50:
        primary = "STRONG"
    elif rho >= 0.30:
        primary = "WEAK"
    else:
        primary = "FAIL"
    secondary = "PASS" if frac >= 0.65 else "FAIL"
    return f"{primary}+SEC_{secondary}"


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 3: Micro-error -> Catastrophic Prediction")
    print("=" * 70)

    files = sorted(SCORES_DIR.glob("*.jsonl"))
    files = [f for f in files if f.stem not in EXCLUDED]
    print(f"Loading {len(files)} models...")

    model_data = {}
    for f in files:
        easy, hard = load_scores_split(f)
        # Skip models with too few records (e.g., partials)
        if easy.size < 100 or hard.size < 100:
            print(f"  SKIP {f.stem}: easy={easy.size}, hard={hard.size}")
            continue
        model_data[f.stem] = (easy, hard)
        print(f"  {f.stem}: easy={easy.size} (errors={int((easy>0).sum())}), "
              f"hard={hard.size} (catastrophes>=3.0={int((hard>=3.0).sum())})")

    print(f"\nLoaded {len(model_data)} models with valid data")
    print()

    results = {"models_used": sorted(model_data.keys())}

    for target in (TARGET_MAGNITUDE, FALLBACK_TARGET):
        key = f"target_{target}"
        print(f"--- Running with target={target} ---")
        result = run_for_target(target, model_data)
        results[key] = result

        if "spearman_rho_counts" in result:
            rho_c = result["spearman_rho_counts"]
            rho_r = result["spearman_rho_rates"]
            tau_c = result["kendall_tau_counts"]
            frac = result["within_1_5x_fraction"]
            n = result["n_valid"]

            print(f"  n_valid = {n}")
            print(f"  Spearman rho (counts) = {rho_c:.4f} (p={result['spearman_p_counts']:.4f})")
            print(f"  Spearman rho (rates)  = {rho_r:.4f} (p={result['spearman_p_rates']:.4f})")
            print(f"  Kendall  tau (counts) = {tau_c:.4f} (p={result['kendall_p_counts']:.4f})")
            print(f"  Within 1.5x: {result['within_1_5x_count']}/{n} = {frac:.1%}")

            v = verdict_for(rho_c, frac)
            print(f"  VERDICT (counts, 1.5x): {v}")
        else:
            print(f"  ERROR: {result.get('error')}")
        print()

    # Pick the canonical verdict from primary target
    primary = results[f"target_{TARGET_MAGNITUDE}"]
    if "spearman_rho_counts" in primary:
        rho = primary["spearman_rho_counts"]
        frac = primary["within_1_5x_fraction"]
        results["headline_verdict"] = verdict_for(rho, frac)
        results["headline_rho"] = rho
        results["headline_within_1_5x"] = frac

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {OUT_PATH}")
    print()
    print("=" * 70)
    print("HEADLINE")
    print("=" * 70)
    if "headline_rho" in results:
        print(f"Primary target = {TARGET_MAGNITUDE}")
        print(f"Spearman rho   = {results['headline_rho']:.4f}")
        print(f"Within 1.5x    = {results['headline_within_1_5x']:.1%}")
        print(f"VERDICT        = {results['headline_verdict']}")


if __name__ == "__main__":
    main()
