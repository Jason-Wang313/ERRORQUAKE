"""Sensitivity analyses S1-S3.

S1: Scale remap (9-point -> 7-point and 5-level). Does the b-value
    RANKING of the 21 models survive coarsening?

S2: Overcall correction. Randomly downgrade 33% of 2.0 scores to 0.0
    and refit. Repeat 100 times. Does the ranking survive?

S3: Subsample stability. Randomly take 50% of records and refit.
    Repeat 100 times. Compute coefficient of variation per model.

Output: results/analysis/sensitivity.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.analyze import estimate_b_value

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
OUT_PATH = PROJECT_ROOT / "results" / "analysis" / "sensitivity.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_scores(stem: str) -> np.ndarray:
    out = []
    for line in open(SCORES_DIR / f"{stem}.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = r.get("final_score")
        if s is not None:
            out.append(float(s))
    return np.asarray(out, dtype=float)


def remap_to_7_point(scores: np.ndarray) -> np.ndarray:
    """Remap 9-point (0..4 in 0.5 steps) to 7-point (0..3 in 0.5 steps).
    Collapse the 3.5 and 4.0 catastrophic levels into 3.0.
    """
    out = scores.copy()
    out[out >= 3.0] = 3.0
    return out


def remap_to_5_level(scores: np.ndarray) -> np.ndarray:
    """Remap to coarse 5-level: 0=correct, 1=minor, 2=moderate, 3=major, 4=catastrophic."""
    out = np.zeros_like(scores)
    out[(scores > 0) & (scores <= 1.0)] = 1.0
    out[(scores > 1.0) & (scores <= 2.0)] = 2.0
    out[(scores > 2.0) & (scores <= 3.0)] = 3.0
    out[scores > 3.0] = 4.0
    return out


def fit_b(scores: np.ndarray, name: str) -> float | None:
    pos = scores[scores > 0]
    if pos.size < 30:
        return None
    try:
        return float(estimate_b_value(pos, model_name=name).b)
    except Exception:
        return None


def s1_scale_sensitivity(all_scores: dict[str, np.ndarray]) -> dict:
    print("\n--- S1: Scale sensitivity (remap to 7-point and 5-level) ---")
    original = {}
    seven = {}
    five = {}
    for name, scores in all_scores.items():
        original[name] = fit_b(scores, name)
        seven[name] = fit_b(remap_to_7_point(scores), name)
        five[name] = fit_b(remap_to_5_level(scores), name)

    common = [n for n in all_scores if all(d.get(n) is not None for d in (original, seven, five))]
    o = np.array([original[n] for n in common])
    s7 = np.array([seven[n] for n in common])
    s5 = np.array([five[n] for n in common])

    rho_7, p_7 = stats.spearmanr(o, s7)
    rho_5, p_5 = stats.spearmanr(o, s5)

    print(f"  n_models = {len(common)}")
    print(f"  9pt -> 7pt: Spearman rho = {rho_7:.3f} (p={p_7:.4g})")
    print(f"  9pt -> 5lvl: Spearman rho = {rho_5:.3f} (p={p_5:.4g})")

    return {
        "n_models": len(common),
        "spearman_9pt_to_7pt": float(rho_7),
        "spearman_9pt_to_5lvl": float(rho_5),
        "p_9pt_to_7pt": float(p_7),
        "p_9pt_to_5lvl": float(p_5),
        "ranking_stable_7pt": bool(float(rho_7) > 0.85),
        "ranking_stable_5lvl": bool(float(rho_5) > 0.85),
        "per_model": {
            n: {"9pt": original[n], "7pt": seven[n], "5lvl": five[n]}
            for n in common
        },
    }


def s2_overcall_correction(all_scores: dict[str, np.ndarray], n_trials: int = 100) -> dict:
    print(f"\n--- S2: Overcall correction (33% of 2.0 -> 0.0, {n_trials} trials) ---")
    rng = np.random.default_rng(7)
    original = {n: fit_b(s, n) for n, s in all_scores.items()}
    common = [n for n in all_scores if original[n] is not None]
    orig_ranks = stats.rankdata([original[n] for n in common])

    rhos = []
    bs_per_trial: dict[str, list[float]] = {n: [] for n in common}
    for trial in range(n_trials):
        corrected = {}
        for n in common:
            s = all_scores[n].copy()
            mask_2 = (s == 2.0)
            n_to_drop = int(round(0.33 * mask_2.sum()))
            if n_to_drop > 0:
                idx = rng.choice(np.where(mask_2)[0], size=n_to_drop, replace=False)
                s[idx] = 0.0
            corrected[n] = fit_b(s, n) or original[n]
            bs_per_trial[n].append(corrected[n])
        ranks = stats.rankdata([corrected[n] for n in common])
        rho, _ = stats.spearmanr(orig_ranks, ranks)
        rhos.append(rho)

    rhos = np.array(rhos)
    print(f"  Mean Spearman rho = {rhos.mean():.3f} +/- {rhos.std():.3f}")
    print(f"  Min / max = {rhos.min():.3f} / {rhos.max():.3f}")
    print(f"  Pass threshold (rho>0.85): {(rhos > 0.85).mean():.0%} of trials")
    return {
        "n_trials": n_trials,
        "mean_spearman": float(rhos.mean()),
        "std_spearman": float(rhos.std()),
        "min_spearman": float(rhos.min()),
        "max_spearman": float(rhos.max()),
        "fraction_above_0_85": float((rhos > 0.85).mean()),
        "ranking_stable": bool(float((rhos > 0.85).mean()) > 0.95),
        "per_model_mean_b": {n: float(np.mean(bs_per_trial[n])) for n in common},
        "per_model_std_b": {n: float(np.std(bs_per_trial[n])) for n in common},
    }


def s3_subsample_stability(all_scores: dict[str, np.ndarray], n_trials: int = 100,
                           frac: float = 0.5) -> dict:
    print(f"\n--- S3: Subsample stability ({int(frac*100)}% of records, {n_trials} trials) ---")
    rng = np.random.default_rng(11)
    per_model: dict[str, list[float]] = {}
    for name, scores in all_scores.items():
        bs = []
        for _ in range(n_trials):
            n = scores.size
            idx = rng.choice(n, size=int(frac * n), replace=False)
            sub = scores[idx]
            b = fit_b(sub, name)
            if b is not None:
                bs.append(b)
        per_model[name] = bs

    rows = []
    for name, bs in per_model.items():
        if not bs:
            continue
        arr = np.array(bs)
        cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else float("nan")
        rows.append({"model": name, "mean_b": float(arr.mean()),
                     "std_b": float(arr.std()), "cv": cv,
                     "n_trials": len(bs)})

    rows.sort(key=lambda r: r["cv"])
    print(f"  {'Model':<28} {'mean b':>8} {'std':>8} {'CV':>8}")
    for r in rows:
        print(f"  {r['model']:<28} {r['mean_b']:>8.3f} {r['std_b']:>8.3f} {r['cv']:>8.3f}")
    cvs = np.array([r["cv"] for r in rows])
    print(f"\n  Median CV = {np.median(cvs):.3f}, mean = {cvs.mean():.3f}")
    return {
        "n_trials": n_trials,
        "fraction": frac,
        "per_model": rows,
        "median_cv": float(np.median(cvs)),
        "mean_cv": float(cvs.mean()),
        "max_cv": float(cvs.max()),
    }


def main() -> None:
    print("=" * 70)
    print("SENSITIVITY ANALYSES")
    print("=" * 70)

    files = sorted(f for f in SCORES_DIR.glob("*.jsonl") if f.stem not in EXCLUDED)
    all_scores = {f.stem: load_scores(f.stem) for f in files}
    print(f"Loaded {len(all_scores)} models")

    s1 = s1_scale_sensitivity(all_scores)
    s2 = s2_overcall_correction(all_scores, n_trials=50)  # 50 trials = ~5min
    s3 = s3_subsample_stability(all_scores, n_trials=50, frac=0.5)

    out = {"S1_scale": s1, "S2_overcall": s2, "S3_subsample": s3}
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
