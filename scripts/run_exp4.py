"""Experiment 4: Domain variation in b-values.

For each (model, domain), fit b on the within-domain errors. Build a
21x8 matrix of b-values. Cells with <50 errors are flagged. Compute:
  - Per-model: range and std of b across domains
  - Per-domain: range and std of b across models
  - Friedman test: do domains differ systematically?
  - Kendall's W: do models agree on which domains have heavy tails?

Output: results/analysis/exp4_domains.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.analyze import estimate_b_value

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
OUT_PATH = PROJECT_ROOT / "results" / "analysis" / "exp4_domains.json"

EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}
DOMAINS = ["BIO", "LAW", "HIST", "GEO", "SCI", "TECH", "FIN", "CULT"]
MIN_ERRORS = 50


def load_by_domain(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = r.get("domain")
        s = r.get("final_score")
        if d is not None and s is not None:
            out[d].append(float(s))
    return out


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 4: Domain variation in b-values")
    print("=" * 70)

    files = sorted(f for f in SCORES_DIR.glob("*.jsonl") if f.stem not in EXCLUDED)
    matrix: dict[str, dict[str, dict]] = {}
    print(f"\nFitting b for {len(files)} models x {len(DOMAINS)} domains...")
    print()

    header = f"{'Model':<28} " + " ".join(f"{d:>6}" for d in DOMAINS)
    print(header)
    print("-" * len(header))

    for f in files:
        per_domain = load_by_domain(f)
        row = {}
        bs = []
        printable = []
        for d in DOMAINS:
            scores = np.asarray(per_domain.get(d, []), dtype=float)
            errors = scores[scores > 0]
            n_err = int(errors.size)
            if n_err < MIN_ERRORS:
                row[d] = {"b": None, "n_errors": n_err, "flag": "too_few"}
                printable.append("--")
                continue
            try:
                bv = estimate_b_value(errors, model_name=f"{f.stem}_{d}")
                row[d] = {
                    "b": float(bv.b),
                    "ci_lower": float(bv.b_ci_lower),
                    "ci_upper": float(bv.b_ci_upper),
                    "m_min": float(bv.m_min),
                    "n_errors": n_err,
                    "n_above_mmin": int(bv.n_above_mmin),
                    "flag": "ok",
                }
                bs.append(float(bv.b))
                printable.append(f"{bv.b:.2f}")
            except Exception as exc:
                row[d] = {"b": None, "n_errors": n_err, "flag": "fit_error", "error": str(exc)[:80]}
                printable.append("X")

        matrix[f.stem] = row
        print(f"{f.stem:<28} " + " ".join(f"{x:>6}" for x in printable))

    # Per-model spread
    print()
    print("Per-model b-value spread (max - min across domains):")
    print(f"{'Model':<28} {'min':>7} {'max':>7} {'spread':>7} {'std':>7} {'n_dom':>6}")
    print("-" * 70)
    per_model_spread = {}
    for name, row in matrix.items():
        bs = [v["b"] for v in row.values() if v["b"] is not None]
        if len(bs) >= 2:
            per_model_spread[name] = {
                "min": min(bs), "max": max(bs),
                "spread": max(bs) - min(bs),
                "std": float(np.std(bs)),
                "n_domains": len(bs),
            }
            print(f"{name:<28} {min(bs):>7.3f} {max(bs):>7.3f} "
                  f"{max(bs)-min(bs):>7.3f} {np.std(bs):>7.3f} {len(bs):>6}")

    # Friedman test: per-model b across domains (need a complete-ish matrix)
    # Build matrix with only models that have all 8 domains valid
    complete_models = [m for m in matrix
                       if all(matrix[m][d]["b"] is not None for d in DOMAINS)]
    print(f"\nModels with all 8 domains valid: {len(complete_models)}")

    friedman_result = None
    kendall_w = None
    if len(complete_models) >= 5:
        data_matrix = np.array([
            [matrix[m][d]["b"] for d in DOMAINS] for m in complete_models
        ])
        # Friedman: are the domain b-values systematically different?
        stat, p = stats.friedmanchisquare(*[data_matrix[:, j] for j in range(len(DOMAINS))])
        friedman_result = {"statistic": float(stat), "p_value": float(p),
                           "n_models": len(complete_models), "n_domains": len(DOMAINS)}
        print(f"\nFriedman test: chi^2 = {stat:.3f}, p = {p:.4g}")

        # Kendall's W: agreement across models on domain ranking
        # Rank within each model row, then average
        ranks = np.array([stats.rankdata(row) for row in data_matrix])
        m, k = ranks.shape
        rank_sums = ranks.sum(axis=0)
        s = float(np.sum((rank_sums - rank_sums.mean()) ** 2))
        w = 12 * s / (m ** 2 * (k ** 3 - k))
        kendall_w = {"W": w, "interpretation":
                     "1.0=perfect agreement, 0=no agreement"}
        print(f"Kendall's W: {w:.3f}")

    # Per-domain stats
    per_domain_stats = {}
    for d in DOMAINS:
        bs = [matrix[m][d]["b"] for m in matrix if matrix[m][d]["b"] is not None]
        if len(bs) >= 2:
            per_domain_stats[d] = {
                "mean_b": float(np.mean(bs)),
                "std_b": float(np.std(bs)),
                "min_b": float(min(bs)),
                "max_b": float(max(bs)),
                "n_models": len(bs),
            }

    print(f"\nPer-domain mean b across models:")
    for d in DOMAINS:
        s = per_domain_stats.get(d)
        if s:
            print(f"  {d}: mean={s['mean_b']:.3f}, std={s['std_b']:.3f}, "
                  f"range=[{s['min_b']:.3f}, {s['max_b']:.3f}], n={s['n_models']}")

    out = {
        "matrix": matrix,
        "per_model_spread": per_model_spread,
        "per_domain_stats": per_domain_stats,
        "friedman": friedman_result,
        "kendall_w": kendall_w,
        "domains": DOMAINS,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
