"""Experiment 1: Distribution characterization with Vuong tests.

For each of 21 models, fit all 5 distributions and run Vuong's
likelihood ratio test between the BIC-best fit and each runner-up.
A model has a "decisive" best fit if it beats all runners at p < 0.05.

Output: results/analysis/exp1_distribution.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from errorquake.analyze import fit_all_distributions, vuong_test

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = PROJECT_ROOT / "results" / "scores"
OUT_PATH = PROJECT_ROOT / "results" / "analysis" / "exp1_distribution.json"

EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_scores(path: Path) -> np.ndarray:
    out = []
    for line in open(path, encoding="utf-8"):
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


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT 1: Distribution characterization with Vuong tests")
    print("=" * 70)

    files = sorted(f for f in SCORES_DIR.glob("*.jsonl") if f.stem not in EXCLUDED)

    results = {}
    best_count = Counter()
    decisive_count = 0
    print()
    print(f"{'Model':<28} {'Best':<22} {'2nd':<22} {'Vuong p':<10} {'Decisive':<8}")
    print("-" * 90)

    for f in files:
        scores = load_scores(f)
        positives = scores[scores > 0]
        if positives.size < 30:
            continue

        fits = fit_all_distributions(positives, model_name=f.stem, m_min=0.5)
        valid_fits = [fit for fit in fits if not (fit.bic != fit.bic) and fit.bic != float("inf")]
        if len(valid_fits) < 2:
            continue

        best = valid_fits[0]
        runners = valid_fits[1:]

        # Vuong: best vs each runner
        vuong_results = []
        decisive = True
        for r in runners:
            try:
                v = vuong_test(positives, best, r)
            except Exception as exc:
                v = {"z_statistic": None, "p_value": None,
                     "preferred": "error", "error": str(exc)[:80]}
                decisive = False
                vuong_results.append({"vs": r.distribution, **v, "delta_bic": r.bic - best.bic})
                continue
            vuong_results.append({
                "vs": r.distribution,
                "z_statistic": v["z_statistic"],
                "p_value": v["p_value"],
                "preferred": v["preferred"],
                "delta_bic": r.bic - best.bic,
            })
            # Decisive if best is significantly preferred OR delta_bic > 10
            if v["preferred"] != best.distribution and v["p_value"] < 0.05:
                decisive = False
            if v["preferred"] == best.distribution and v["p_value"] >= 0.05 and (r.bic - best.bic) < 6:
                # Best is not significantly better and BIC gap is small
                decisive = False

        if decisive:
            decisive_count += 1
        best_count[best.distribution] += 1

        # Pretty-print
        runner_first = runners[0]
        runner_p = vuong_results[0].get("p_value")
        runner_p_str = f"{runner_p:.3f}" if runner_p is not None else "--"
        print(f"{f.stem:<28} {best.distribution:<22} {runner_first.distribution:<22} "
              f"{runner_p_str:<10} {'YES' if decisive else 'no':<8}")

        results[f.stem] = {
            "best_distribution": best.distribution,
            "best_bic": float(best.bic),
            "best_aic": float(best.aic),
            "best_parameters": best.parameters,
            "n_above_m_min": int(best.n_errors),
            "decisive": decisive,
            "vuong_tests": vuong_results,
            "all_fits_bic": [
                {"dist": fit.distribution, "bic": float(fit.bic)} for fit in valid_fits
            ],
        }

    summary = {
        "n_models": len(results),
        "best_fit_counts": dict(best_count),
        "n_decisive": decisive_count,
        "fraction_decisive": decisive_count / max(len(results), 1),
    }

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models: {summary['n_models']}")
    print(f"Best-fit distribution counts:")
    for dist, n in sorted(best_count.items(), key=lambda x: -x[1]):
        print(f"  {dist}: {n}")
    print(f"Decisive (Vuong p<0.05 vs all alternatives): "
          f"{decisive_count}/{summary['n_models']} = {summary['fraction_decisive']:.1%}")

    OUT_PATH.write_text(
        json.dumps({"summary": summary, "models": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
