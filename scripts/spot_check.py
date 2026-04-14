"""Spot-check key paper claims against saved analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "results" / "analysis"
EXPANDED_HUMAN = ROOT / "data" / "human_audit" / "expanded_study" / "analysis_report.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(label: str, actual, expected, tol: float = 0.005) -> bool:
    if isinstance(expected, int):
        ok = actual == expected
    else:
        ok = abs(float(actual) - float(expected)) <= tol
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {label}: actual={actual}, paper={expected}")
    return ok


def main() -> None:
    results: list[bool] = []

    scaleup = load(ANALYSIS / "v7_4k_vs_10k.json")
    results.append(check("Exp1 decisive fits (10K)", scaleup["n_decisive"]["10k"], 17))
    results.append(check("Exp1 non-exponential fits (10K)", scaleup["n_nonexp"]["10k"], 17))

    exp3 = load(ANALYSIS / "exp3_prediction.json")
    target3 = exp3["target_3.0"]
    results.append(check("Exp3 rho (M>=3.0)", target3["spearman_rho_counts"], 0.443, 0.002))
    results.append(check("Exp3 p (M>=3.0)", target3["spearman_p_counts"], 0.044, 0.002))
    results.append(check("Exp3 within-1.5x count", target3["within_1_5x_count"], 4))

    results.append(check("Exp5 dense rho (10K)", scaleup["rho_scale"]["10k"], -0.5617, 0.002))
    results.append(check("Exp5 partial rho (10K)", scaleup["partial_rho"]["10k"], -0.2044, 0.002))

    sensitivity = load(ANALYSIS / "sensitivity.json")
    results.append(check("S1 7-point rho", sensitivity["S1_scale"]["spearman_9pt_to_7pt"], 0.43, 0.02))
    results.append(check("S1 5-level rho", sensitivity["S1_scale"]["spearman_9pt_to_5lvl"], 0.16, 0.02))
    results.append(check("S2 mean rho", sensitivity["S2_overcall"]["mean_spearman"], 0.847, 0.002))
    results.append(check("S3 median CV", sensitivity["S3_subsample"]["median_cv"], 0.143, 0.005))

    oral = load(ANALYSIS / "oral_upgrade" / "oral_upgrade_analyses.json")
    mi = oral["mi_decomposition"]
    results.append(check("Conditional MI I(b;model|eps)", mi["I_b_model_given_eps"], 1.5645, 0.002))
    results.append(check("R^2(b~eps)", mi["direct_correlation"]["r_squared_linear"], 0.3555, 0.002))

    full_human = load(ANALYSIS / "v10_full_human.json")
    results.append(check("Human headline disjoint CIs", full_human["headline_human"]["n_disjoint_CIs"], 85))
    results.append(check("Judge baseline pairs", full_human["headline_human"]["n_judge_baseline"], 31))

    expanded = load(EXPANDED_HUMAN)
    results.append(check("Expanded-study ICC(2,k=3)", expanded["icc_9pt"]["icc_2k"], 0.8513, 0.002))
    results.append(check("Expanded-study human/judge rho", expanded["b_values"]["rho"], 0.8857, 0.002))
    results.append(check("Expanded-study dense scaling rho", expanded["b_values"]["dense_scaling_rho"], -0.8636, 0.002))
    results.append(check("Expanded-study Fleiss kappa", expanded["fleiss_kappa"]["kappa"], 0.8308, 0.002))
    results.append(check("Expanded-study overcall", expanded["overcall"]["overcall_rate"], 0.1365, 0.002))

    passed = sum(results)
    total = len(results)
    print(f"\nSPOT CHECK: {passed}/{total} match")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
