"""Spot-check numbers asserted in paper/main.tex against JSON sources.

Each check loads a specific JSON and compares against the rounded
claim in the paper. Fails loudly on mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

A = Path("C:/projects/errorquake/results/analysis")


def check(label: str, actual, expected, tol=0.005):
    if isinstance(expected, (int, str)):
        ok = actual == expected
    else:
        ok = abs(float(actual) - float(expected)) <= tol
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: actual={actual}, paper={expected}")
    return ok


results = []

# 1. Exp 5: dense Spearman rho and p-value
d = json.loads((A / "exp5_scaling.json").read_text(encoding="utf-8"))
results.append(check("Exp5 dense Spearman rho",
                     d["correlations"]["dense"]["spearman_rho"], -0.689, 0.001))
results.append(check("Exp5 dense Spearman p",
                     d["correlations"]["dense"]["spearman_p"], 0.006, 0.001))
results.append(check("Exp5 dense n", d["correlations"]["dense"]["n"], 14))

# 2. Exp 5 all-models
results.append(check("Exp5 all Spearman rho",
                     d["correlations"]["all"]["spearman_rho"], -0.585, 0.001))
results.append(check("Exp5 all n", d["correlations"]["all"]["n"], 21))

# 3. Exp 3 primary result
d = json.loads((A / "exp3_prediction.json").read_text(encoding="utf-8"))
t3 = d["target_3.0"]
results.append(check("Exp3 Spearman counts rho",
                     t3["spearman_rho_counts"], 0.443, 0.002))
results.append(check("Exp3 Spearman counts p",
                     t3["spearman_p_counts"], 0.044, 0.002))
results.append(check("Exp3 n_valid", t3["n_valid"], 21))
results.append(check("Exp3 within_1_5x_count", t3["within_1_5x_count"], 4))

# 4. Exp 3 secondary
t25 = d["target_2.5"]
results.append(check("Exp3 M>=2.5 Spearman rho",
                     t25["spearman_rho_counts"], 0.637, 0.002))
results.append(check("Exp3 M>=2.5 p",
                     t25["spearman_p_counts"], 0.002, 0.001))

# 5. Exp 2 disjoint CI count
d = json.loads((A / "exp2_discriminator.json").read_text(encoding="utf-8"))
results.append(check("Exp2 disjoint-CI pairs",
                     d["summary"]["n_pairs_disjoint_CIs"], 30))
results.append(check("Exp2 qualifying pairs",
                     d["summary"]["n_pairs_qualifying"], 42))

# 6. Exp 1 decisive fits and family counts
d = json.loads((A / "exp1_distribution.json").read_text(encoding="utf-8"))
results.append(check("Exp1 n_decisive",
                     d["summary"]["n_decisive"], 18))
results.append(check("Exp1 n_models",
                     d["summary"]["n_models"], 21))
results.append(check("Exp1 stretched_exp count",
                     d["summary"]["best_fit_counts"].get("stretched_exp", 0), 10))

# 7. Exp 4 Friedman + Kendall W
d = json.loads((A / "exp4_domains.json").read_text(encoding="utf-8"))
results.append(check("Exp4 Friedman chi2",
                     d["friedman"]["statistic"], 15.94, 0.1))
results.append(check("Exp4 Friedman p",
                     d["friedman"]["p_value"], 0.026, 0.002))
results.append(check("Exp4 Kendall W",
                     d["kendall_w"]["W"], 0.108, 0.005))

# 8. Sensitivity
d = json.loads((A / "sensitivity.json").read_text(encoding="utf-8"))
results.append(check("S1 7pt rho", d["S1_scale"]["spearman_9pt_to_7pt"], 0.43, 0.02))
results.append(check("S1 5lvl rho", d["S1_scale"]["spearman_9pt_to_5lvl"], 0.16, 0.02))
results.append(check("S2 mean rho", d["S2_overcall"]["mean_spearman"], 0.847, 0.002))
results.append(check("S3 median CV", d["S3_subsample"]["median_cv"], 0.143, 0.005))

# 9. Overcall
d = json.loads((A / "overcall_diagnostic.json").read_text(encoding="utf-8"))
results.append(check("Overcall n_sampled",
                     d["overall"]["n_total_sampled"], 340))
results.append(check("Overcall overall rate",
                     d["overall"]["overall_overcall_rate"], 0.335, 0.003))

# 10. Exp 5 leave-one-out
d = json.loads((A / "exp5_loo.json").read_text(encoding="utf-8"))
results.append(check("LOO n sig/14",
                     d["n_significant_after_drop"], 14))
results.append(check("LOO worst-case p",
                     d["loo_max_p"], 0.0263, 0.001))

n_pass = sum(results)
n_total = len(results)
print()
print(f"SPOT CHECK: {n_pass}/{n_total} match")
if n_pass != n_total:
    import sys
    sys.exit(1)
