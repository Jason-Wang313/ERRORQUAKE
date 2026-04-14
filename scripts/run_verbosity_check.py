"""Verbosity confound check for the Exp 5 scaling correlation.

For each model, compute mean response length (chars + words) from
results/evaluations/*.jsonl, then:
  1. Check if mean length monotonically increases with parameter count
     (the reviewer's hypothesis).
  2. Check if overcall rates monotonically track length.
  3. Refit the scaling correlation as a partial Spearman controlling
     for log(mean length).

Output: results/analysis/verbosity_check.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "results" / "evaluations"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OVERCALL = ROOT / "results" / "analysis" / "overcall_diagnostic.json"
OUT = ROOT / "results" / "analysis" / "verbosity_check.json"

EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def length_stats(path: Path) -> dict:
    chars = []
    words = []
    n = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = r.get("response_text") or ""
        if not text:
            continue
        chars.append(len(text))
        words.append(len(text.split()))
        n += 1
    if not chars:
        return {"n": 0}
    return {
        "n": n,
        "mean_chars": float(np.mean(chars)),
        "median_chars": float(np.median(chars)),
        "mean_words": float(np.mean(words)),
        "median_words": float(np.median(words)),
    }


def main() -> None:
    print("=" * 70)
    print("VERBOSITY CONFOUND CHECK (Reviewer Fix 4)")
    print("=" * 70)

    # Load Exp 5 data
    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = {p["name"]: p for p in exp5["points"]}

    # Load overcall data
    overcall = json.loads(OVERCALL.read_text(encoding="utf-8"))
    overcall_per_model = overcall.get("per_model", {})

    files = sorted(EVALS.glob("*.jsonl"))
    files = [f for f in files if f.stem not in EXCLUDED]

    rows = []
    for f in files:
        s = length_stats(f)
        if s["n"] == 0:
            continue
        name = f.stem
        if name not in points:
            continue
        p = points[name]
        oc = overcall_per_model.get(name, {})
        rows.append({
            "name": name,
            "architecture": p["architecture"],
            "active_params_b": p["active_params_b"],
            "log_params": p["log_params"],
            "b_value": p["b_value"],
            "error_rate": p["error_rate"],
            "mean_chars": s["mean_chars"],
            "median_chars": s["median_chars"],
            "mean_words": s["mean_words"],
            "median_words": s["median_words"],
            "n_responses": s["n"],
            "overcall_rate": oc.get("overcall_rate"),
        })

    rows.sort(key=lambda r: r["active_params_b"])

    print()
    print(f"{'name':<28} {'arch':<6} {'params':>7} {'b':>7} {'mean_w':>8} {'overcall':>9}")
    print("-" * 70)
    for r in rows:
        oc = r["overcall_rate"]
        oc_str = f"{oc:.2f}" if oc is not None else "--"
        print(f"{r['name']:<28} {r['architecture']:<6} {r['active_params_b']:>7.1f} "
              f"{r['b_value']:>7.3f} {r['mean_words']:>8.1f} {oc_str:>9}")

    # Q1: Is length monotonically related to params?
    log_p = np.array([r["log_params"] for r in rows])
    log_w = np.log10(np.array([r["mean_words"] for r in rows]))
    rho_lw, p_lw = stats.spearmanr(log_p, log_w)
    print(f"\nSpearman(log_params, log_mean_words) all  = {rho_lw:+.3f} (p={p_lw:.4f})")

    dense_rows = [r for r in rows if r["architecture"] == "dense"]
    log_p_d = np.array([r["log_params"] for r in dense_rows])
    log_w_d = np.log10(np.array([r["mean_words"] for r in dense_rows]))
    bs_d = np.array([r["b_value"] for r in dense_rows])
    rho_lw_d, p_lw_d = stats.spearmanr(log_p_d, log_w_d)
    print(f"Spearman(log_params, log_mean_words) dense= {rho_lw_d:+.3f} (p={p_lw_d:.4f})")

    # Q2: Does length predict overcall?
    rows_with_oc = [r for r in rows if r["overcall_rate"] is not None]
    if len(rows_with_oc) >= 5:
        words_oc = np.log10(np.array([r["mean_words"] for r in rows_with_oc]))
        ocs = np.array([r["overcall_rate"] for r in rows_with_oc])
        rho_wo, p_wo = stats.spearmanr(words_oc, ocs)
        print(f"Spearman(log_mean_words, overcall_rate)   = {rho_wo:+.3f} (p={p_wo:.4f}, n={len(rows_with_oc)})")
    else:
        rho_wo, p_wo = None, None

    # Q3: Partial correlation: b ~ log_params controlling for log_words (dense only)
    # Use OLS regression: b = a + b1*log_params + b2*log_words; report b1 + p
    from numpy.linalg import lstsq
    X_d = np.column_stack([np.ones(len(dense_rows)), log_p_d, log_w_d])
    coef, _, _, _ = lstsq(X_d, bs_d, rcond=None)
    pred = X_d @ coef
    resid = bs_d - pred
    ss_resid = float(np.sum(resid ** 2))
    n_d = len(dense_rows)
    df_resid = n_d - 3
    sigma2 = ss_resid / df_resid
    cov = sigma2 * np.linalg.inv(X_d.T @ X_d)
    se = np.sqrt(np.diag(cov))
    t_stats = coef / se
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))

    print()
    print("Partial regression on dense models: b = a + b1*log10(params) + b2*log10(mean_words)")
    print(f"  intercept:    {coef[0]:+.3f} (SE {se[0]:.3f}, p={p_vals[0]:.4f})")
    print(f"  log_params:   {coef[1]:+.3f} (SE {se[1]:.3f}, p={p_vals[1]:.4f})")
    print(f"  log_words:    {coef[2]:+.3f} (SE {se[2]:.3f}, p={p_vals[2]:.4f})")

    # Headline rho for dense (recompute for sanity)
    rho_pb_d, p_pb_d = stats.spearmanr(log_p_d, bs_d)
    print(f"\n  Univariate Spearman(log_params, b) dense = {rho_pb_d:+.3f} (p={p_pb_d:.4f})")

    out = {
        "rows": rows,
        "spearman_log_params_log_words_all": float(rho_lw),
        "spearman_log_params_log_words_dense": float(rho_lw_d),
        "p_lw_all": float(p_lw),
        "p_lw_dense": float(p_lw_d),
        "spearman_log_words_overcall": float(rho_wo) if rho_wo is not None else None,
        "p_wo": float(p_wo) if p_wo is not None else None,
        "n_with_overcall": len(rows_with_oc),
        "partial_regression_dense": {
            "n": int(n_d),
            "intercept": float(coef[0]),
            "intercept_se": float(se[0]),
            "intercept_p": float(p_vals[0]),
            "beta_log_params": float(coef[1]),
            "beta_log_params_se": float(se[1]),
            "beta_log_params_p": float(p_vals[1]),
            "beta_log_words": float(coef[2]),
            "beta_log_words_se": float(se[2]),
            "beta_log_words_p": float(p_vals[2]),
        },
        "univariate_spearman_dense": {
            "rho": float(rho_pb_d),
            "p": float(p_pb_d),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

