"""S5 — m_min sensitivity for the Exp 5 headline scaling correlation.

Recompute the dense scaling Spearman under two alternative m_min
strategies:
  (a) fixed m_min = 1.5 for all models (large tail samples)
  (b) Clauset-style: smallest grid point with at least 100 exceedances

Output: results/analysis/s5_mmin.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import _estimate_b, _quantize_to_grid, DEFAULT_SCALE_POINTS

SCORES = ROOT / "results" / "scores"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "s5_mmin.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_positive_scores(stem: str) -> np.ndarray:
    out = []
    for line in open(SCORES / f"{stem}.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = r.get("final_score")
        if s is not None and s > 0:
            out.append(float(s))
    return _quantize_to_grid(np.asarray(out, dtype=float))


def fit_b_at(scores: np.ndarray, m_min: float, min_n: int) -> tuple[float, int]:
    """Return (b, n_above) with n_above guaranteed >= min_n or NaN."""
    above = scores[scores >= m_min - 1e-9]
    if above.size < min_n:
        return float("nan"), int(above.size)
    return float(_estimate_b(scores, m_min)), int(above.size)


def fit_b_clauset(scores: np.ndarray, min_n: int = 100) -> tuple[float, float, int]:
    """Smallest grid m_min with at least min_n exceedances; return (b, m_min, n_above)."""
    for gp in DEFAULT_SCALE_POINTS:
        n_above = int((scores >= gp - 1e-9).sum())
        if n_above >= min_n:
            return float(_estimate_b(scores, float(gp))), float(gp), n_above
    return float("nan"), float("nan"), 0


def main() -> None:
    print("=" * 70)
    print("S5 — m_min sensitivity")
    print("=" * 70)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = {p["name"]: p for p in exp5["points"]}

    rows = []
    for name, p in points.items():
        if name in EXCLUDED:
            continue
        scores = load_positive_scores(name)
        b_fixed, n_fixed = fit_b_at(scores, m_min=1.5, min_n=30)
        b_clauset, m_clauset, n_clauset = fit_b_clauset(scores, min_n=100)
        rows.append({
            "name": name,
            "architecture": p["architecture"],
            "log_params": p["log_params"],
            "b_default": p["b_value"],
            "b_fixed_1_5": b_fixed,
            "n_above_fixed": n_fixed,
            "b_clauset_100": b_clauset,
            "m_min_clauset": m_clauset,
            "n_above_clauset": n_clauset,
        })

    rows.sort(key=lambda r: (r["architecture"], r["log_params"]))

    print()
    print(f"{'name':<28} {'arch':<6} {'def':>7} {'fix1.5':>8} {'n':>5} {'clauset':>9} {'m':>5} {'n':>5}")
    print("-" * 80)
    for r in rows:
        b_def = r["b_default"]
        b_fix = r["b_fixed_1_5"]
        b_cl = r["b_clauset_100"]
        m_cl = r["m_min_clauset"]
        nc = r["n_above_clauset"]
        nf = r["n_above_fixed"]
        b_fix_s = f"{b_fix:.3f}" if not np.isnan(b_fix) else "  --   "
        b_cl_s = f"{b_cl:.3f}" if not np.isnan(b_cl) else "  --   "
        m_cl_s = f"{m_cl:.1f}" if not np.isnan(m_cl) else " -- "
        print(f"{r['name']:<28} {r['architecture']:<6} {b_def:>7.3f} {b_fix_s:>8} "
              f"{nf:>5} {b_cl_s:>9} {m_cl_s:>5} {nc:>5}")

    # Headline correlation under each strategy (dense only)
    dense = [r for r in rows if r["architecture"] == "dense"]
    log_p_d = np.array([r["log_params"] for r in dense])

    def corr(b_key: str) -> dict:
        valid = [(r["log_params"], r[b_key]) for r in dense
                 if r[b_key] is not None and not np.isnan(r[b_key])]
        if len(valid) < 5:
            return {"n": len(valid), "rho": None, "p": None}
        xs = np.array([v[0] for v in valid])
        ys = np.array([v[1] for v in valid])
        rho, p = stats.spearmanr(xs, ys)
        return {"n": len(valid), "rho": float(rho), "p": float(p)}

    head_def = corr("b_default")
    head_fix = corr("b_fixed_1_5")
    head_cl = corr("b_clauset_100")

    print()
    print("Headline scaling correlation (dense only):")
    print(f"  default selector       : n={head_def['n']:2d}  rho={head_def['rho']:+.3f}  p={head_def['p']:.4f}")
    print(f"  fixed m_min = 1.5      : n={head_fix['n']:2d}  "
          f"rho={head_fix['rho']:+.3f}  p={head_fix['p']:.4f}")
    print(f"  Clauset (>=100 above)  : n={head_cl['n']:2d}  "
          f"rho={head_cl['rho']:+.3f}  p={head_cl['p']:.4f}")

    out = {
        "rows": rows,
        "headline_default": head_def,
        "headline_fixed_1_5": head_fix,
        "headline_clauset_100": head_cl,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
