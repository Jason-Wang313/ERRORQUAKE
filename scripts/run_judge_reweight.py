"""Task 1.2: Judge reweighting schemes.

Four alternative aggregation rules for the per-record final score:
  primary_only  — use only primary_score
  secondary_only — use only secondary_score
  max           — take max(primary, secondary)  (conservative / worst-case)
  min           — take min(primary, secondary)  (lenient / best-case)

For each scheme: refit b on all 21 models, recompute dense headline
correlation (Spearman and partial controlling for ε).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
sys.path.insert(0, str(ROOT / "src"))
from errorquake.analyze import estimate_b_value

SCORES = ROOT / "results" / "scores"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "judge_reweight.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}

SCHEMES = {
    "primary_only": lambda p, s: p if p is not None else None,
    "secondary_only": lambda p, s: s if s is not None else None,
    "max": lambda p, s: max(p, s) if (p is not None and s is not None)
                        else (p if p is not None else s),
    "min": lambda p, s: min(p, s) if (p is not None and s is not None)
                        else (p if p is not None else s),
}


def load_records(stem: str) -> list[dict]:
    out = []
    for line in open(SCORES / f"{stem}.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def fit_b_eps(scores: list[float | None], name: str) -> tuple[float | None, float]:
    arr = np.array([s for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return None, float("nan")
    pos = arr[arr > 0]
    eps = float((arr > 0).mean())
    if pos.size < 30:
        return None, eps
    try:
        return float(estimate_b_value(pos, model_name=name).b), eps
    except Exception:
        return None, eps


def residualise(y, x):
    Xm = np.column_stack([np.ones(len(x)), x])
    c, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    return y - Xm @ c


def main() -> None:
    print("=" * 70)
    print("JUDGE REWEIGHTING (v5 Task 1.2)")
    print("=" * 70)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    dense = [p["name"] for p in exp5["points"] if p["architecture"] == "dense"]
    all_models = [p["name"] for p in exp5["points"]]
    log_p_by = {p["name"]: p["log_params"] for p in exp5["points"]}

    model_recs = {m: load_records(m) for m in all_models if m not in EXCLUDED}

    results = {}
    for scheme, fn in SCHEMES.items():
        b_by_model = {}
        eps_by_model = {}
        for m in all_models:
            if m not in model_recs:
                continue
            scores = []
            for r in model_recs[m]:
                s = fn(r.get("primary_score"), r.get("secondary_score"))
                scores.append(s)
            b, eps = fit_b_eps(scores, f"{m}_{scheme}")
            b_by_model[m] = b
            eps_by_model[m] = eps

        valid = [(log_p_by[m], b_by_model[m], eps_by_model[m]) for m in dense
                 if b_by_model.get(m) is not None]
        if len(valid) < 5:
            results[scheme] = {"error": "too_few_valid"}
            continue
        lp = np.array([v[0] for v in valid])
        bs = np.array([v[1] for v in valid])
        eps_arr = np.array([v[2] for v in valid])
        rho, p = stats.spearmanr(lp, bs)
        lp_r = residualise(lp, eps_arr)
        b_r = residualise(bs, eps_arr)
        partial_rho, partial_p = stats.spearmanr(lp_r, b_r)

        results[scheme] = {
            "n_dense": len(valid),
            "rho_dense": float(rho),
            "p_dense": float(p),
            "partial_rho_given_eps": float(partial_rho),
            "partial_p_given_eps": float(partial_p),
            "sign_preserved": bool(rho < 0),
            "p_below_05": bool(p < 0.05),
            "mean_b": float(bs.mean()),
            "mean_eps": float(eps_arr.mean()),
        }
        print(f"  {scheme:<16}  n={len(valid)}  rho={rho:+.3f} (p={p:.4f})  "
              f"partial={partial_rho:+.3f} (p={partial_p:.4f})  "
              f"mean_b={bs.mean():.3f}  mean_eps={eps_arr.mean():.3f}")

    out = {"baseline": {"rho": -0.689, "p": 0.006}, "schemes": results}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
