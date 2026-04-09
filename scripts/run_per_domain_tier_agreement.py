"""Task 1.3: per-domain and per-tier judge agreement.

Stratify the 60,568 dual-scored records by domain (8) and tier (5),
and compute linear + quadratic Cohen's kappa and Spearman rho per
stratum. Also test whether mean judge disagreement per domain
correlates with mean b per domain (a potential confound).

Uses integer-index quantisation for sklearn kappa.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT = Path("C:/projects/errorquake")
SCORES = ROOT / "results" / "scores"
EXP4 = ROOT / "results" / "analysis" / "exp4_domains.json"
OUT = ROOT / "results" / "analysis" / "agreement_per_domain_tier.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}

GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])


def quantize(x: float) -> int:
    return int(np.argmin(np.abs(GRID - float(x))))


def main() -> None:
    print("=" * 70)
    print("PER-DOMAIN AND PER-TIER JUDGE AGREEMENT (v5 Task 1.3)")
    print("=" * 70)

    pri_by_dom = {}
    sec_by_dom = {}
    pri_by_tier = {}
    sec_by_tier = {}

    for f in sorted(SCORES.glob("*.jsonl")):
        if f.stem in EXCLUDED:
            continue
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p, s = r.get("primary_score"), r.get("secondary_score")
            if p is None or s is None:
                continue
            dom = r.get("domain")
            tier = r.get("tier")
            if dom:
                pri_by_dom.setdefault(dom, []).append(quantize(p))
                sec_by_dom.setdefault(dom, []).append(quantize(s))
            if tier is not None:
                pri_by_tier.setdefault(tier, []).append(quantize(p))
                sec_by_tier.setdefault(tier, []).append(quantize(s))

    def agreement(pri, sec):
        if len(pri) < 2:
            return None
        a = np.asarray(pri)
        b = np.asarray(sec)
        return {
            "n": int(len(pri)),
            "kappa_linear": float(cohen_kappa_score(a, b, weights="linear")),
            "kappa_quadratic": float(cohen_kappa_score(a, b, weights="quadratic")),
            "spearman": float(stats.spearmanr(a, b)[0]),
            "mad": float(np.mean(np.abs(a - b))),
        }

    domain_results = {}
    print()
    print(f"{'domain':<6} {'n':>7} {'k_lin':>7} {'k_quad':>7} {'rho':>7} {'MAD':>6}")
    print("-" * 50)
    for dom in sorted(pri_by_dom):
        a = agreement(pri_by_dom[dom], sec_by_dom[dom])
        domain_results[dom] = a
        print(f"{dom:<6} {a['n']:>7} {a['kappa_linear']:>7.3f} "
              f"{a['kappa_quadratic']:>7.3f} {a['spearman']:>7.3f} {a['mad']:>6.2f}")

    tier_results = {}
    print()
    print(f"{'tier':<6} {'n':>7} {'k_lin':>7} {'k_quad':>7} {'rho':>7} {'MAD':>6}")
    print("-" * 50)
    for t in sorted(pri_by_tier):
        a = agreement(pri_by_tier[t], sec_by_tier[t])
        tier_results[t] = a
        print(f"T{t:<5} {a['n']:>7} {a['kappa_linear']:>7.3f} "
              f"{a['kappa_quadratic']:>7.3f} {a['spearman']:>7.3f} {a['mad']:>6.2f}")

    # Does domain judge noise track domain b?
    exp4 = json.loads(EXP4.read_text(encoding="utf-8"))
    per_domain_stats = exp4.get("per_domain_stats", {})
    dom_mean_b = {d: s.get("mean_b") for d, s in per_domain_stats.items()}
    rows = [(d, domain_results[d]["kappa_linear"], dom_mean_b.get(d))
            for d in domain_results if dom_mean_b.get(d) is not None]
    if len(rows) >= 5:
        k = np.array([r[1] for r in rows])
        b = np.array([r[2] for r in rows])
        rho_kb, p_kb = stats.spearmanr(k, b)
        print(f"\nSpearman(domain kappa, domain mean_b) = {rho_kb:+.3f} (p={p_kb:.3f})")
        print("  (if non-significant, judge noise is orthogonal to tail shape → exonerating)")
    else:
        rho_kb, p_kb = None, None

    out = {
        "domain": domain_results,
        "tier": tier_results,
        "kappa_vs_domain_b": {
            "rho": float(rho_kb) if rho_kb is not None else None,
            "p": float(p_kb) if p_kb is not None else None,
            "n": len(rows),
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
