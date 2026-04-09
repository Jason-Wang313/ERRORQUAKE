"""Priority 0: investigate why dense error rate scales POSITIVELY
with parameter count (ρ_s(log_p, ε) = +0.59 in v4).

Four hypotheses:
  H1 abstention   — small models abstain more → fewer errors
  H2 verbosity    — already falsified in v4 (ρ_s(log_p, words) = -0.08)
  H3 difficulty   — larger models fail specifically on hard tiers
  H4 family       — one family drives the correlation

Output: results/analysis/priority0_eps_size.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("C:/projects/errorquake")
EVALS = ROOT / "results" / "evaluations"
SCORES = ROOT / "results" / "scores"
EXP5 = ROOT / "results" / "analysis" / "exp5_scaling.json"
OUT = ROOT / "results" / "analysis" / "priority0_eps_size.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}

# Family mapping (drop suffixes, group by shared architecture lineage).
FAMILY = {
    "llama-3.2-3b-instruct": "llama",
    "llama-3.1-8b-instruct": "llama",
    "llama-4-maverick": "llama",
    "gemma-3-4b": "gemma",
    "gemma-3-12b": "gemma",
    "gemma-3-27b": "gemma",
    "gemma-2-27b": "gemma",
    "qwen2.5-7b": "qwen",
    "qwen3-next-80b": "qwen",
    "mistral-small-24b": "mistral",
    "mistral-small-4-119b": "mistral",
    "mistral-medium-3": "mistral",
    "ministral-14b": "mistral",
    "deepseek-v3.1": "deepseek",
    "deepseek-v3.2": "deepseek",
    "phi-3.5-mini": "phi",
    "solar-10.7b": "solar",
    "eurollm-9b": "eurollm",
    "seed-oss-36b": "seed",
    "gpt-oss-20b": "gpt-oss",
    "kimi-k2-instruct": "kimi",
}

ABSTENTION_PATTERNS = [
    r"(?i)\bI don'?t know\b",
    r"(?i)\bI'?m not sure\b",
    r"(?i)\bI cannot (answer|provide|help|tell|determine|confirm|give)\b",
    r"(?i)\bI don'?t have (enough )?information\b",
    r"(?i)\bunable to (answer|provide|determine|verify|confirm)\b",
    r"(?i)\bbeyond my (knowledge|ability|training)\b",
    r"(?i)\bI apologize.{0,40}(cannot|unable|don'?t)\b",
    r"(?i)\bas an AI\b",
    r"(?i)\bI'?m (just )?an AI\b",
    r"(?i)\bno (reliable )?(information|data|evidence) (is )?available\b",
    r"(?i)\bthis (question|query) (is )?outside (my|the)\b",
    r"(?i)\bI do not have (the )?(specific|detailed)\b",
    r"(?i)\bI can'?t (accurately|reliably) (answer|say|tell|determine)\b",
]
ABSTENTION_RE = re.compile("|".join(ABSTENTION_PATTERNS))


def load_responses(stem: str) -> list[dict]:
    """Load per-query responses + scores joined by query_id."""
    eval_path = EVALS / f"{stem}.jsonl"
    score_path = SCORES / f"{stem}.jsonl"
    # Load scores first for join
    scores_by_qid = {}
    if score_path.exists():
        for line in open(score_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            scores_by_qid[r["query_id"]] = r
    records = []
    if eval_path.exists():
        for line in open(eval_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = r.get("query_id")
            text = r.get("response_text") or ""
            score_rec = scores_by_qid.get(qid, {})
            final = score_rec.get("final_score")
            records.append({
                "query_id": qid,
                "domain": r.get("domain"),
                "tier": r.get("tier"),
                "response_text": text,
                "final_score": final,
                "text_len": len(text.split()),
            })
    return records


def is_abstention(text: str) -> bool:
    if not text:
        return True  # empty response counts as abstention
    return bool(ABSTENTION_RE.search(text))


def main() -> None:
    print("=" * 70)
    print("PRIORITY 0: ε-size anomaly investigation")
    print("=" * 70)

    exp5 = json.loads(EXP5.read_text(encoding="utf-8"))
    points = {p["name"]: p for p in exp5["points"]}

    rows = []
    for name, p in points.items():
        if name in EXCLUDED:
            continue
        recs = load_responses(name)
        if not recs:
            continue
        total = len(recs)
        # Only records with both response and score (valid for this analysis)
        scored = [r for r in recs if r.get("final_score") is not None and r.get("response_text")]
        if len(scored) < 100:
            continue  # need enough data
        # Abstentions
        abst = [r for r in scored if is_abstention(r["response_text"])]
        n_abst = len(abst)
        # Attempted = non-abstentions
        attempted = [r for r in scored if not is_abstention(r["response_text"])]
        n_att = len(attempted)
        # Original error counts
        errors_all = [r for r in scored if r["final_score"] > 0]
        eps_overall = len(errors_all) / max(len(scored), 1)
        # Attempted error rate
        att_errors = [r for r in attempted if r["final_score"] > 0]
        eps_attempted = len(att_errors) / max(n_att, 1) if n_att else float("nan")
        # Per-tier error rates on attempted
        per_tier_eps = {}
        per_tier_att_eps = {}
        for t in range(1, 6):
            t_all = [r for r in scored if r.get("tier") == t]
            t_att = [r for r in attempted if r.get("tier") == t]
            per_tier_eps[t] = (
                sum(1 for r in t_all if r["final_score"] > 0) / max(len(t_all), 1)
                if t_all else float("nan")
            )
            per_tier_att_eps[t] = (
                sum(1 for r in t_att if r["final_score"] > 0) / max(len(t_att), 1)
                if t_att else float("nan")
            )
        rows.append({
            "name": name,
            "architecture": p["architecture"],
            "log_params": p["log_params"],
            "b_value": p["b_value"],
            "family": FAMILY.get(name, "unknown"),
            "n_scored": len(scored),
            "n_abst": n_abst,
            "abst_rate": n_abst / len(scored),
            "eps_overall": eps_overall,
            "n_attempted": n_att,
            "eps_attempted": eps_attempted,
            "per_tier_eps": per_tier_eps,
            "per_tier_att_eps": per_tier_att_eps,
        })

    print()
    print(f"{'name':<28} {'arch':<6} {'log_p':>7} {'b':>6} {'abst%':>6} "
          f"{'eps_all':>8} {'eps_att':>8}")
    print("-" * 75)
    for r in sorted(rows, key=lambda x: x["log_params"]):
        print(f"{r['name']:<28} {r['architecture']:<6} {r['log_params']:>7.2f} "
              f"{r['b_value']:>6.3f} {r['abst_rate']*100:>5.1f}% "
              f"{r['eps_overall']:>8.3f} {r['eps_attempted']:>8.3f}")

    dense = [r for r in rows if r["architecture"] == "dense"]
    log_p_d = np.array([r["log_params"] for r in dense])
    bs_d = np.array([r["b_value"] for r in dense])
    eps_d = np.array([r["eps_overall"] for r in dense])
    eps_att_d = np.array([r["eps_attempted"] for r in dense])
    abst_d = np.array([r["abst_rate"] for r in dense])

    # H1: abstention mediation
    rho_pa, p_pa = stats.spearmanr(log_p_d, abst_d)
    rho_ae, p_ae = stats.spearmanr(abst_d, eps_d)
    rho_pe, p_pe = stats.spearmanr(log_p_d, eps_d)
    rho_pe_att, p_pe_att = stats.spearmanr(log_p_d, eps_att_d)
    rho_pb, p_pb = stats.spearmanr(log_p_d, bs_d)

    print()
    print("=" * 70)
    print("H1 — ABSTENTION")
    print("=" * 70)
    print(f"  rho(log_p, abst)         = {rho_pa:+.3f} (p={p_pa:.3f})")
    print(f"  rho(abst, eps_overall)   = {rho_ae:+.3f} (p={p_ae:.3f})")
    print(f"  rho(log_p, eps_overall)  = {rho_pe:+.3f} (p={p_pe:.3f})  [baseline v4 finding]")
    print(f"  rho(log_p, eps_attempted)= {rho_pe_att:+.3f} (p={p_pe_att:.3f})  [CORRECTED]")

    # Partial correlations
    def residualise(y, x):
        Xm = np.column_stack([np.ones(len(x)), x])
        c, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        return y - Xm @ c

    # Partial rho(log_p, eps | abst)
    lp_resid = residualise(log_p_d, abst_d)
    eps_resid = residualise(eps_d, abst_d)
    partial_pe_abst, partial_p_pe_abst = stats.spearmanr(lp_resid, eps_resid)
    print(f"  partial rho(log_p, eps | abst) = {partial_pe_abst:+.3f} "
          f"(p={partial_p_pe_abst:.3f})")

    # Partial rho(log_p, b | eps_att)
    lp_resid2 = residualise(log_p_d, eps_att_d)
    b_resid2 = residualise(bs_d, eps_att_d)
    partial_pb_att, partial_p_pb_att = stats.spearmanr(lp_resid2, b_resid2)
    print(f"  partial rho(log_p, b | eps_attempted) = {partial_pb_att:+.3f} "
          f"(p={partial_p_pb_att:.3f})")

    # Partial rho(log_p, b | eps_overall, abst) — triple partial
    # Residualise log_p and b against both [eps, abst] simultaneously
    X_full = np.column_stack([np.ones(len(dense)), eps_d, abst_d])
    c_lp, *_ = np.linalg.lstsq(X_full, log_p_d, rcond=None)
    lp_resid3 = log_p_d - X_full @ c_lp
    c_b, *_ = np.linalg.lstsq(X_full, bs_d, rcond=None)
    b_resid3 = bs_d - X_full @ c_b
    triple_pb, triple_p = stats.spearmanr(lp_resid3, b_resid3)
    print(f"  triple partial rho(log_p, b | eps, abst) = {triple_pb:+.3f} "
          f"(p={triple_p:.3f})")

    # H3: per-tier difficulty
    print()
    print("=" * 70)
    print("H3 — BENCHMARK DIFFICULTY (per-tier eps vs size on dense)")
    print("=" * 70)
    per_tier_results = {}
    for t in range(1, 6):
        tier_eps = np.array([r["per_tier_eps"].get(t, float("nan")) for r in dense])
        tier_att = np.array([r["per_tier_att_eps"].get(t, float("nan")) for r in dense])
        mask = ~np.isnan(tier_eps)
        if mask.sum() >= 5:
            rho_pe_t, p_pe_t = stats.spearmanr(log_p_d[mask], tier_eps[mask])
            rho_pe_at_t, p_pe_at_t = stats.spearmanr(log_p_d[mask], tier_att[mask])
            per_tier_results[t] = {
                "rho_log_p_eps": float(rho_pe_t), "p": float(p_pe_t),
                "rho_log_p_eps_att": float(rho_pe_at_t), "p_att": float(p_pe_at_t),
                "n": int(mask.sum()),
            }
            print(f"  T{t}: rho(log_p, eps) = {rho_pe_t:+.3f} (p={p_pe_t:.3f})  "
                  f"rho(log_p, eps_att) = {rho_pe_at_t:+.3f} (p={p_pe_at_t:.3f})")

    # H4: family confound (leave-one-family-out on rho(log_p, eps))
    print()
    print("=" * 70)
    print("H4 — FAMILY CONFOUND (LOFO on rho(log_p, eps))")
    print("=" * 70)
    lofo = {}
    families = sorted({r["family"] for r in dense})
    for fam in families:
        sub = [r for r in dense if r["family"] != fam]
        if len(sub) < 5:
            continue
        lp = np.array([r["log_params"] for r in sub])
        ep = np.array([r["eps_overall"] for r in sub])
        rho, p = stats.spearmanr(lp, ep)
        lofo[fam] = {"n_remaining": len(sub), "rho": float(rho), "p": float(p)}
        print(f"  drop {fam:<10}: n={len(sub):>2}  rho={rho:+.3f}  p={p:.3f}")

    summary = {
        "baseline_v4": {
            "rho_log_p_eps": float(rho_pe),
            "rho_log_p_b": float(rho_pb),
            "rho_log_p_eps_attempted": float(rho_pe_att),
            "rho_log_p_abst": float(rho_pa),
            "rho_abst_eps": float(rho_ae),
        },
        "partial_correlations": {
            "partial_rho_log_p_eps_given_abst": float(partial_pe_abst),
            "partial_p_log_p_eps_given_abst": float(partial_p_pe_abst),
            "partial_rho_log_p_b_given_eps_attempted": float(partial_pb_att),
            "partial_p_log_p_b_given_eps_attempted": float(partial_p_pb_att),
            "triple_partial_rho_log_p_b_given_eps_and_abst": float(triple_pb),
            "triple_partial_p": float(triple_p),
        },
        "per_tier": per_tier_results,
        "lofo_family": lofo,
        "per_model": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
