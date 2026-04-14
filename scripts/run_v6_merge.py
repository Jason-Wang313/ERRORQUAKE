"""A5: merge 4K baseline + 6K supplement → 10K consolidated dataset.

Reads:
  results/evaluations/{model}.jsonl       (4K baseline, untouched)
  results/scores/{model}.jsonl            (4K baseline, untouched)
  results/evaluations_v6_supplement/{model}.jsonl   (new 6K)
  results/scores_v6_supplement/{model}.jsonl        (new 6K)

Writes:
  results/evaluations_10k/{model}.jsonl   (merged)
  results/scores_10k/{model}.jsonl        (merged)
  results/analysis/v6_merge_report.json   (per-model coverage,
                                            batch-consistency check)

Crash-resistant: idempotent merges, never overwrites existing
records, ignores duplicates (keeps the older 4K record on conflict).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_4K = ROOT / "results" / "evaluations"
EVAL_6K = ROOT / "results" / "evaluations_v6_supplement"
SCORES_4K = ROOT / "results" / "scores"
SCORES_6K = ROOT / "results" / "scores_v6_supplement"
EVAL_10K = ROOT / "results" / "evaluations_10k"
SCORES_10K = ROOT / "results" / "scores_10k"
REPORT = ROOT / "results" / "analysis" / "v6_merge_report.json"
EXCLUDED = {"llama-3.1-70b-instruct", "phi-4-mini-flash-reasoning"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def merge_dicts(rec_a: list[dict], rec_b: list[dict],
                key: str = "query_id") -> tuple[list[dict], int, int, int]:
    """Merge two record lists keyed on `key`. On conflict prefer rec_a."""
    by_id_a = {r[key]: r for r in rec_a if key in r}
    by_id_b = {r[key]: r for r in rec_b if key in r}
    overlap = set(by_id_a) & set(by_id_b)
    only_a = set(by_id_a) - set(by_id_b)
    only_b = set(by_id_b) - set(by_id_a)
    merged = list(rec_a) + [by_id_b[i] for i in only_b]
    return merged, len(only_a), len(overlap), len(only_b)


def main() -> None:
    EVAL_10K.mkdir(parents=True, exist_ok=True)
    SCORES_10K.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A5: MERGE 4K + 6K → 10K")
    print("=" * 70)

    # Identify which models are present in both batches
    models_4k = {f.stem for f in EVAL_4K.glob("*.jsonl") if f.stem not in EXCLUDED}
    models_6k = {f.stem for f in EVAL_6K.glob("*.jsonl") if f.stem not in EXCLUDED}
    common = sorted(models_4k & models_6k)
    print(f"\n4K models:    {len(models_4k)}")
    print(f"6K models:    {len(models_6k)}")
    print(f"Both:         {len(common)}")
    print(f"Only 4K:      {sorted(models_4k - models_6k)}")
    print(f"Only 6K:      {sorted(models_6k - models_4k)}")

    report = {"per_model": {}, "merge_summary": {}}

    print()
    print(f"{'model':<28} {'4K eval':>8} {'6K eval':>8} "
          f"{'merged':>8} {'4K err%':>8} {'6K err%':>8} {'Δerr':>7}")
    print("-" * 80)

    for m in common:
        # Eval merge (just append non-overlapping)
        eval_4k = load_jsonl(EVAL_4K / f"{m}.jsonl")
        eval_6k = load_jsonl(EVAL_6K / f"{m}.jsonl")
        eval_merged, only4, overlap, only6 = merge_dicts(eval_4k, eval_6k)
        (EVAL_10K / f"{m}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_merged) + "\n",
            encoding="utf-8")

        # Scores merge
        scores_4k = load_jsonl(SCORES_4K / f"{m}.jsonl")
        scores_6k = load_jsonl(SCORES_6K / f"{m}.jsonl")
        sc_merged, sc_only4, sc_overlap, sc_only6 = merge_dicts(scores_4k, scores_6k)
        (SCORES_10K / f"{m}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in sc_merged) + "\n",
            encoding="utf-8")

        # Per-batch error rate (consistency check)
        err_4k = sum(1 for r in scores_4k if r.get("final_score") and r["final_score"] > 0)
        err_6k = sum(1 for r in scores_6k if r.get("final_score") and r["final_score"] > 0)
        n_4k = sum(1 for r in scores_4k if r.get("final_score") is not None)
        n_6k = sum(1 for r in scores_6k if r.get("final_score") is not None)
        eps_4k = err_4k / max(n_4k, 1) if n_4k else 0
        eps_6k = err_6k / max(n_6k, 1) if n_6k else 0
        delta = eps_6k - eps_4k

        print(f"{m:<28} {len(eval_4k):>8} {len(eval_6k):>8} "
              f"{len(eval_merged):>8} {eps_4k*100:>7.1f}% {eps_6k*100:>7.1f}% "
              f"{delta*100:>+6.1f}")

        report["per_model"][m] = {
            "n_eval_4k": len(eval_4k),
            "n_eval_6k": len(eval_6k),
            "n_eval_merged": len(eval_merged),
            "n_scores_4k": n_4k,
            "n_scores_6k": n_6k,
            "n_scores_merged": len(sc_merged),
            "eps_4k": eps_4k,
            "eps_6k": eps_6k,
            "delta_eps": delta,
            "batch_inconsistent": abs(delta) > 0.10,
        }

    inconsistent = [m for m, r in report["per_model"].items()
                    if r["batch_inconsistent"]]
    report["merge_summary"] = {
        "n_models_merged": len(common),
        "n_4k_only": len(models_4k - models_6k),
        "n_6k_only": len(models_6k - models_4k),
        "n_batch_inconsistent": len(inconsistent),
        "inconsistent_models": inconsistent,
    }

    print()
    print(f"Models merged:           {len(common)}")
    print(f"Batch-inconsistent (|Δε| > 10pt): {len(inconsistent)}")
    if inconsistent:
        print(f"  -> {inconsistent}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport -> {REPORT}")


if __name__ == "__main__":
    main()

