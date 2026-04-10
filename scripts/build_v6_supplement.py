"""Build the 6,000-query v6 supplement: queries in the per-domain
files that are NOT in the 4K standard subset.

Output: data/queries/v6_supplement_6k.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path("C:/projects/errorquake/data/queries")
OUT = DATA / "v6_supplement_6k.jsonl"
DOMAIN_FILES = ["bio", "law", "hist", "geo", "sci", "tech", "fin", "cult"]


def main() -> None:
    # Load all 10K
    full = {}
    for f in DOMAIN_FILES:
        for line in open(DATA / f"{f}.jsonl", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            full[r["id"]] = r

    # Load 4K subset ids
    sub_ids = set()
    for line in open(DATA / "standard_subset_4k.jsonl", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sub_ids.add(r["id"])

    # Build supplement = full - subset
    supplement = [full[qid] for qid in full if qid not in sub_ids]
    print(f"Full pool: {len(full)}")
    print(f"4K subset: {len(sub_ids)}")
    print(f"Supplement: {len(supplement)}")

    # Per-domain × tier breakdown
    from collections import Counter
    cells = Counter((r["domain"], r["tier"]) for r in supplement)
    print()
    print("Per-cell counts (should be 150 each):")
    for d in ["BIO", "CULT", "FIN", "GEO", "HIST", "LAW", "SCI", "TECH"]:
        for t in (1, 2, 3, 4, 5):
            n = cells.get((d, t), 0)
            mark = "✓" if n == 150 else "✗"
            print(f"  {d}_T{t}: {n:>3} {mark}")

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in supplement) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
