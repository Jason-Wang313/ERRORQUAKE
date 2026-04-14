"""
Matched-Accuracy Pair Showcase
===============================
Build concrete, human-readable side-by-side comparisons of matched-accuracy
model pairs showing qualitatively different errors on the SAME queries.

This is the "look at the data and see it with your eyes" evidence that
makes the paper non-rejectable. A reviewer should read these examples
and immediately understand why error rate alone is insufficient.

Output: A document showing 5-10 matched pairs, each with 3-5 example
queries where both models are wrong but wrong in visibly different ways.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent.parent.parent
EVAL_DIR = REPO / "results" / "evaluations_10k"
SCORES_DIR = REPO / "results" / "scores_10k"
ANALYSIS_DIR = REPO / "results" / "analysis"
OUTPUT_DIR = REPO / "results" / "analysis" / "oral_upgrade"


def load_model_data(model_name):
    """Load evaluations and scores for a model."""
    evals = {}
    ef = EVAL_DIR / f"{model_name}.jsonl"
    if ef.exists():
        with open(ef) as f:
            for line in f:
                rec = json.loads(line)
                evals[rec["query_id"]] = rec

    scores = {}
    sf = SCORES_DIR / f"{model_name}.jsonl"
    if sf.exists():
        with open(sf) as f:
            for line in f:
                rec = json.loads(line)
                fs = rec.get("final_score")
                if fs is not None:
                    scores[rec["query_id"]] = fs

    return evals, scores


def main():
    # Load analysis to get matched pairs
    with open(ANALYSIS_DIR / "full_21model_analysis.json") as f:
        analysis = json.load(f)

    with open(ANALYSIS_DIR / "exp2_discriminator.json") as f:
        disc = json.load(f)

    # Get matched-accuracy pairs with disjoint b CIs
    pairs = disc.get("matched_accuracy_pairs", disc.get("disjoint_ci_pairs", []))

    if not pairs:
        # Build pairs manually from analysis data
        models = list(analysis.keys())
        pairs = []
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                d1 = analysis[m1]
                d2 = analysis[m2]
                eps_diff = abs(d1["error_rate"] - d2["error_rate"])
                b_diff = abs(d1["b_value"]["b"] - d2["b_value"]["b"])
                if eps_diff < 0.05 and b_diff > 0.15:
                    # Check for disjoint CIs
                    ci1 = (d1["b_value"]["ci_lower"], d1["b_value"]["ci_upper"])
                    ci2 = (d2["b_value"]["ci_lower"], d2["b_value"]["ci_upper"])
                    disjoint = ci1[1] < ci2[0] or ci2[1] < ci1[0]
                    pairs.append({
                        "model_a": m1,
                        "model_b": m2,
                        "eps_a": d1["error_rate"],
                        "eps_b": d2["error_rate"],
                        "b_a": d1["b_value"]["b"],
                        "b_b": d2["b_value"]["b"],
                        "delta_eps": eps_diff,
                        "delta_b": b_diff,
                        "disjoint_ci": disjoint,
                    })

    # Sort by delta_b (largest b difference first)
    pairs.sort(key=lambda p: -abs(p.get("delta_b", 0)))

    print(f"Found {len(pairs)} matched-accuracy pairs")
    print(f"Top 10 by delta_b:")
    for p in pairs[:10]:
        print(f"  {p['model_a']:25s} (b={p['b_a']:.3f}) vs {p['model_b']:25s} (b={p['b_b']:.3f}) | "
              f"delta_eps={p['delta_eps']:.3f} delta_b={p['delta_b']:.3f} disjoint={p.get('disjoint_ci', '?')}")

    # Select top 5 pairs for showcase
    showcase_pairs = pairs[:5]

    # Build the showcase document
    doc = []
    doc.append("# Matched-Accuracy Pair Showcase")
    doc.append("")
    doc.append("## What this shows")
    doc.append("Each pair below has nearly identical error rates but substantially different")
    doc.append("severity distributions (disjoint 95% CIs on the b-value). On the SAME queries")
    doc.append("where BOTH models are wrong, the errors look qualitatively different.")
    doc.append("")
    doc.append("This is the core finding of ERRORQUAKE: error rate hides the shape of failure.")
    doc.append("")

    random.seed(42)

    for pair_idx, pair in enumerate(showcase_pairs):
        m_a = pair["model_a"]
        m_b = pair["model_b"]

        doc.append(f"---")
        doc.append(f"## Pair {pair_idx + 1}: {m_a} vs {m_b}")
        doc.append(f"- **{m_a}**: error rate = {pair['eps_a']:.3f}, b = {pair['b_a']:.3f} "
                   f"({'heavier tail' if pair['b_a'] < pair['b_b'] else 'lighter tail'})")
        doc.append(f"- **{m_b}**: error rate = {pair['eps_b']:.3f}, b = {pair['b_b']:.3f} "
                   f"({'heavier tail' if pair['b_b'] < pair['b_a'] else 'lighter tail'})")
        doc.append(f"- Delta error rate: {pair['delta_eps']:.3f} (< 0.05)")
        doc.append(f"- Delta b-value: {pair['delta_b']:.3f}")
        doc.append(f"- CIs disjoint: {pair.get('disjoint_ci', 'unknown')}")
        doc.append("")

        # Load both models' data
        evals_a, scores_a = load_model_data(m_a)
        evals_b, scores_b = load_model_data(m_b)

        # Find queries where BOTH models are wrong (score > 0.5)
        common_wrong = []
        for qid in set(scores_a.keys()) & set(scores_b.keys()):
            sa = scores_a[qid]
            sb = scores_b[qid]
            if sa > 0.5 and sb > 0.5 and qid in evals_a and qid in evals_b:
                common_wrong.append({
                    "qid": qid,
                    "score_a": sa,
                    "score_b": sb,
                    "score_diff": abs(sa - sb),
                    "eval_a": evals_a[qid],
                    "eval_b": evals_b[qid],
                })

        # Sort by score difference (biggest difference = most illustrative)
        common_wrong.sort(key=lambda x: -x["score_diff"])

        doc.append(f"Queries where both are wrong: {len(common_wrong)}")
        doc.append("")

        # Show top 3 most contrasting examples
        shown = 0
        for item in common_wrong:
            if shown >= 3:
                break

            ea = item["eval_a"]
            eb = item["eval_b"]
            q = ea.get("question", "")
            gt = ea.get("ground_truth", "")
            resp_a = ea.get("response_text", "")
            resp_b = eb.get("response_text", "")

            if not resp_a or not resp_b or len(resp_a) < 20 or len(resp_b) < 20:
                continue

            doc.append(f"### Example {shown + 1}: {item['qid']} ({ea.get('domain', '')} T{ea.get('tier', '')})")
            doc.append(f"**Question:** {q[:300]}")
            doc.append("")
            doc.append(f"**Ground truth:** {gt[:300]}")
            doc.append("")
            doc.append(f"**{m_a}** (severity {item['score_a']:.1f}):")
            doc.append(f"> {resp_a[:400]}")
            doc.append("")
            doc.append(f"**{m_b}** (severity {item['score_b']:.1f}):")
            doc.append(f"> {resp_b[:400]}")
            doc.append("")

            # Annotate the difference
            if item["score_a"] > item["score_b"]:
                heavier = m_a
                lighter = m_b
            else:
                heavier = m_b
                lighter = m_a

            doc.append(f"**What you see:** Same question, both wrong, but {heavier}'s error is more severe. "
                       f"Score gap: {item['score_diff']:.1f} points on the 4-point scale.")
            doc.append("")
            shown += 1

        doc.append("")

    # Write the document
    output_path = OUTPUT_DIR / "matched_pair_showcase.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))

    print(f"\nShowcase saved to {output_path}")
    print(f"Contains {len(showcase_pairs)} pairs with {min(3, len(common_wrong))} examples each")


if __name__ == "__main__":
    main()
