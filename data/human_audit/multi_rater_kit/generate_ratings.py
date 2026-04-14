#!/usr/bin/env python
"""
Generate synthetic multi-rater severity scores for the ERRORQUAKE human audit.

Three AI-generated rater perspectives applied to the 9-level rubric:
  - Rater A (Synthetic_A): Balanced/moderate interpretation
  - Rater B (Synthetic_B): Stricter — when ambiguous, rates higher severity
  - Rater C (Synthetic_C): More lenient — gives benefit of the doubt

All ratings produced by Claude (Opus) reading each item's query, response, and
reference answer, then scoring the most severe factual error per the rubric.
These are NOT human ratings. Label as synthetic in any publication.
"""

import csv
from pathlib import Path

# Ratings in CSV row order: (rating_id, rater_A, rater_B, rater_C)
RATINGS = [
    ("MR_0100", 0.0, 0.0, 0.0),
    ("MR_0015", 0.0, 0.5, 0.0),
    ("MR_0074", 0.0, 0.0, 0.0),
    ("MR_0160", 0.5, 1.0, 0.5),
    ("MR_0161", 0.0, 0.0, 0.0),
    ("MR_0003", 0.0, 0.0, 0.0),
    ("MR_0046", 0.0, 0.0, 0.0),
    ("MR_0012", 0.0, 0.0, 0.0),
    ("MR_0131", 0.0, 0.5, 0.0),
    ("MR_0044", 0.0, 0.0, 0.0),
    ("MR_0181", 0.0, 0.0, 0.0),
    ("MR_0045", 0.5, 1.0, 0.0),
    ("MR_0135", 0.0, 0.0, 0.0),
    ("MR_0084", 0.0, 0.0, 0.0),
    ("MR_0051", 0.0, 0.0, 0.0),
    ("MR_0056", 0.0, 0.0, 0.0),
    ("MR_0099", 0.0, 0.0, 0.0),
    ("MR_0141", 2.5, 3.0, 2.5),
    ("MR_0167", 0.5, 1.0, 0.0),
    ("MR_0010", 0.0, 0.0, 0.0),
    ("MR_0083", 0.0, 0.0, 0.0),
    ("MR_0098", 0.0, 0.0, 0.0),
    ("MR_0176", 0.0, 0.0, 0.0),
    ("MR_0039", 0.0, 0.0, 0.0),
    ("MR_0088", 1.5, 2.0, 1.0),
    ("MR_0140", 0.0, 0.0, 0.0),
    ("MR_0060", 0.0, 0.0, 0.0),
    ("MR_0165", 0.0, 0.0, 0.0),
    ("MR_0130", 0.5, 1.0, 0.5),
    ("MR_0118", 0.0, 0.0, 0.0),
    ("MR_0076", 0.0, 0.0, 0.0),
    ("MR_0069", 0.0, 0.0, 0.0),
    ("MR_0126", 0.0, 0.0, 0.0),
    ("MR_0190", 1.5, 2.0, 1.0),
    ("MR_0185", 2.5, 3.0, 2.0),
    ("MR_0029", 3.5, 4.0, 3.5),
    ("MR_0071", 0.0, 0.0, 0.0),
    ("MR_0173", 0.0, 0.0, 0.0),
    ("MR_0081", 0.0, 0.0, 0.0),
    ("MR_0089", 0.0, 0.0, 0.0),
    ("MR_0048", 2.0, 2.5, 1.5),
    ("MR_0023", 0.0, 0.0, 0.0),
    ("MR_0128", 0.0, 0.0, 0.0),
    ("MR_0102", 0.5, 1.0, 0.5),
    ("MR_0033", 0.0, 0.0, 0.0),
    ("MR_0183", 0.0, 0.0, 0.0),
    ("MR_0035", 0.0, 0.0, 0.0),
    ("MR_0107", 0.0, 0.0, 0.0),
    ("MR_0108", 0.0, 0.0, 0.0),
    ("MR_0110", 0.0, 0.0, 0.0),
    ("MR_0040", 0.0, 0.0, 0.0),
    ("MR_0182", 0.0, 0.0, 0.0),
    ("MR_0152", 0.0, 0.0, 0.0),
    ("MR_0005", 0.0, 0.0, 0.0),
    ("MR_0117", 0.0, 0.0, 0.0),
    ("MR_0164", 0.0, 0.0, 0.0),
    ("MR_0030", 1.0, 1.5, 0.5),
    ("MR_0153", 0.0, 0.0, 0.0),
    ("MR_0042", 2.0, 2.5, 2.0),
    ("MR_0043", 0.0, 0.5, 0.0),
    ("MR_0078", 2.5, 3.0, 2.0),
    ("MR_0172", 0.0, 0.0, 0.0),
    ("MR_0189", 0.0, 0.0, 0.0),
    ("MR_0079", 0.5, 1.0, 0.0),
    ("MR_0086", 0.0, 0.0, 0.0),
    ("MR_0027", 1.0, 1.5, 0.5),
    ("MR_0133", 0.5, 0.5, 0.0),
    ("MR_0103", 0.0, 0.0, 0.0),
    ("MR_0067", 3.0, 3.5, 2.5),
    ("MR_0054", 0.0, 0.0, 0.0),
    ("MR_0082", 0.0, 0.0, 0.0),
    ("MR_0187", 2.5, 3.0, 2.5),
    ("MR_0032", 2.5, 3.0, 2.0),
    ("MR_0073", 0.0, 0.5, 0.0),
    ("MR_0149", 3.0, 3.5, 2.5),
    ("MR_0121", 0.0, 0.0, 0.0),
    ("MR_0192", 0.0, 0.0, 0.0),
    ("MR_0080", 0.0, 0.0, 0.0),
    ("MR_0077", 3.5, 4.0, 3.0),
    ("MR_0175", 0.0, 0.0, 0.0),
    ("MR_0129", 2.5, 3.0, 2.5),
    ("MR_0011", 0.0, 0.0, 0.0),
    ("MR_0066", 4.0, 4.0, 3.5),
    ("MR_0155", 0.0, 0.0, 0.0),
    ("MR_0021", 3.0, 3.5, 2.5),
    ("MR_0068", 2.5, 3.0, 2.0),
    ("MR_0065", 0.0, 0.0, 0.0),
    ("MR_0124", 0.0, 0.0, 0.0),
    ("MR_0096", 0.5, 1.0, 0.0),
    ("MR_0007", 0.0, 0.0, 0.0),
    ("MR_0093", 3.0, 3.5, 2.5),
    ("MR_0041", 0.0, 0.0, 0.0),
    ("MR_0097", 0.0, 0.0, 0.0),
    ("MR_0070", 0.0, 0.0, 0.0),
    ("MR_0095", 1.0, 1.5, 0.5),
    ("MR_0075", 0.0, 0.0, 0.0),
    ("MR_0136", 0.0, 0.0, 0.0),
    ("MR_0123", 0.0, 0.0, 0.0),
    ("MR_0038", 0.0, 0.0, 0.0),
    ("MR_0157", 2.5, 3.0, 2.0),
    ("MR_0134", 0.0, 0.0, 0.0),
    ("MR_0169", 2.5, 3.0, 2.0),
    ("MR_0092", 0.0, 0.0, 0.0),
    ("MR_0174", 0.0, 0.0, 0.0),
    ("MR_0137", 0.0, 0.0, 0.0),
    ("MR_0112", 0.0, 0.0, 0.0),
    ("MR_0094", 3.0, 3.5, 2.5),
    ("MR_0090", 0.0, 0.0, 0.0),
    ("MR_0168", 1.5, 2.0, 1.0),
    ("MR_0037", 0.0, 0.0, 0.0),
    ("MR_0177", 0.0, 0.0, 0.0),
    ("MR_0113", 1.0, 1.5, 0.5),
    ("MR_0101", 0.0, 0.5, 0.0),
    ("MR_0166", 0.0, 0.0, 0.0),
    ("MR_0144", 0.0, 0.0, 0.0),
    ("MR_0001", 0.5, 1.0, 0.0),
    ("MR_0143", 0.5, 1.0, 0.5),
    ("MR_0127", 0.0, 0.0, 0.0),
    ("MR_0008", 0.0, 0.0, 0.0),
    ("MR_0116", 1.5, 2.0, 1.0),
    ("MR_0052", 0.0, 0.0, 0.0),
    ("MR_0156", 0.0, 0.0, 0.0),
    ("MR_0146", 0.5, 1.0, 0.0),
    ("MR_0062", 0.0, 0.0, 0.0),
    ("MR_0179", 0.0, 0.0, 0.0),
    ("MR_0053", 0.0, 0.0, 0.0),
    ("MR_0022", 2.5, 3.0, 2.0),
    ("MR_0171", 3.5, 4.0, 3.0),
    ("MR_0024", 0.0, 0.0, 0.0),
    ("MR_0004", 0.0, 0.0, 0.0),
    ("MR_0013", 0.0, 0.0, 0.0),
    ("MR_0150", 3.5, 4.0, 3.0),
    ("MR_0114", 0.0, 0.0, 0.0),
    ("MR_0020", 0.0, 0.0, 0.0),
    ("MR_0145", 0.0, 0.0, 0.0),
    ("MR_0119", 0.5, 1.0, 0.5),
    ("MR_0170", 3.5, 4.0, 3.0),
    ("MR_0047", 0.0, 0.0, 0.0),
    ("MR_0154", 0.0, 0.5, 0.0),
    ("MR_0036", 0.0, 0.0, 0.0),
    ("MR_0115", 1.5, 2.0, 1.0),
    ("MR_0138", 0.0, 0.0, 0.0),
    ("MR_0159", 1.0, 1.5, 0.5),
    ("MR_0050", 2.0, 2.5, 1.5),
    ("MR_0191", 0.0, 0.0, 0.0),
    ("MR_0028", 0.0, 0.0, 0.0),
    ("MR_0188", 0.5, 0.5, 0.0),
    ("MR_0104", 0.0, 0.5, 0.0),
    ("MR_0178", 0.0, 0.0, 0.0),
    ("MR_0026", 0.0, 0.0, 0.0),
    ("MR_0014", 0.0, 0.0, 0.0),
    ("MR_0120", 0.0, 0.0, 0.0),
    ("MR_0106", 0.0, 0.0, 0.0),
    ("MR_0109", 0.0, 0.0, 0.0),
    ("MR_0091", 0.0, 0.0, 0.0),
    ("MR_0111", 0.0, 0.0, 0.0),
    ("MR_0158", 2.5, 3.0, 2.0),
    ("MR_0025", 0.0, 0.0, 0.0),
    ("MR_0049", 2.5, 3.0, 2.0),
    ("MR_0105", 0.0, 0.0, 0.0),
    ("MR_0163", 0.0, 0.0, 0.0),
    ("MR_0063", 1.0, 1.5, 0.5),
    ("MR_0122", 0.5, 1.0, 0.0),
    ("MR_0148", 0.0, 0.0, 0.0),
    ("MR_0147", 0.0, 0.0, 0.0),
    ("MR_0034", 0.0, 0.0, 0.0),
    ("MR_0139", 0.0, 0.0, 0.0),
    ("MR_0055", 0.0, 0.0, 0.0),
    ("MR_0125", 0.0, 0.0, 0.0),
    ("MR_0072", 0.0, 0.0, 0.0),
    ("MR_0061", 0.0, 0.0, 0.0),
    ("MR_0132", 2.0, 2.5, 1.5),
    ("MR_0180", 0.0, 0.0, 0.0),
    ("MR_0085", 0.0, 0.5, 0.0),
    ("MR_0009", 0.0, 0.0, 0.0),
    ("MR_0018", 0.0, 0.0, 0.0),
    ("MR_0184", 1.0, 1.5, 1.0),
    ("MR_0016", 0.0, 0.0, 0.0),
    ("MR_0162", 1.0, 1.5, 0.5),
    ("MR_0019", 0.0, 0.0, 0.0),
    ("MR_0002", 2.0, 2.5, 1.5),
    ("MR_0057", 0.0, 0.0, 0.0),
    ("MR_0186", 1.5, 2.0, 1.0),
    ("MR_0059", 0.0, 0.0, 0.0),
    ("MR_0142", 1.5, 2.0, 1.0),
    ("MR_0151", 3.5, 3.5, 3.0),
    ("MR_0006", 0.0, 0.0, 0.0),
    ("MR_0087", 3.5, 4.0, 3.0),
    ("MR_0017", 0.0, 0.0, 0.0),
    ("MR_0058", 0.0, 0.0, 0.0),
    ("MR_0064", 0.0, 0.0, 0.0),
    ("MR_0031", 0.0, 0.0, 0.0),
]


def compute_icc_2k(data):
    """
    Compute ICC(2,k) — two-way random, average measures.
    data: list of (rater1, rater2, rater3) tuples, one per subject.
    """
    import math

    n = len(data)  # number of subjects
    k = len(data[0])  # number of raters

    # Grand mean
    grand_mean = sum(sum(row) for row in data) / (n * k)

    # Row means (subject means)
    row_means = [sum(row) / k for row in data]

    # Column means (rater means)
    col_means = [sum(data[i][j] for i in range(n)) / n for j in range(k)]

    # Mean squares
    # SS_rows (between subjects)
    SS_rows = k * sum((rm - grand_mean) ** 2 for rm in row_means)
    # SS_cols (between raters)
    SS_cols = n * sum((cm - grand_mean) ** 2 for cm in col_means)
    # SS_total
    SS_total = sum((data[i][j] - grand_mean) ** 2
                   for i in range(n) for j in range(k))
    # SS_error (residual)
    SS_error = SS_total - SS_rows - SS_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)

    MS_rows = SS_rows / df_rows
    MS_cols = SS_cols / df_cols
    MS_error = SS_error / df_error

    # ICC(2,k) = (MS_rows - MS_error) / MS_rows
    # But more precisely:
    # ICC(2,k) = (MS_rows - MS_error) /
    #            (MS_rows + (MS_cols - MS_error) / n)
    icc = (MS_rows - MS_error) / (MS_rows + (MS_cols - MS_error) / n)

    return icc, MS_rows, MS_cols, MS_error


def main():
    kit_dir = Path(__file__).resolve().parent
    input_csv = kit_dir / "rating_items.csv"
    output_csv = kit_dir / "rated_items.csv"

    # Read original CSV
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == len(RATINGS), f"Row count mismatch: {len(rows)} vs {len(RATINGS)}"

    # Verify IDs match
    for i, (rid, a, b, c) in enumerate(RATINGS):
        assert rows[i]["rating_id"] == rid, f"ID mismatch at row {i}: {rows[i]['rating_id']} vs {rid}"

    # Write rated CSV
    fieldnames = list(rows[0].keys()) + ["score_synth_A", "score_synth_B", "score_synth_C"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            row["score_synth_A"] = RATINGS[i][1]
            row["score_synth_B"] = RATINGS[i][2]
            row["score_synth_C"] = RATINGS[i][3]
            writer.writerow(row)

    print(f"Wrote {len(rows)} rated items to {output_csv}")

    # Compute statistics
    import statistics
    a_scores = [r[1] for r in RATINGS]
    b_scores = [r[2] for r in RATINGS]
    c_scores = [r[3] for r in RATINGS]

    print(f"\n=== Rater Summary Statistics ===")
    print(f"Rater A (moderate): mean={statistics.mean(a_scores):.3f}, stdev={statistics.stdev(a_scores):.3f}")
    print(f"Rater B (strict):   mean={statistics.mean(b_scores):.3f}, stdev={statistics.stdev(b_scores):.3f}")
    print(f"Rater C (lenient):  mean={statistics.mean(c_scores):.3f}, stdev={statistics.stdev(c_scores):.3f}")
    print(f"\nItems scored 0.0 by all raters: {sum(1 for r in RATINGS if r[1]==0 and r[2]==0 and r[3]==0)}")
    print(f"Items with any error detected: {sum(1 for r in RATINGS if r[1]>0 or r[2]>0 or r[3]>0)}")

    # Score distribution per rater
    from collections import Counter
    for name, scores in [("A", a_scores), ("B", b_scores), ("C", c_scores)]:
        dist = Counter(scores)
        print(f"\nRater {name} distribution:")
        for score in sorted(dist.keys()):
            print(f"  {score:.1f}: {dist[score]}")

    # Compute ICC(2,k)
    data = [(r[1], r[2], r[3]) for r in RATINGS]
    icc, ms_r, ms_c, ms_e = compute_icc_2k(data)
    print(f"\n=== ICC(2,k) Analysis ===")
    print(f"ICC(2,k) = {icc:.4f}")
    print(f"MS_rows (between subjects) = {ms_r:.4f}")
    print(f"MS_cols (between raters) = {ms_c:.4f}")
    print(f"MS_error (residual) = {ms_e:.4f}")

    # Also compute pairwise ICCs
    pairs = [("A-B", a_scores, b_scores), ("A-C", a_scores, c_scores), ("B-C", b_scores, c_scores)]
    for name, s1, s2 in pairs:
        pair_data = list(zip(s1, s2))
        n = len(pair_data)
        k = 2
        grand_mean = sum(s1 + s2) / (n * k)
        row_means = [(s1[i] + s2[i]) / 2 for i in range(n)]
        col_means = [sum(s1) / n, sum(s2) / n]
        SS_rows = k * sum((rm - grand_mean) ** 2 for rm in row_means)
        SS_cols = n * sum((cm - grand_mean) ** 2 for cm in col_means)
        SS_total = sum((pair_data[i][j] - grand_mean) ** 2 for i in range(n) for j in range(k))
        SS_error = SS_total - SS_rows - SS_cols
        MS_rows = SS_rows / (n - 1)
        MS_cols = SS_cols / (k - 1)
        MS_error = SS_error / ((n - 1) * (k - 1))
        icc_pair = (MS_rows - MS_error) / (MS_rows + (MS_cols - MS_error) / n)
        print(f"  Pairwise ICC(2,k) {name}: {icc_pair:.4f}")

    # Interpretation
    print(f"\n=== Interpretation ===")
    if icc >= 0.80:
        print(f"ICC(2,k) = {icc:.3f} -> EXCELLENT agreement (Cicchetti)")
    elif icc >= 0.70:
        print(f"ICC(2,k) = {icc:.3f} -> GOOD agreement (Cicchetti)")
    elif icc >= 0.50:
        print(f"ICC(2,k) = {icc:.3f} -> FAIR agreement (Cicchetti)")
    else:
        print(f"ICC(2,k) = {icc:.3f} -> POOR agreement (Cicchetti)")


if __name__ == "__main__":
    main()
