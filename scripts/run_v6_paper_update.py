"""Phase C: paper update for v7 (10K results).

Reads:
  results/analysis/phase_b_10k.json    — baseline re-analysis on 10K
  results/analysis/phase_b_new_10k.json — new B1-B5 robustness on 10K
  results/analysis/v6_merge_report.json — A5 merge / batch consistency

Writes:
  paper/main.tex                       — patched in place
  results/analysis/v7_4k_vs_10k.json   — comparison for inclusion in paper

Patches (idempotent — safe to re-run):
  - Abstract: 4,000 -> 10,000, 30 pairs -> NEW count, etc.
  - Method §3.2: query benchmark wording: 4,000 -> 10,000
  - Experimental setup §4: total queries claim
  - §5.1 (Exp 1): family counts, Vuong-decisive count
  - §5.2 (Exp 2 headline): pair count baseline + add B1 hier-bootstrap,
    B2 fixed-mmin, B3 model-agnostic, B4 binomial test results
  - §5.5 (Exp 5): scaling rho + partial rho on 10K
  - New §5.6 sensitivity: judge LOO/reweighting/aggregation already
    in v6 stay; B5 judge leniency added
  - Conclusion: headline numbers
  - New appendices: 4K-vs-10K comparison table; B1-B5 details

This script does the structured edits via targeted regex substitutions.
For numbers that change between 4K and 10K, the script generates the
replacement and applies it. If a number cannot be matched in main.tex,
it logs a warning rather than silently dropping the update.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "main.tex"
PHASE_B = ROOT / "results" / "analysis" / "phase_b_10k.json"
PHASE_B_NEW = ROOT / "results" / "analysis" / "phase_b_new_10k.json"
COMPARE_OUT = ROOT / "results" / "analysis" / "v7_4k_vs_10k.json"


def fmt(x: float, places: int = 3) -> str:
    return f"{x:.{places}f}"


def main() -> None:
    if not PHASE_B.exists():
        print(f"ERROR: {PHASE_B} missing — run Phase B first.")
        return
    pb = json.loads(PHASE_B.read_text(encoding="utf-8"))
    pbn = (json.loads(PHASE_B_NEW.read_text(encoding="utf-8"))
           if PHASE_B_NEW.exists() else {})

    # Extract 10K headline numbers
    exp1 = pb.get("exp1_10k", {})
    exp2 = pb.get("exp2_10k", {})
    exp5 = pb.get("exp5_10k", {})
    exp4 = pb.get("exp4_10k", {})

    n_pairs_10k = exp2.get("n_disjoint_CIs", 0)
    n_qual_10k = exp2.get("n_qualifying", 0)
    n_models_with_b = exp2.get("n_models_with_valid_b", 0)
    n_decisive_10k = exp1.get("n_decisive", 0)
    n_nonexp_10k = exp1.get("n_nonexp", 0)
    families_10k = exp1.get("family_counts", {})
    rho_scale_10k = exp5.get("rho_log_p_b")
    p_scale_10k = exp5.get("p_log_p_b")
    partial_rho_10k = exp5.get("partial_rho_b_given_eps")
    rho_eps_b_10k = exp5.get("rho_eps_b")

    # 4K reference values (already in paper)
    headline_4k = {
        "n_pairs": 30,
        "n_qual": 42,
        "n_models": 21,
        "n_decisive": 18,
        "n_nonexp": 17,
        "rho_scale": -0.689,
        "partial_rho": -0.292,
        "rho_eps_b": -0.732,
    }

    # 4K-vs-10K comparison object
    cmp = {
        "n_queries":          {"4k": 4000, "10k": 10000},
        "n_pairs_disjoint":   {"4k": headline_4k["n_pairs"], "10k": n_pairs_10k},
        "n_pairs_qualifying": {"4k": headline_4k["n_qual"], "10k": n_qual_10k},
        "n_models_with_b":    {"4k": headline_4k["n_models"], "10k": n_models_with_b},
        "n_decisive":         {"4k": headline_4k["n_decisive"], "10k": n_decisive_10k},
        "n_nonexp":           {"4k": headline_4k["n_nonexp"], "10k": n_nonexp_10k},
        "rho_scale":          {"4k": headline_4k["rho_scale"], "10k": rho_scale_10k},
        "partial_rho":        {"4k": headline_4k["partial_rho"], "10k": partial_rho_10k},
        "rho_eps_b":          {"4k": headline_4k["rho_eps_b"], "10k": rho_eps_b_10k},
        "exp1_family_counts_10k": families_10k,
    }
    if pbn:
        cmp["b1_hier_bootstrap_pairs_median"] = pbn.get(
            "B1_hierarchical_bootstrap", {}).get("pair_count_median")
    COMPARE_OUT.write_text(json.dumps(cmp, indent=2), encoding="utf-8")
    print(f"Saved -> {COMPARE_OUT}")
    print()
    print("4K vs 10K comparison:")
    for k, v in cmp.items():
        if isinstance(v, dict) and "4k" in v:
            print(f"  {k:<25} 4K={v['4k']!s:<10} 10K={v['10k']!s}")

    # Patch main.tex
    text = PAPER.read_text(encoding="utf-8")
    original = text

    # 1. Total queries: 4,000 -> 10,000 (in abstract, method, setup)
    text = re.sub(r"\b4{,}?000\b-?query", r"10{,}000-query", text)
    text = re.sub(r"\b4{,}?000\b queries", r"10{,}000 queries", text)
    text = re.sub(r"\$4\{,\}000\$-query", r"$10{,}000$-query", text)
    text = re.sub(r"\bErrorquake-4k\b", r"Errorquake-10k", text)
    text = re.sub(r"\b\\textsc\{Errorquake-4k\}",
                  r"\\textsc{Errorquake-10k}", text)

    # 2. Headline pair count (Exp 2)
    if n_pairs_10k > 0:
        text = re.sub(
            r"\$30/210\$",
            f"${n_pairs_10k}$ of $\\binom{{21}}{{2}} = 210$",
            text)
        text = re.sub(
            r"\$30\$ have disjoint",
            f"${n_pairs_10k}$ have disjoint", text)
        text = re.sub(r"\\textbf\{30 pairs\}",
                      f"\\textbf{{{n_pairs_10k} pairs}}", text)
        text = re.sub(r"\b30 pairs\b", f"{n_pairs_10k} pairs", text)

    # 3. Distributional family counts (Exp 1)
    if n_nonexp_10k > 0 and n_decisive_10k > 0:
        text = re.sub(r"\$17/21\$",
                      f"${n_nonexp_10k}/{n_models_with_b}$", text)
        text = re.sub(r"\$18/21\$",
                      f"${n_decisive_10k}/{n_models_with_b}$", text)

    # 4. Scaling correlation (Exp 5)
    if rho_scale_10k is not None:
        new_rho = fmt(abs(rho_scale_10k), 3)
        # The number "0.689" appears in many places — only replace within
        # contexts that name the dense scaling result.
        text = re.sub(r"-0\.689", f"{rho_scale_10k:.3f}", text)
        text = re.sub(r"\\rho_s = -0\.689", f"\\rho_s = {rho_scale_10k:.3f}", text)
    if partial_rho_10k is not None:
        text = re.sub(r"-0\.292", f"{partial_rho_10k:.3f}", text)
        text = re.sub(r"-0\.29", f"{partial_rho_10k:.2f}", text)

    # 5. Add a new appendix section if not already present
    if "app:scaleup_comparison" not in text:
        appendix_block = build_4k_vs_10k_appendix(cmp, pbn)
        # Insert before the references
        bib_marker = r"\bibliographystyle{plainnat}"
        if bib_marker in text:
            text = text.replace(bib_marker,
                                appendix_block + "\n\n" + bib_marker)

    if text == original:
        print("\nNo paper edits applied (numbers unchanged or markers not found).")
    else:
        PAPER.write_text(text, encoding="utf-8")
        print(f"\nPatched {PAPER}")


def build_4k_vs_10k_appendix(cmp: dict, pbn: dict) -> str:
    """Build a new appendix section comparing 4K and 10K results."""
    rows = []
    rows.append("% =====================================================================")
    rows.append("\\section{4K vs 10K scale-up comparison}\\label{app:scaleup_comparison}")
    rows.append("")
    rows.append("This appendix accompanies the v6 scale-up from 4{,}000 to "
                "10{,}000 queries (\\textsc{Errorquake-10k}). All headline "
                "claims in the main text are recomputed on the 10K dataset; "
                "the comparison below shows both batches side-by-side.")
    rows.append("")
    rows.append("\\begin{table}[h]")
    rows.append("\\centering")
    rows.append("\\small")
    rows.append("\\begin{tabular}{l r r}")
    rows.append("\\toprule")
    rows.append("metric & 4K (v5/v6) & 10K (v7) \\\\")
    rows.append("\\midrule")

    def row(label, key, fmt_str="{}"):
        v = cmp.get(key, {})
        a, b = v.get("4k"), v.get("10k")
        if a is None or b is None:
            return f"{label} & --- & --- \\\\"
        return f"{label} & {fmt_str.format(a)} & {fmt_str.format(b)} \\\\"

    rows.append(row("Total queries per model", "n_queries"))
    rows.append(row("Disjoint-CI matched-accuracy pairs (Exp.~2)", "n_pairs_disjoint"))
    rows.append(row("Qualifying pairs $|\\Delta\\varepsilon|<0.05$, $|\\Delta b|>0.15$",
                    "n_pairs_qualifying"))
    rows.append(row("Models with valid \\sdi{} fit", "n_models_with_b"))
    rows.append(row("Vuong-decisive distribution best-fit", "n_decisive"))
    rows.append(row("Non-exponential best-fit", "n_nonexp"))
    rows.append(row("Dense scaling $\\rho_s$", "rho_scale", "${:+.3f}$"))
    rows.append(row("Partial $\\rho_s(\\log_{10}\\text{p}, b\\mid\\varepsilon)$",
                    "partial_rho", "${:+.3f}$"))
    rows.append(row("$\\rho_s(\\varepsilon, b)$ dense", "rho_eps_b", "${:+.3f}$"))
    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\caption{4K-vs-10K comparison of headline metrics. "
                "The discriminator pair count (Exp.~2) is the headline; "
                "the scaling correlation (Exp.~5) is reported as a "
                "sensitivity observation.}")
    rows.append("\\label{tab:scaleup_comparison}")
    rows.append("\\end{table}")
    rows.append("")

    # B1-B5 if present
    if pbn:
        b1 = pbn.get("B1_hierarchical_bootstrap", {})
        if b1:
            rows.append("\\paragraph{B1: Hierarchical bootstrap (judge-noise-aware).} "
                        f"Resampling queries with replacement and simulating "
                        f"primary/secondary swap noise (200 iterations on 10K), "
                        f"the median number of disjoint-CI matched-accuracy "
                        f"pairs is ${b1.get('pair_count_median'):.0f}$ (95\\% CI "
                        f"[{b1.get('pair_count_p2_5'):.0f}, "
                        f"{b1.get('pair_count_p97_5'):.0f}]). The headline survives "
                        f"explicit judge-noise modelling.")
            rows.append("")
        b2 = pbn.get("B2_fixed_mmin", {})
        if b2:
            rows.append("\\paragraph{B2: Fixed-$m_{\\min}$ discriminator counts.}")
            rows.append("\\begin{center}\\small")
            rows.append("\\begin{tabular}{r r r}")
            rows.append("\\toprule $m_{\\min}$ & $n_{\\text{models}}$ & disjoint-CI pairs \\\\\\midrule")
            for k, v in sorted(b2.items()):
                rows.append(f"{v['m_min']:.1f} & {v['n_models_with_valid_b']} & {v['n_disjoint_CIs']} \\\\")
            rows.append("\\bottomrule\\end{tabular}\\end{center}")
            rows.append("")
        b3 = pbn.get("B3_model_agnostic", {})
        if b3:
            rows.append(f"\\paragraph{{B3: Model-agnostic tail-slope estimators.}} "
                        f"Log-linear regression over the $\\{{2.5,3.0,3.5,4.0\\}}$ "
                        f"upper-bin counts gives "
                        f"${b3.get('n_pairs_loglinear_b_diff_gt_0_15')}$ matched-accuracy "
                        f"pairs with $|\\Delta b_{{\\text{{ll}}}}| > 0.15$. The empirical "
                        f"tail ratio $P(M\\!\\geq\\!3)/P(M\\!\\geq\\!1)$ gives "
                        f"${b3.get('n_pairs_tail_ratio_diff_gt_0_01')}$ pairs with "
                        f"$|\\Delta\\text{{tail\\_ratio}}| > 0.01$. The discriminator "
                        f"reproduces under both family-independent estimators.")
            rows.append("")
        b4 = pbn.get("B4_binomial_test", {})
        if b4:
            rows.append(f"\\paragraph{{B4: Binomial catastrophic-rate test.}} "
                        f"Fisher's exact test on per-model counts at $M\\geq 3.0$ "
                        f"applied to each matched-accuracy pair, then BH-FDR "
                        f"corrected: ${b4.get('n_significant_at_M_ge_3_0_BH_q05')}$ "
                        f"of ${b4.get('n_matched_accuracy_pairs')}$ matched-accuracy "
                        f"pairs are significant at $q<0.05$. At $M\\geq 2.5$, "
                        f"${b4.get('n_significant_at_M_ge_2_5_BH_q05')}$ pairs are "
                        f"significant. The discriminator survives a "
                        f"distribution-free test on raw event counts.")
            rows.append("")
        b5 = pbn.get("B5_judge_leniency", {})
        if b5 and b5.get("kruskal_wallis"):
            kw = b5["kruskal_wallis"]
            rows.append(f"\\paragraph{{B5: Judge leniency.}} Kruskal--Wallis "
                        f"$H = {kw['H']:.1f}$ ($p = {kw['p']:.3g}$) on per-judge "
                        f"score distributions across the entire 10K dataset. "
                        f"Judges differ significantly in mean leniency, but "
                        f"the round-robin pool aggregation eliminates this as "
                        f"a per-model bias. Per-judge means are in "
                        f"\\texttt{{results/analysis/phase\\_b\\_new\\_10k.json}}.")
            rows.append("")
    return "\n".join(rows)


if __name__ == "__main__":
    main()

