"""Apply human rater scores to the 100-item human rating form.

Rater: Claude (acting as human rater per instructions)
Scoring method: Read question, ground truth, model response. Identify all
factual claims. Compare to ground truth. Score based on MOST SEVERE error.
"""

import csv
import json
from pathlib import Path

PILOT_DIR = Path("C:/projects/errorquake/data/pilot")

# ============================================================
# SCORES: (rating_id, score_11pt, score_7pt, score_5level, notes)
# ============================================================

RATINGS = [
    # HR_001: Bacterial growth curve stationary phase. Correct.
    ("HR_001", 0.0, 0.0, 0, ""),
    # HR_002: Right-hand rule. Correct (mentions multiple valid names).
    ("HR_002", 0.0, 0.0, 0, ""),
    # HR_003: Salmonella Typhi reservoir = humans. Correct.
    ("HR_003", 0.0, 0.0, 0, ""),
    # HR_004: Hamlet = Shakespeare. Correct.
    ("HR_004", 0.0, 0.0, 0, ""),
    # HR_005: Packet switching paper = Paul Baran. Correct.
    ("HR_005", 0.0, 0.0, 0, ""),
    # HR_006: Polycythemia vera. Correct.
    ("HR_006", 0.0, 0.0, 0, ""),
    # HR_007: Svalbard = Arctic climate zone (correct core). Mentions BWk
    # (cold desert) alongside ET (tundra) — BWk not applicable to Svalbard.
    ("HR_007", 0.5, 0.5, 0, "Core answer Arctic/ET correct; BWk classification is inapplicable to Svalbard"),
    # HR_008: HBV chronic infection mechanism. Correct explanation.
    ("HR_008", 0.0, 0.0, 0, ""),
    # HR_009: Asset allocation. Correct.
    ("HR_009", 0.0, 0.0, 0, ""),
    # HR_010: CheckMate 227 HR=0.79 correct. CI uses 97.72% not 95% —
    # different confidence level (actually used in the trial's pre-specified alpha).
    ("HR_010", 0.5, 0.5, 0, "HR correct; CI uses 97.72% confidence level vs GT 95%, both are valid reported values"),
    # HR_011: GT says 'representative fraction'. Response says 'map scale' —
    # generic term, not the specific technical name asked for.
    ("HR_011", 1.0, 1.0, 1, "Generic term 'map scale' instead of specific 'representative fraction'"),
    # HR_012: Acropolis Museum = 2009. Correct.
    ("HR_012", 0.0, 0.0, 0, ""),
    # HR_013: GT = 'cartographic generalization'. Response = 'Simplification'.
    # Simplification is one component of generalization, not the full term.
    ("HR_013", 1.5, 2.0, 2, "Gives a sub-process (simplification) instead of the overarching term (cartographic generalization)"),
    # HR_014: Prasugrel targets P2Y12. Correct.
    ("HR_014", 0.0, 0.0, 0, ""),
    # HR_015: Constantinople fell 1453. Correct.
    ("HR_015", 0.0, 0.0, 0, ""),
    # HR_016: Divine Right = England, France, Spain. GT says primarily England
    # and France. Spain also used divine right concepts — historically defensible.
    ("HR_016", 0.0, 0.0, 0, "Including Spain is historically defensible"),
    # HR_017: CFPB purpose. Correct.
    ("HR_017", 0.0, 0.0, 0, ""),
    # HR_018: Divine Right = England, France and others. Correct.
    ("HR_018", 0.0, 0.0, 0, ""),
    # HR_019: Arbitrage definition. Correct.
    ("HR_019", 0.0, 0.0, 0, ""),
    # HR_020: Electron wave nature experiment. Response leads with 'Double-Slit
    # Experiment' by de Broglie (1927) — de Broglie proposed the theory (1924),
    # didn't perform the experiment. Electron double-slit came much later
    # (Jönsson 1961). Does mention Davisson-Germer correctly but buries it.
    ("HR_020", 1.5, 2.0, 2, "Incorrectly attributes electron double-slit experiment to de Broglie; correct answer (Davisson-Germer) mentioned but secondary"),
    # HR_021: Bitcoin = SHA-256 hashing. Correct.
    ("HR_021", 0.0, 0.0, 0, ""),
    # HR_022: Ørsted 1820. Exact match with GT.
    ("HR_022", 0.0, 0.0, 0, ""),
    # HR_023: Quran. Correct.
    ("HR_023", 0.0, 0.0, 0, ""),
    # HR_024: Zero = ancient India. Correct.
    ("HR_024", 0.0, 0.0, 0, ""),
    # HR_025: CFPB HPML appraisal disclosure. Correct core.
    ("HR_025", 0.0, 0.0, 0, ""),
    # HR_026: SYN flood description. Correct (says DDoS, which is a valid form).
    ("HR_026", 0.0, 0.0, 0, ""),
    # HR_027: Tibetan Plateau climate. GT = alpine/highland. Response = 'cold
    # and semi-arid to arid, ET or BWk'. Misses the alpine/highland classification;
    # BWk (cold desert) is inaccurate for much of the plateau.
    ("HR_027", 1.0, 1.0, 1, "Uses 'semi-arid to arid' and BWk instead of alpine/highland; core area partially correct"),
    # HR_028: Foucault pendulum. Correct.
    ("HR_028", 0.0, 0.0, 0, ""),
    # HR_029: Commercial speech case. Response = 'Branding Iron Co. v. Stevens
    # (1990)' — this case does not exist. GT = Virginia State Board of Pharmacy.
    ("HR_029", 3.0, 3.0, 3, "Fabricated case name — 'Branding Iron Co. v. Stevens' does not exist"),
    # HR_030: Spontaneous emission. Correct.
    ("HR_030", 0.0, 0.0, 0, ""),
    # HR_031: Quantum tunneling explanation. Correct.
    ("HR_031", 0.0, 0.0, 0, ""),
    # HR_032: Measurement uncertainty implications. Correct.
    ("HR_032", 0.0, 0.0, 0, ""),
    # HR_033: Thousand Islands = 1,864. Exact match.
    ("HR_033", 0.0, 0.0, 0, ""),
    # HR_034: Learned treatise FRE 803(18). Correct.
    ("HR_034", 0.0, 0.0, 0, ""),
    # HR_035: Restatement Third Torts sections. GT=47, response=16. Wrong number.
    ("HR_035", 2.0, 2.0, 2, "States 16 sections; GT says 47"),
    # HR_036: Fruit of poisonous tree. Response names Silverthorne Lumber (1920)
    # then Wong Sun (1963). Both are legitimate — Silverthorne established the
    # principle, Wong Sun the phrase. More complete than GT.
    ("HR_036", 0.0, 0.0, 0, "Response is more complete than GT — Silverthorne established concept, Wong Sun the term"),
    # HR_037: Tabula rasa = John Locke. Correct.
    ("HR_037", 0.0, 0.0, 0, ""),
    # HR_038: ECMAScript sort algorithm. GT = SortCompare. Response =
    # SortIndexedProperties — a real abstract operation in the spec but is the
    # outer sorting operation, not the comparison operation. Wrong answer.
    ("HR_038", 2.0, 2.0, 2, "SortIndexedProperties is real but is the outer sort operation, not the comparison (SortCompare)"),
    # HR_039: Same question. Response = 'native_sort' — does not exist in spec.
    ("HR_039", 3.0, 3.0, 3, "Fabricated algorithm name 'native_sort' — not in ECMAScript spec"),
    # HR_040: Financial ratios. Correct.
    ("HR_040", 0.0, 0.0, 0, ""),
    # HR_041: Friends = New York City. Correct.
    ("HR_041", 0.0, 0.0, 0, ""),
    # HR_042: Treaty of Westphalia outcomes. Correct.
    ("HR_042", 0.0, 0.0, 0, ""),
    # HR_043: Détente definition. Correct.
    ("HR_043", 0.0, 0.0, 0, ""),
    # HR_044: FRE 609 prior convictions. Response includes both categories
    # (>1 year crimes and dishonesty crimes). Accurate and complete.
    ("HR_044", 0.0, 0.0, 0, ""),
    # HR_045: Data warehouse role. Correct.
    ("HR_045", 0.0, 0.0, 0, ""),
    # HR_046: Periodic table = Mendeleev 1869. Correct.
    ("HR_046", 0.0, 0.0, 0, ""),
    # HR_047: Nosferatu = F.W. Murnau. Correct.
    ("HR_047", 0.0, 0.0, 0, ""),
    # HR_048: Noether's theorem. Correct with good examples.
    ("HR_048", 0.0, 0.0, 0, ""),
    # HR_049: Mamluk Sultan at Marj Dabiq = Al-Ashraf Qansuh al-Ghawri. Correct.
    ("HR_049", 0.0, 0.0, 0, ""),
    # HR_050: Telegraph = Samuel Morse (correct). Adds 'Charles Thomas' as
    # co-inventor — no well-known associate by this name; likely confusion.
    ("HR_050", 1.0, 1.0, 1, "Core answer (Morse) correct; 'Charles Thomas' as co-inventor is not a recognized associate — possibly confused with Leonard Gale or Alfred Vail's contributions"),
    # HR_051: Hedging in derivatives. Correct.
    ("HR_051", 0.0, 0.0, 0, ""),
    # HR_052: Periodic table = Mendeleev 1869. Exact match.
    ("HR_052", 0.0, 0.0, 0, ""),
    # HR_053: Soviet heaviest tank losses. GT = Battle of Prokhorovka.
    # Response = Battle of Kursk. Prokhorovka was part of Kursk, but the
    # specific question is about the single battle with heaviest losses.
    ("HR_053", 1.5, 2.0, 2, "Gives the broader campaign (Kursk) not the specific battle (Prokhorovka) as asked"),
    # HR_054: Janus v. AFSCME. Correct holding described.
    ("HR_054", 0.0, 0.0, 0, ""),
    # HR_055: French Revolution causes. Correct.
    ("HR_055", 0.0, 0.0, 0, ""),
    # HR_056: Southernmost SA latitude = 55°59'S ≈ 55.983°S. Matches GT.
    ("HR_056", 0.0, 0.0, 0, ""),
    # HR_057: Dodd-Frank = $50 billion threshold. Correct.
    ("HR_057", 0.0, 0.0, 0, ""),
    # HR_058: Battle of Algiers = Gillo Pontecorvo. Correct.
    ("HR_058", 0.0, 0.0, 0, ""),
    # HR_059: AVX-512 VNNI. Correct, with valid detail about Cascade Lake.
    ("HR_059", 0.0, 0.0, 0, ""),
    # HR_060: Same VNNI question. Core answer (AVX-512 VNNI) present but
    # response is confused/rambling, incorrectly claims VNNI is in VP2INTERSECT
    # and VBMI2 extensions.
    ("HR_060", 1.0, 1.0, 1, "Core answer correct but adds incorrect claims about VP2INTERSECT and VBMI2 containing VNNI"),
    # HR_061: Council of Trent = 16th century. Correct.
    ("HR_061", 0.0, 0.0, 0, ""),
    # HR_062: HBV chronic infection. Correct mechanism described.
    ("HR_062", 0.0, 0.0, 0, ""),
    # HR_063: Omicron RBD substitutions. Key RBD mutations correctly listed
    # but includes non-RBD mutations (D614G, H655Y, N679K, T547K) in the
    # RBD list — those are in other spike domains.
    ("HR_063", 1.0, 1.0, 1, "Key RBD mutations correct but lists several non-RBD spike mutations (D614G, H655Y, N679K) as RBD substitutions"),
    # HR_064: CFPB purpose. Correct.
    ("HR_064", 0.0, 0.0, 0, ""),
    # HR_065: Brain imaging = MRI. Correct.
    ("HR_065", 0.0, 0.0, 0, ""),
    # HR_066: US v. Nixon = Chief Justice Warren Burger. Correct.
    ("HR_066", 0.0, 0.0, 0, ""),
    # HR_067: FRE 703. GT = 'reasonably relied upon by experts in the field'.
    # Response paraphrases as 'reliable facts/data' and 'reliable foundation' —
    # loses the specific legal standard (type reasonably relied upon).
    ("HR_067", 1.5, 2.0, 2, "Paraphrases the standard incorrectly — FRE 703 requires 'reasonably relied upon by experts in the field', not general reliability"),
    # HR_068: Klopfer v. NC. GT = 6th Amend speedy trial applies to states
    # via 14th Amend. Response fabricates a 'Klopfer Rule' about delay and
    # incorrectly mentions waiver of right to counsel.
    ("HR_068", 2.0, 2.0, 2, "Fabricates 'Klopfer Rule'; mixes up holding — case is about incorporation of speedy trial right, not waiver of counsel"),
    # HR_069: DCF model. Correct.
    ("HR_069", 0.0, 0.0, 0, ""),
    # HR_070: Congress of Troppau = Congress System. GT says Concert of Europe.
    # Congress System and Concert of Europe are closely related —
    # the congresses were the mechanism of the Concert. Defensible answer.
    ("HR_070", 0.0, 0.0, 0, "Congress System is the series of meetings within the Concert of Europe framework — closely related terms"),
    # HR_071: A Midsummer Night's Dream = Shakespeare. Correct.
    ("HR_071", 0.0, 0.0, 0, ""),
    # HR_072: Mercury perihelion precession = 43 arcseconds/century. Correct.
    ("HR_072", 0.0, 0.0, 0, ""),
    # HR_073: Cash conversion cycle. Correct.
    ("HR_073", 0.0, 0.0, 0, ""),
    # HR_074: Bogomilism = Bulgaria / First Bulgarian Empire. Correct.
    ("HR_074", 0.0, 0.0, 0, ""),
    # HR_075: ERBB2 gene. Correct.
    ("HR_075", 0.0, 0.0, 0, ""),
    # HR_076: Oratorio = late Renaissance / early Baroque. Accurate.
    ("HR_076", 0.0, 0.0, 0, ""),
    # HR_077: Panic of 1873 origin. GT = Austria and Germany. Response = US.
    # Wrong — started with Vienna Stock Exchange crash May 1873, then spread
    # to US (Jay Cooke & Co. Sept 1873).
    ("HR_077", 2.0, 2.0, 2, "Started in Austria/Germany, not the US — response gets the origin country wrong"),
    # HR_078: Thousand Islands = 1,864. Exact match.
    ("HR_078", 0.0, 0.0, 0, ""),
    # HR_079: Southernmost SA = 55°58'48"S ≈ 55.98°S. Matches GT.
    ("HR_079", 0.0, 0.0, 0, ""),
    # HR_080: Beta-lactam mechanism. Correct — PBP binding, peptidoglycan.
    ("HR_080", 0.0, 0.0, 0, ""),
    # HR_081: Compiler vs interpreter. Correct.
    ("HR_081", 0.0, 0.0, 0, ""),
    # HR_082: FSB = 2009. Correct (with useful FSF context).
    ("HR_082", 0.0, 0.0, 0, ""),
    # HR_083: DCF model. Correct.
    ("HR_083", 0.0, 0.0, 0, ""),
    # HR_084: Too Big to Fail. Correct.
    ("HR_084", 0.0, 0.0, 0, ""),
    # HR_085: Historical linguistics. Correct.
    ("HR_085", 0.0, 0.0, 0, ""),
    # HR_086: DNS = translates domain names to IPs. Correct.
    ("HR_086", 0.0, 0.0, 0, ""),
    # HR_087: Dystrophin = structural support, cytoskeleton-ECM link. Correct.
    ("HR_087", 0.0, 0.0, 0, ""),
    # HR_088: Tabula rasa = John Locke. Correct.
    ("HR_088", 0.0, 0.0, 0, ""),
    # HR_089: Klangfarbenmelodie = Arnold Schoenberg. Correct.
    ("HR_089", 0.0, 0.0, 0, ""),
    # HR_090: Harper Lee = To Kill a Mockingbird. Correct.
    ("HR_090", 0.0, 0.0, 0, ""),
    # HR_091: SARs in China. Response correctly says only HK and Macau are SARs.
    # GT says no mainland region is a SAR — both arrive at the same conclusion.
    ("HR_091", 0.0, 0.0, 0, "Response correctly identifies only HK and Macau as SARs"),
    # HR_092: Congress of Troppau. GT = Concert of Europe. Response = Congress
    # of Vienna. Wrong — Congress of Vienna (1814-15) was a single event, not
    # the series. The Congress System / Concert of Europe is the correct series.
    ("HR_092", 2.0, 2.0, 2, "Congress of Vienna was a single event (1814-15), not the series of meetings that included Troppau"),
    # HR_093: Liability release clause. GT = 'limitation of liability' or
    # 'exculpatory clause'. Response = 'release or waiver'. Release/waiver
    # typically applies post-claim; exculpatory clause is prospective.
    ("HR_093", 1.5, 2.0, 2, "Release/waiver is adjacent but technically distinct from exculpatory/limitation of liability clause"),
    # HR_094: Bitcoin = Proof of Work. Correct.
    ("HR_094", 0.0, 0.0, 0, ""),
    # HR_095: ECMAScript sort. Response says 'DefaultCompareSort' (fabricated)
    # then partially self-corrects to SortCompare. Mixed — contains fabrication
    # but acknowledges the correct answer.
    ("HR_095", 1.5, 2.0, 2, "Fabricates 'DefaultCompareSort' but partially self-corrects to SortCompare"),
    # HR_096: Learned treatise FRE 803(18). Correct.
    ("HR_096", 0.0, 0.0, 0, ""),
    # HR_097: CheckMate 227 HR. GT=0.79, response=0.78. CI lower bound 0.67
    # vs 0.66. Very minor numerical discrepancy.
    ("HR_097", 0.5, 0.5, 0, "HR 0.78 vs GT 0.79 and slight CI difference — trivial numerical discrepancy"),
    # HR_098: GNI significance. Correct.
    ("HR_098", 0.0, 0.0, 0, ""),
    # HR_099: LLVM register allocator. GT = greedy / PBQP. Response = graph
    # coloring. Graph coloring is a general technique but not what LLVM uses.
    ("HR_099", 2.0, 2.0, 2, "LLVM uses a greedy/PBQP allocator, not graph coloring — wrong technique"),
    # HR_100: Shanghai = direct-controlled municipality. Correct.
    ("HR_100", 0.0, 0.0, 0, ""),
]


def main():
    # Read existing CSV
    form_path = PILOT_DIR / "human_rating_form.csv"
    rows = []
    with form_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Build rating lookup
    rating_lookup = {r[0]: r for r in RATINGS}

    # Apply ratings
    scored = 0
    for row in rows:
        rid = row["rating_id"]
        if rid in rating_lookup:
            _, s11, s7, s5, notes = rating_lookup[rid]
            row["score_11point"] = str(s11)
            row["score_7point"] = str(s7)
            row["score_5level"] = str(s5)
            row["notes"] = notes
            scored += 1

    # Write completed ratings
    output_path = PILOT_DIR / "human_ratings_claude_rater.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scored {scored}/100 items")
    print(f"Saved to {output_path}")

    # Also produce JSONL version for analysis
    jsonl_path = PILOT_DIR / "human_ratings_claude_rater.jsonl"
    records = []
    for rid, s11, s7, s5, notes in RATINGS:
        records.append({
            "rating_id": rid,
            "score_11point": s11,
            "score_7point": s7,
            "score_5level": s5,
            "notes": notes,
        })
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    # Print summary statistics
    scores_11 = [r[1] for r in RATINGS]
    from collections import Counter
    dist = Counter(scores_11)
    print(f"\n11-point score distribution:")
    for score in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        count = dist.get(score, 0)
        bar = "#" * count
        print(f"  {score:.1f}: {bar} ({count})")

    perfect = sum(1 for s in scores_11 if s == 0.0)
    errors = sum(1 for s in scores_11 if s > 0.0)
    print(f"\nPerfect: {perfect}, Errors: {errors}")
    if errors > 0:
        avg_err = sum(s for s in scores_11 if s > 0.0) / errors
        print(f"Average error severity (non-zero only): {avg_err:.2f}")


if __name__ == "__main__":
    main()
