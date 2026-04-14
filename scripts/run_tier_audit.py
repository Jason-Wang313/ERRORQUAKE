"""Audit T1/T5 tier calibration across final query outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

DOMAINS = ["BIO", "LAW", "HIST", "GEO", "SCI", "TECH", "FIN", "CULT"]

PERSON_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
ALL_CAPS_RE = re.compile(r"\b[A-Z]{3,}(?:-[A-Z0-9]+)*\b")
DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b")
VERSION_RE = re.compile(
    r"\b(?:v(?:ersion)?\s*\d+(?:\.\d+)+|version\s+\d+(?:\.\d+)+|edition\s+\d+|release\s+\d+)\b",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(
    r"""
    (?:
        \b\d+(?:\.\d+)?\s*(?:%|percent|mV|mm|cm|nm|um|μm|kg|g|mg|mcg|ug|mL|L|Hz|kHz|MHz|GHz|GB|TB|MB|ms|s|hr|hours|days|weeks|months|years|mmHg)\b
        |
        \b(?:hazard|odds|risk|fold|rate)\s+ratio\b
        |
        \b\d+(?:\.\d+)?\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
NUMERIC_WORD_RE = re.compile(
    r"""
    (?:
        \b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|
        fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|
        sixty|seventy|eighty|ninety|hundred)\b
        |
        \bhow\ many\b
        |
        \bexact\ number\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
STUDY_RE = re.compile(
    r"""
    (?:
        \b(?:trial|study|paper|cohort|registry|guideline|protocol|consensus|meta-analysis|phase\s+[ivx]+)\b
        |
        \b[A-Z][A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
RATIONALE_PRECISION_RE = re.compile(
    r"""
    (?:
        precise\ recall
        |
        exact\ (?:value|figure|measurement|data|parameter|date|version)
        |
        specific\ data\ point
        |
        specific\ numerical\ (?:detail|fact|result|value)
        |
        numerical\ (?:result|value|measurement)
        |
        detailed\ fact
        |
        precise\ knowledge
        |
        exact\ number
        |
        precise\ understanding
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
RATIONALE_BASIC_RE = re.compile(
    r"\b(?:basic|elementary|introductory|widely known|common knowledge|suitable for T1|widely available)\b",
    re.IGNORECASE,
)
RATIONALE_SPECIALIZED_RE = re.compile(
    r"""
    (?:
        precise
        |
        technical
        |
        nuanced
        |
        detailed
        |
        clinical
        |
        genetic
        |
        doctrine
        |
        mechanism
        |
        innovation
        |
        domain-specific
        |
        specialist
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
TECHNICAL_KEYWORD_RE = re.compile(
    r"""
    \b(?:
        glucuronidation|cerebrospinal|lumbar\ puncture|electrophysiolog\w*|pharmacokinetic\w*|parasympathetic|
        promissory|restitution|estoppel|establishment\ clause|jurisdiction|indemnity|usufruct|
        select\ for\ update|dimensionality|principal\ component\ analysis|kernel|transformer|checkpointing|
        muddling|puddling|scholasticism|archipelago|hemispheric|stoichiometr\w*|spectroscop\w*|genotype|phenotype|
        amendment|enzyme|pathway|protocol|procedure|statement|trial|study|doctrine|tort|wrought\ iron
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A lightweight allowlist of household-name figures. Everyone else is treated as
# potentially specialist when they appear in T1 questions.
COMMON_FIGURES = {
    "Albert Einstein",
    "Alexander the Great",
    "Abraham Lincoln",
    "Adolf Hitler",
    "Aristotle",
    "Barack Obama",
    "Benjamin Franklin",
    "Buddha",
    "Charles Darwin",
    "Christopher Columbus",
    "Cleopatra",
    "Confucius",
    "Elizabeth I",
    "Galileo Galilei",
    "George Washington",
    "Henry VIII",
    "Isaac Newton",
    "Julius Caesar",
    "Karl Marx",
    "Leonardo da Vinci",
    "Louis XIV",
    "Mahatma Gandhi",
    "Martin Luther",
    "Martin Luther King",
    "Napoleon Bonaparte",
    "Nelson Mandela",
    "Plato",
    "Queen Victoria",
    "Ronald Reagan",
    "Thomas Aquinas",
    "Voltaire",
    "William Shakespeare",
    "Winston Churchill",
}

NON_PERSON_PHRASE_TOKENS = {
    "Act",
    "Acts",
    "Arabic",
    "Bank",
    "Battle",
    "Basis",
    "Clause",
    "Code",
    "Company",
    "Court",
    "Civil",
    "Constitution",
    "Convention",
    "Doctrine",
    "Empire",
    "Evidence",
    "Federal",
    "Front",
    "Kingdom",
    "Law",
    "Laws",
    "Modern",
    "Procedure",
    "Protection",
    "Railway",
    "Republic",
    "Revolution",
    "Rule",
    "Rules",
    "Scale",
    "Standard",
    "States",
    "Theory",
    "United",
    "War",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _join_sources(record: dict[str, Any]) -> str:
    return " | ".join(str(item) for item in record.get("sources", []))


def _t5_signals(record: dict[str, Any]) -> dict[str, bool]:
    question = str(record.get("question", ""))
    ground_truth = str(record.get("ground_truth", ""))
    rationale = str(record.get("difficulty_rationale", ""))
    sources = _join_sources(record)
    full_text = " ".join([question, ground_truth, sources])

    return {
        "named_study_trial_or_paper": bool(STUDY_RE.search(full_text)),
        "specific_numerical_result": bool(
            NUMERIC_RE.search(" ".join([question, ground_truth]))
            or NUMERIC_WORD_RE.search(" ".join([question, ground_truth]))
        ),
        "specific_date_version_or_edition": bool(
            DATE_RE.search(full_text) or VERSION_RE.search(full_text)
        ),
        "rationale_mentions_precise_recall": bool(RATIONALE_PRECISION_RE.search(rationale)),
    }


def _extract_person_names(question: str) -> list[str]:
    names = [match.group(1) for match in PERSON_NAME_RE.finditer(question)]
    filtered: list[str] = []
    for name in names:
        if name in COMMON_FIGURES:
            continue
        if name.lower().startswith(("the ", "what ", "which ", "in ")):
            continue
        tokens = name.split()
        if any(token in NON_PERSON_PHRASE_TOKENS for token in tokens):
            continue
        filtered.append(name)
    return filtered


def _t1_signals(record: dict[str, Any]) -> dict[str, Any]:
    question = str(record.get("question", ""))
    ground_truth = str(record.get("ground_truth", ""))
    rationale = str(record.get("difficulty_rationale", ""))
    names = _extract_person_names(question)
    long_tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z-]{9,}", " ".join([question, ground_truth]))
        if len(token) >= 14
    ]
    all_caps = [token for token in ALL_CAPS_RE.findall(question) if token not in {"T1", "T5"}]
    specialist_hits = {
        "named_person_outside_household_figures": bool(names),
        "named_process_or_technical_term": bool(TECHNICAL_KEYWORD_RE.search(question + " " + ground_truth)),
        "very_long_specialized_token": bool(long_tokens),
        "specialized_acronym": bool(all_caps),
    }
    specialist_score = sum(bool(value) for value in specialist_hits.values())
    rationale_basic = bool(RATIONALE_BASIC_RE.search(rationale))
    rationale_specialized = bool(RATIONALE_SPECIALIZED_RE.search(rationale))
    contradiction = (
        rationale_basic
        and rationale_specialized
        and (
            specialist_hits["named_person_outside_household_figures"]
            or specialist_hits["named_process_or_technical_term"]
            or specialist_hits["specialized_acronym"]
        )
    )
    likely_specialist = specialist_score >= 2 or (
        specialist_hits["named_person_outside_household_figures"]
        and specialist_hits["named_process_or_technical_term"]
    )
    return {
        "specialist_signals": specialist_hits,
        "specialist_score": specialist_score,
        "named_people": names,
        "long_tokens": long_tokens[:5],
        "contradictory_rationale": contradiction,
        "likely_specialist": likely_specialist,
    }


def _make_example(record: dict[str, Any], reasons: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "ground_truth": record.get("ground_truth"),
        "difficulty_rationale": record.get("difficulty_rationale"),
        "reasons": reasons,
        "evidence": evidence,
    }


def collect_flagged_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    t1_records = [record for record in records if int(record.get("tier", 0)) == 1]
    t5_records = [record for record in records if int(record.get("tier", 0)) == 5]

    t5_flagged: list[dict[str, Any]] = []
    for record in t5_records:
        signals = _t5_signals(record)
        if not any(signals.values()):
            t5_flagged.append(
                _make_example(
                    record,
                    reasons=["none_of_the_required_T5_signals_present"],
                    evidence=signals,
                )
            )

    t1_flagged: list[dict[str, Any]] = []
    for record in t1_records:
        t1 = _t1_signals(record)
        reasons: list[str] = []
        if t1["likely_specialist"]:
            reasons.append("requires_specialist_knowledge")
        if t1["contradictory_rationale"]:
            reasons.append("difficulty_rationale_contradicts_specialization")
        if reasons:
            evidence = {
                "specialist_signals": t1["specialist_signals"],
                "specialist_score": t1["specialist_score"],
                "named_people": t1["named_people"],
                "long_tokens": t1["long_tokens"],
                "contradictory_rationale": t1["contradictory_rationale"],
            }
            t1_flagged.append(_make_example(record, reasons=reasons, evidence=evidence))

    return {"t1": t1_flagged, "t5": t5_flagged}


def _domain_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    t1_records = [record for record in records if int(record.get("tier", 0)) == 1]
    t5_records = [record for record in records if int(record.get("tier", 0)) == 5]
    flagged = collect_flagged_records(records)
    t1_flagged = flagged["t1"]
    t5_flagged = flagged["t5"]

    def pct(flagged: list[dict[str, Any]], total: list[dict[str, Any]]) -> float:
        return round((len(flagged) / len(total) * 100.0), 2) if total else 0.0

    return {
        "t5": {
            "total": len(t5_records),
            "likely_miscalibrated": len(t5_flagged),
            "percentage": pct(t5_flagged, t5_records),
            "examples": t5_flagged[:3],
        },
        "t1": {
            "total": len(t1_records),
            "likely_miscalibrated": len(t1_flagged),
            "percentage": pct(t1_flagged, t1_records),
            "examples": t1_flagged[:3],
        },
    }


def _overall_verdict(report_by_domain: dict[str, Any]) -> dict[str, Any]:
    t5_rates = [domain_report["t5"]["percentage"] for domain_report in report_by_domain.values()]
    t1_rates = [domain_report["t1"]["percentage"] for domain_report in report_by_domain.values()]
    high_t5_domains = [
        domain
        for domain, domain_report in report_by_domain.items()
        if domain_report["t5"]["percentage"] >= 50.0
    ]
    high_t1_domains = [
        domain
        for domain, domain_report in report_by_domain.items()
        if domain_report["t1"]["percentage"] >= 50.0
    ]

    avg_t5 = round(mean(t5_rates), 2) if t5_rates else 0.0
    avg_t1 = round(mean(t1_rates), 2) if t1_rates else 0.0
    if len(high_t5_domains) >= 5 or (avg_t5 >= 50.0 and len(high_t5_domains) >= 4):
        verdict = "model_wide_problem"
        explanation = "T5 miscalibration appears broad across domains rather than isolated."
    elif len(high_t1_domains) >= 5 or (avg_t1 >= 50.0 and len(high_t1_domains) >= 4):
        verdict = "model_wide_problem"
        explanation = "T1 miscalibration appears broad across domains rather than isolated."
    else:
        verdict = "domain_specific_problem"
        explanation = "Miscalibration clusters in particular domains rather than hitting most domains uniformly."

    return {
        "verdict": verdict,
        "explanation": explanation,
        "average_t5_miscalibration_pct": avg_t5,
        "average_t1_miscalibration_pct": avg_t1,
        "domains_with_t5_pct_at_least_50": high_t5_domains,
        "domains_with_t1_pct_at_least_50": high_t1_domains,
    }


def _parse_cells(value: str | None) -> set[tuple[str, int]] | None:
    if not value:
        return None
    cells: set[tuple[str, int]] = set()
    for chunk in value.split(","):
        item = chunk.strip().upper()
        if not item:
            continue
        domain, tier = item.split("_T", 1)
        cells.add((domain, int(tier)))
    return cells


def build_report(
    *,
    output_dir: Path,
    selected_cells: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    report_by_domain: dict[str, Any] = {}

    for domain in DOMAINS:
        path = output_dir / f"{domain.lower()}.jsonl"
        records = _read_jsonl(path)
        if selected_cells is not None:
            records = [
                record
                for record in records
                if (str(record.get("domain", "")).upper(), int(record.get("tier", 0)))
                in selected_cells
            ]
        if not records:
            continue
        report_by_domain[domain] = _domain_report(records)

    return {
        "methodology": {
            "t5_rule": "Flagged when none of the required T5 signals are present.",
            "t1_rule": "Flagged when specialist-knowledge heuristics fire or the rationale contradicts itself.",
            "signals_checked": {
                "t5": [
                    "named study/trial/paper",
                    "specific numerical result",
                    "specific date/version/edition",
                    "rationale mentions precise recall of data points",
                ],
                "t1": [
                    "specialist named person outside household-figure allowlist",
                    "named process or technical mechanism",
                    "long specialized token or acronym",
                    "rationale basic/specialized contradiction",
                ],
            },
        },
        "per_domain": report_by_domain,
        "overall": _overall_verdict(report_by_domain),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit T1/T5 tier calibration across final query outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "queries",
        help="Directory containing final domain JSONL files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to save the JSON report. Defaults to <output-dir>/tier_audit_report.json.",
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="Optional comma-separated cell filter like BIO_T1,LAW_T1,TECH_T5.",
    )
    args = parser.parse_args()

    report_path = args.output_path or (args.output_dir / "tier_audit_report.json")
    report = build_report(output_dir=args.output_dir, selected_cells=_parse_cells(args.cells))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
