"""Rescore synthetic responses with DeepSeek-V3.2 as primary judge."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from errorquake.analyze import estimate_b_value, fit_all_distributions
from errorquake.synthetic import (
    DEFAULT_DISTRIBUTIONS,
    SYNTHETIC_SPECS,
    _resolve_distribution_scores,
    score_synthetic_responses,
    validate_pipeline_recovery,
)
from errorquake.utils import now_iso, read_jsonl, setup_logger

PRIMARY_MODEL = "deepseek-ai/deepseek-v3.2"
SECONDARY_MODEL = "qwen/qwen3-next-80b-a3b-instruct"
SCALE_POINTS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
MAP_7 = {1.5: 1.0, 2.5: 3.0}
MAP_5 = {
    0.0: 0.0,
    0.5: 0.0,
    1.0: 1.0,
    1.5: 1.0,
    2.0: 2.0,
    2.5: 2.0,
    3.0: 3.0,
    3.5: 3.0,
    4.0: 4.0,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("results/analysis/experiment_0_report_v3judge.json"),
    )
    parser.add_argument("--rpm", type=int, default=35)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--distributions",
        type=str,
        default=",".join(DEFAULT_DISTRIBUTIONS),
        help="Comma-separated subset of distributions.",
    )
    return parser.parse_args()


def _remap(values: list[float], mapping: dict[float, float]) -> np.ndarray:
    remapped: list[float] = []
    for value in values:
        rounded = round(float(value), 1)
        remapped.append(mapping.get(rounded, rounded))
    return np.array(remapped, dtype=float)


def _fit_payload(
    distribution: str,
    target_scores: np.ndarray,
    primary_scores: np.ndarray,
    final_scores: np.ndarray,
    *,
    mapping_name: str,
) -> dict[str, Any]:
    fits = fit_all_distributions(final_scores, model_name=f"{distribution}-{mapping_name}")
    b_estimate = estimate_b_value(final_scores, model_name=f"{distribution}-{mapping_name}")
    recovery = validate_pipeline_recovery(
        distribution,
        SYNTHETIC_SPECS[distribution]["params"],
        fits,
        final_scores=final_scores,
        target_scores=target_scores,
        b_estimate=b_estimate,
    )
    rho_final, _ = stats.spearmanr(target_scores, final_scores)
    rho_primary, _ = stats.spearmanr(target_scores, primary_scores)
    rho_final = 0.0 if np.isnan(rho_final) else float(rho_final)
    rho_primary = 0.0 if np.isnan(rho_primary) else float(rho_primary)
    return {
        "target_final_spearman_rho": rho_final,
        "target_primary_spearman_rho": rho_primary,
        "rho_passes_0_70": rho_final >= 0.70,
        "best_fit_family": fits[0].distribution,
        "bic_ranking": [
            {
                "distribution": fit.distribution,
                "bic": float(fit.bic),
                "aic": float(fit.aic),
                "chi2_pvalue": float(fit.chi2_pvalue),
                "parameters": {key: float(value) for key, value in fit.parameters.items()},
            }
            for fit in fits
        ],
        "recovered_b": float(b_estimate.b) if distribution == "power_law" else None,
        "recovery": recovery,
    }


def _column_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {f"{point:.1f}": 0 for point in SCALE_POINTS}
    for record in records:
        key = f"{float(record['primary_score']):.1f}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _confusion_matrix(target_scores: list[float], judged_scores: list[float]) -> list[list[int]]:
    matrix: list[list[int]] = []
    for target in SCALE_POINTS:
        row: list[int] = []
        for judged in SCALE_POINTS:
            row.append(
                sum(
                    int(round(float(t), 1) == target and round(float(j), 1) == judged)
                    for t, j in zip(target_scores, judged_scores)
                )
            )
        matrix.append(row)
    return matrix


async def main() -> None:
    args = _parse_args()
    synthetic_dir = args.synthetic_dir
    report_path = args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("errorquake.v3judge", synthetic_dir / "logs")
    selected = [item.strip() for item in args.distributions.split(",") if item.strip()]

    all_primary_records: list[dict[str, Any]] = []
    all_target_scores: list[float] = []
    all_primary_scores: list[float] = []
    all_final_scores: list[float] = []
    report: dict[str, Any] = {
        "experiment": "Experiment 0: Synthetic Pipeline Validation (DeepSeek-V3.2 primary judge)",
        "date": now_iso().split("T", 1)[0],
        "n_per_distribution": 500,
        "primary_judge": PRIMARY_MODEL,
        "secondary_judge": SECONDARY_MODEL,
        "scale": "11-point",
        "results": {},
    }

    for distribution in selected:
        responses_path = synthetic_dir / f"responses_{distribution}.jsonl"
        secondary_path = synthetic_dir / f"scores_secondary_{distribution}.jsonl"
        primary_path = synthetic_dir / f"scores_primary_v3judge_{distribution}.jsonl"
        resolved_path = synthetic_dir / f"resolved_scores_v3judge_{distribution}.jsonl"

        responses = read_jsonl(responses_path)
        secondary_scores = read_jsonl(secondary_path)
        if not responses:
            raise RuntimeError(f"Missing responses for {distribution}: {responses_path}")
        if len(secondary_scores) != len(responses):
            raise RuntimeError(
                f"Secondary scores incomplete for {distribution}: {len(secondary_scores)} / {len(responses)}"
            )

        logger.info("[%s] Starting V3.2 primary rescoring", distribution)
        primary_scores = await score_synthetic_responses(
            distribution=distribution,
            responses=responses,
            output_path=primary_path,
            judge_role="primary",
            judge_model=PRIMARY_MODEL,
            rpm=args.rpm,
            resume=args.resume,
            logger=logger,
        )
        if len(primary_scores) != len(responses):
            raise RuntimeError(
                f"Primary scores incomplete for {distribution}: {len(primary_scores)} / {len(responses)}"
            )

        resolved_records = _resolve_distribution_scores(
            distribution=distribution,
            responses=responses,
            primary_scores=primary_scores,
            secondary_scores=secondary_scores,
            output_path=resolved_path,
        )

        target_scores = np.array([float(record["target_score"]) for record in resolved_records], dtype=float)
        primary_values = np.array([float(record["primary_score"]) for record in resolved_records], dtype=float)
        final_scores = np.array([float(record["final_score"]) for record in resolved_records], dtype=float)

        payload_11 = _fit_payload(
            distribution,
            target_scores,
            primary_values,
            final_scores,
            mapping_name="11point-v3judge",
        )
        target_7 = _remap(target_scores.tolist(), MAP_7)
        primary_7 = _remap(primary_values.tolist(), MAP_7)
        final_7 = _remap(final_scores.tolist(), MAP_7)
        payload_7 = _fit_payload(
            distribution,
            target_7,
            primary_7,
            final_7,
            mapping_name="7point-v3judge",
        )
        target_5 = _remap(target_scores.tolist(), MAP_5)
        primary_5 = _remap(primary_values.tolist(), MAP_5)
        final_5 = _remap(final_scores.tolist(), MAP_5)
        payload_5 = _fit_payload(
            distribution,
            target_5,
            primary_5,
            final_5,
            mapping_name="5level-v3judge",
        )

        report["results"][distribution] = {
            "11_point": payload_11,
            "7_point_remap": payload_7,
            "5_level_remap": payload_5,
            "primary_column_totals": _column_totals(primary_scores),
        }

        all_primary_records.extend(primary_scores)
        all_target_scores.extend(target_scores.tolist())
        all_primary_scores.extend(primary_values.tolist())
        all_final_scores.extend(final_scores.tolist())

    report["primary_column_totals_overall"] = _column_totals(all_primary_records)
    report["overall_11_point_rho_passes_0_70"] = all(
        report["results"][distribution]["11_point"]["rho_passes_0_70"] for distribution in selected
    )
    report["overall_7_point_rho_passes_0_70"] = all(
        report["results"][distribution]["7_point_remap"]["rho_passes_0_70"] for distribution in selected
    )
    report["overall_5_level_rho_passes_0_70"] = all(
        report["results"][distribution]["5_level_remap"]["rho_passes_0_70"] for distribution in selected
    )
    report["judge_confusion_matrix"] = {
        "scale_points": SCALE_POINTS,
        "target_vs_primary": _confusion_matrix(all_target_scores, all_primary_scores),
        "target_vs_final": _confusion_matrix(all_target_scores, all_final_scores),
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
