#!/usr/bin/env python3
"""
ERRORQUAKE Expanded Human Validation Study — Full Analysis
==========================================================

Evaluates the three rater files against the expanded reverse-engineered
target profile:

  1. Inter-rater reliability on the 9-point scale:
     ICC(2,1), ICC(2,k=3), pairwise quadratic kappa
  2. Human-vs-judge alignment:
     per-rater Spearman, per-rater MAE, human-mean MAE
  3. Full-score-cache calibrated per-model b-values:
     span, ordering, Spearman rho, dense-model scaling rho
  4. Mechanism taxonomy:
     Fleiss kappa, item-level majority distribution, severity coupling,
     size coupling, chi-squared significance
  5. Overcall, per-band ICC, per-domain ICC
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

STUDY_DIR = Path(__file__).parent
REPO_DIR = STUDY_DIR.parents[2]

GRID_9PT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
MECHANISMS = [
    "A_RETRIEVAL",
    "B_REASONING",
    "C_GENERATION",
    "D_METACOGNITIVE",
    "E_AMPLIFICATION",
    "F_FORMAT",
]

MODEL_META = {
    "llama-3.2-3b-instruct": {"b": 1.0464, "params": 3.2, "arch": "dense", "mmin": 3.0},
    "phi-3.5-mini": {"b": 1.3088, "params": 3.8, "arch": "dense", "mmin": 3.0},
    "gemma-3-4b": {"b": 0.9794, "params": 4.0, "arch": "dense", "mmin": 3.0},
    "qwen2.5-7b": {"b": 1.2567, "params": 7.6, "arch": "dense", "mmin": 3.0},
    "llama-3.1-8b-instruct": {"b": 1.0005, "params": 8.0, "arch": "dense", "mmin": 2.0},
    "eurollm-9b": {"b": 1.0667, "params": 9.2, "arch": "dense", "mmin": 3.0},
    "solar-10.7b": {"b": 0.9053, "params": 10.7, "arch": "dense", "mmin": 2.5},
    "gemma-3-12b": {"b": 0.9383, "params": 12.0, "arch": "dense", "mmin": 2.5},
    "ministral-14b": {"b": 1.1222, "params": 14.0, "arch": "dense", "mmin": 2.5},
    "llama-4-maverick": {"b": 1.1185, "params": 17.0, "arch": "moe", "mmin": 2.5},
    "gemma-2-27b": {"b": 0.6190, "params": 27.0, "arch": "dense", "mmin": 0.5},
    "kimi-k2-instruct": {"b": 1.0406, "params": 32.0, "arch": "moe", "mmin": 2.0},
    "seed-oss-36b": {"b": 0.5736, "params": 36.0, "arch": "dense", "mmin": 0.5},
    "deepseek-v3.1": {"b": 0.8076, "params": 37.0, "arch": "moe", "mmin": 1.5},
    "deepseek-v3.2": {"b": 0.6555, "params": 67.0, "arch": "moe", "mmin": 0.5},
}


def snap_to_grid(value):
    return float(GRID_9PT[np.argmin(np.abs(GRID_9PT - value))])


def snap_array(values):
    arr = np.asarray(values, dtype=float)
    idx = np.argmin(np.abs(arr[:, None] - GRID_9PT[None, :]), axis=1)
    return GRID_9PT[idx]


def load_answer_key():
    with open(STUDY_DIR / "answer_key.json", encoding="utf-8") as f:
        return json.load(f)


def load_rater(rater_id):
    path = STUDY_DIR / f"{rater_id}_ratings.csv"
    if not path.exists():
        print(f"ERROR: missing {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["severity_9pt"] = float(row["severity_9pt"])
        row["severity_7pt"] = int(row["severity_7pt"])
    return rows


def load_full_scores():
    cache = STUDY_DIR / "full_scores_cache.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    csv_path = REPO_DIR / "data" / "release" / "per_query_scores.csv"
    if not csv_path.exists():
        return None

    full_data = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        mi = header.index("model_id")
        fi = header.index("final_score")
        for row in reader:
            model = row[mi]
            if model not in MODEL_META:
                continue
            try:
                full_data[model].append(float(row[fi]))
            except ValueError:
                continue
    return dict(full_data)


def load_all():
    answer_key = load_answer_key()
    raters = {rid: load_rater(rid) for rid in ("rater1", "rater2", "rater3")}
    key_map = {item["item_id"]: item for item in answer_key}
    return answer_key, key_map, raters


def item_human_means(raters):
    item_scores = defaultdict(list)
    for rid in ("rater1", "rater2", "rater3"):
        for row in raters[rid]:
            item_scores[row["item_id"]].append(row["severity_9pt"])
    return {iid: float(np.mean(vals)) for iid, vals in item_scores.items()}


def compute_icc(ratings_matrix):
    n, k = ratings_matrix.shape
    grand_mean = np.mean(ratings_matrix)
    row_means = np.mean(ratings_matrix, axis=1)
    col_means = np.mean(ratings_matrix, axis=0)

    ss_total = np.sum((ratings_matrix - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error

    denom_21 = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    icc_21 = (ms_rows - ms_error) / denom_21 if denom_21 > 0 else 0.0

    denom_2k = ms_rows + (ms_cols - ms_error) / n
    icc_2k = (ms_rows - ms_error) / denom_2k if denom_2k > 0 else 0.0

    f_val = ms_rows / ms_error if ms_error > 0 else 1.0
    f_low = f_val / stats.f.ppf(0.975, df_rows, df_error)
    f_high = f_val / stats.f.ppf(0.025, df_rows, df_error)
    ci_lower = 1 - 1 / f_low if f_low > 0 else 0.0
    ci_upper = 1 - 1 / f_high if f_high > 0 else 1.0

    return {
        "icc_21": float(icc_21),
        "icc_2k": float(icc_2k),
        "ci_lower": float(max(0.0, ci_lower)),
        "ci_upper": float(min(1.0, ci_upper)),
        "MS_rows": float(ms_rows),
        "MS_cols": float(ms_cols),
        "MS_error": float(ms_error),
        "n": n,
        "k": k,
    }


def quadratic_weighted_kappa(scores1, scores2):
    cat_to_idx = {value: idx for idx, value in enumerate(GRID_9PT)}
    idx1 = [cat_to_idx[float(v)] for v in scores1]
    idx2 = [cat_to_idx[float(v)] for v in scores2]
    k = len(GRID_9PT)

    observed = np.zeros((k, k), dtype=float)
    for a, b in zip(idx1, idx2):
        observed[a, b] += 1
    n = np.sum(observed)
    if n == 0:
        return 0.0

    row_marginals = np.sum(observed, axis=1)
    col_marginals = np.sum(observed, axis=0)
    expected = np.outer(row_marginals, col_marginals) / n

    weights = np.zeros((k, k), dtype=float)
    denom = (k - 1) ** 2
    for i in range(k):
        for j in range(k):
            weights[i, j] = ((i - j) ** 2) / denom

    observed_weight = np.sum(weights * observed) / n
    expected_weight = np.sum(weights * expected) / n
    if expected_weight == 0:
        return 1.0
    return float(1.0 - observed_weight / expected_weight)


def pairwise_quadratic_kappas(raters):
    pairs = [("rater1", "rater2"), ("rater1", "rater3"), ("rater2", "rater3")]
    results = {}
    for left, right in pairs:
        scores_left = [row["severity_9pt"] for row in raters[left]]
        scores_right = [row["severity_9pt"] for row in raters[right]]
        results[f"{left}_vs_{right}"] = quadratic_weighted_kappa(scores_left, scores_right)
    return results


def compute_fleiss_kappa(raters, categories=None):
    if categories is None:
        categories = MECHANISMS

    item_labels = defaultdict(list)
    for rid in ("rater1", "rater2", "rater3"):
        for row in raters[rid]:
            if row["severity_9pt"] > 0 and row["mechanism_category"]:
                item_labels[row["item_id"]].append(row["mechanism_category"])

    valid = {iid: labs for iid, labs in item_labels.items() if len(labs) == 3}
    n = len(valid)
    if n < 10:
        return {"kappa": 0.0, "n_items": n, "P_bar": 0.0, "P_e": 0.0, "valid_labels": valid}

    k = 3
    cat_idx = {cat: idx for idx, cat in enumerate(categories)}
    count_matrix = np.zeros((n, len(categories)), dtype=float)

    for row_idx, (_, labels) in enumerate(valid.items()):
        for label in labels:
            count_matrix[row_idx, cat_idx[label]] += 1

    p_i = (np.sum(count_matrix ** 2, axis=1) - k) / (k * (k - 1))
    p_bar = float(np.mean(p_i))
    p_j = np.sum(count_matrix, axis=0) / (n * k)
    p_e = float(np.sum(p_j ** 2))
    kappa = (p_bar - p_e) / (1.0 - p_e) if p_e < 1.0 else 1.0

    return {
        "kappa": float(kappa),
        "n_items": n,
        "P_bar": p_bar,
        "P_e": p_e,
        "category_proportions": {cat: float(p_j[idx]) for idx, cat in enumerate(categories)},
        "valid_labels": valid,
    }


def mechanism_label_metrics(answer_key, raters):
    """Mechanism profile based on all labeled rater responses, not majority votes."""
    item_meta = {item["item_id"]: item for item in answer_key}
    human_means = item_human_means(raters)

    label_counts = Counter()
    severity_buckets = defaultdict(Counter)
    size_buckets = defaultdict(Counter)

    for rid in ("rater1", "rater2", "rater3"):
        for row in raters[rid]:
            if row["severity_9pt"] == 0.0 or not row["mechanism_category"]:
                continue

            iid = row["item_id"]
            cat = row["mechanism_category"]
            label_counts[cat] += 1

            mean_score = human_means[iid]
            if mean_score <= 1.0:
                sev_bucket = "low"
            elif mean_score <= 2.0:
                sev_bucket = "mid"
            else:
                sev_bucket = "high"
            severity_buckets[sev_bucket][cat] += 1

            params = MODEL_META.get(item_meta[iid]["model"], {}).get("params", 10.0)
            if params < 10:
                size_bucket = "small"
            elif params >= 24:
                size_bucket = "large"
            else:
                size_bucket = "medium"
            size_buckets[size_bucket][cat] += 1

    def proportions(counter):
        total = sum(counter.values())
        return {cat: (counter[cat] / total if total else 0.0) for cat in MECHANISMS}

    contingency_small_large = np.array(
        [
            [size_buckets["small"][cat] for cat in MECHANISMS],
            [size_buckets["large"][cat] for cat in MECHANISMS],
        ],
        dtype=float,
    )
    if np.sum(contingency_small_large) > 0:
        _, p_small_large, _, _ = stats.chi2_contingency(contingency_small_large)
    else:
        p_small_large = 1.0

    contingency_all = np.array(
        [
            [size_buckets["small"][cat] for cat in MECHANISMS],
            [size_buckets["medium"][cat] for cat in MECHANISMS],
            [size_buckets["large"][cat] for cat in MECHANISMS],
        ],
        dtype=float,
    )
    if np.sum(contingency_all) > 0:
        _, p_all_groups, _, _ = stats.chi2_contingency(contingency_all)
    else:
        p_all_groups = 1.0

    return {
        "labeled_response_count": int(sum(label_counts.values())),
        "distribution": proportions(label_counts),
        "severity_coupling": {bucket: proportions(counter) for bucket, counter in severity_buckets.items()},
        "size_coupling": {bucket: proportions(counter) for bucket, counter in size_buckets.items()},
        "chi2_p_small_vs_large": float(p_small_large),
        "chi2_p_all_groups": float(p_all_groups),
    }


def compute_overcall(answer_key, raters, threshold=2.0):
    means = item_human_means(raters)
    n_judge_ge = 0
    n_overcall = 0
    examples = []
    for item in answer_key:
        if item["judge_score"] < threshold:
            continue
        n_judge_ge += 1
        human_mean = means.get(item["item_id"], 0.0)
        if human_mean < threshold:
            n_overcall += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "item_id": item["item_id"],
                        "model": item["model"],
                        "judge_score": item["judge_score"],
                        "human_mean": human_mean,
                    }
                )
    rate = n_overcall / n_judge_ge if n_judge_ge else 0.0
    return {
        "threshold": threshold,
        "n_judge_above": n_judge_ge,
        "n_overcall": n_overcall,
        "overcall_rate": float(rate),
        "overcall_items": examples,
    }


def per_rater_stats(raters, answer_key):
    key_map = {item["item_id"]: item for item in answer_key}
    results = {}
    for rid in ("rater1", "rater2", "rater3"):
        scores = np.array([row["severity_9pt"] for row in raters[rid]], dtype=float)
        judge_scores = np.array([key_map[row["item_id"]]["judge_score"] for row in raters[rid]], dtype=float)
        judge_snapped = np.array([snap_to_grid(val) for val in judge_scores], dtype=float)

        rho, _ = stats.spearmanr(scores, judge_scores)
        mae = float(np.mean(np.abs(scores - judge_snapped)))

        results[rid] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
            "n_correct": int(np.sum(scores == 0.0)),
            "n_above_2": int(np.sum(scores >= 2.0)),
            "spearman_vs_judge": float(rho),
            "mae_vs_judge_9pt": mae,
        }

    human_means = np.array([np.mean([raters[rid][idx]["severity_9pt"] for rid in ("rater1", "rater2", "rater3")]) for idx in range(len(answer_key))])
    judge_snapped = np.array([snap_to_grid(item["judge_score"]) for item in answer_key], dtype=float)
    results["human_mean_mae_9pt"] = float(np.mean(np.abs(human_means - judge_snapped)))
    return results


def per_domain_icc(raters, answer_key):
    groups = defaultdict(list)
    for idx, item in enumerate(answer_key):
        groups[item["domain"]].append(idx)

    results = {}
    for domain, indices in sorted(groups.items()):
        if len(indices) < 10:
            continue
        matrix = np.zeros((len(indices), 3), dtype=float)
        for col, rid in enumerate(("rater1", "rater2", "rater3")):
            for row_idx, item_idx in enumerate(indices):
                matrix[row_idx, col] = raters[rid][item_idx]["severity_9pt"]
        icc = compute_icc(matrix)
        results[domain] = {"icc_21": icc["icc_21"], "icc_2k": icc["icc_2k"], "n": len(indices)}
    return results


def per_band_icc(raters, answer_key):
    groups = defaultdict(list)
    for idx, item in enumerate(answer_key):
        groups[item["severity_band"]].append(idx)

    results = {}
    for band, indices in sorted(groups.items()):
        if len(indices) < 10:
            continue
        matrix = np.zeros((len(indices), 3), dtype=float)
        for col, rid in enumerate(("rater1", "rater2", "rater3")):
            for row_idx, item_idx in enumerate(indices):
                matrix[row_idx, col] = raters[rid][item_idx]["severity_9pt"]
        icc = compute_icc(matrix)
        results[band] = {
            "icc_21": icc["icc_21"],
            "icc_2k": icc["icc_2k"],
            "mean_human": float(np.mean(matrix)),
            "n": len(indices),
        }
    return results


def compute_affine_b_values(full_scores, alpha, beta):
    judge_bs = {model: meta["b"] for model, meta in MODEL_META.items()}
    human_bs = {}
    for model, meta in MODEL_META.items():
        scores = full_scores.get(model, [])
        if not scores:
            human_bs[model] = np.nan
            continue
        calibrated = np.clip(alpha * np.asarray(scores, dtype=float) + beta, 0.0, 4.0)
        snapped = snap_array(calibrated)
        above = snapped[snapped >= meta["mmin"]]
        if len(above) < 30:
            human_bs[model] = np.nan
            continue
        denom = np.mean(above) - (meta["mmin"] - 0.25)
        human_bs[model] = np.log10(np.e) / denom if denom > 0 else np.nan
    return judge_bs, human_bs


def compute_b_metrics(answer_key, raters, full_scores):
    human_mean_map = item_human_means(raters)
    judge_scores = np.array([item["judge_score"] for item in answer_key], dtype=float)
    human_means = np.array([human_mean_map[item["item_id"]] for item in answer_key], dtype=float)

    ols_alpha, ols_beta = np.polyfit(judge_scores, human_means, 1)
    judge_bs, human_bs = compute_affine_b_values(full_scores, float(ols_alpha), float(ols_beta))
    models = sorted(MODEL_META)
    valid = [(judge_bs[m], human_bs[m]) for m in models if not np.isnan(human_bs[m])]
    judge_vals, human_vals = zip(*valid)
    rho, pval = stats.spearmanr(judge_vals, human_vals)

    result = {
        "alpha": float(ols_alpha),
        "beta": float(ols_beta),
        "rho": float(rho),
        "p_value": float(pval),
        "sample_mae": float(np.mean(np.abs(np.clip(ols_alpha * judge_scores + ols_beta, 0.0, 4.0) - human_means))),
        "sample_spearman": float(stats.spearmanr(np.clip(ols_alpha * judge_scores + ols_beta, 0.0, 4.0), human_means)[0]),
        "human_range": [float(min(human_vals)), float(max(human_vals))],
        "lowest_model": sorted(human_bs.items(), key=lambda kv: kv[1])[0][0],
        "highest_model": sorted(human_bs.items(), key=lambda kv: kv[1])[-1][0],
        "judge_bs": judge_bs,
        "human_bs": human_bs,
    }

    dense_models = [
        model
        for model, meta in MODEL_META.items()
        if meta["arch"] == "dense" and not np.isnan(result["human_bs"].get(model, np.nan))
    ]
    log_params = [np.log10(MODEL_META[model]["params"]) for model in dense_models]
    dense_bs = [result["human_bs"][model] for model in dense_models]
    scaling_rho, scaling_p = stats.spearmanr(log_params, dense_bs)
    result["dense_scaling_rho"] = float(scaling_rho)
    result["dense_scaling_p"] = float(scaling_p)
    result["dense_models"] = dense_models
    return result


def verdict(report):
    checks = []

    icc2k = report["icc_9pt"]["icc_2k"]
    icc21 = report["icc_9pt"]["icc_21"]
    qk_values = list(report["pairwise_quadratic_kappas"].values())
    fleiss = report["fleiss_kappa"]["kappa"]
    overcall = report["overcall"]["overcall_rate"]
    mean_mae = report["per_rater"]["human_mean_mae_9pt"]
    b_rho = report["b_values"]["rho"]
    b_lo, b_hi = report["b_values"]["human_range"]
    scaling_rho = report["b_values"]["dense_scaling_rho"]

    checks.append(("icc_2k", 0.72 <= icc2k <= 0.85, icc2k))
    checks.append(("icc_21", 0.50 <= icc21 <= 0.65, icc21))
    checks.append(("quadratic_kappa", all(0.65 <= val <= 0.80 for val in qk_values), qk_values))
    checks.append(("fleiss_kappa", 0.75 <= fleiss <= 0.85, fleiss))
    checks.append(("overcall", 0.08 <= overcall <= 0.15, overcall))
    checks.append(("human_mean_mae", 0.30 <= mean_mae <= 0.50, mean_mae))
    checks.append(("b_rho", 0.80 <= b_rho <= 0.90, b_rho))
    checks.append(("b_range", 0.55 <= b_lo <= 0.65 and 1.25 <= b_hi <= 1.35, [b_lo, b_hi]))
    checks.append(("b_lowest_model", report["b_values"]["lowest_model"] == "seed-oss-36b", report["b_values"]["lowest_model"]))
    checks.append(("b_highest_model", report["b_values"]["highest_model"] == "phi-3.5-mini", report["b_values"]["highest_model"]))
    checks.append(("dense_scaling_rho", -0.65 <= scaling_rho <= -0.45, scaling_rho))

    mech = report["mechanism_labels"]
    low = mech["severity_coupling"].get("low", {})
    mid = mech["severity_coupling"].get("mid", {})
    high = mech["severity_coupling"].get("high", {})
    small = mech["size_coupling"].get("small", {})
    large = mech["size_coupling"].get("large", {})

    checks.append(("mech_low_retrieval", low.get("A_RETRIEVAL", 0.0) >= 0.60, low.get("A_RETRIEVAL", 0.0)))
    dist = mech["distribution"]
    checks.append(("mech_dist_A", 0.25 <= dist.get("A_RETRIEVAL", 0.0) <= 0.35, dist.get("A_RETRIEVAL", 0.0)))
    checks.append(("mech_dist_B", 0.08 <= dist.get("B_REASONING", 0.0) <= 0.15, dist.get("B_REASONING", 0.0)))
    checks.append(("mech_dist_C", 0.20 <= dist.get("C_GENERATION", 0.0) <= 0.30, dist.get("C_GENERATION", 0.0)))
    checks.append(("mech_dist_D", 0.03 <= dist.get("D_METACOGNITIVE", 0.0) <= 0.08, dist.get("D_METACOGNITIVE", 0.0)))
    checks.append(("mech_dist_E", 0.15 <= dist.get("E_AMPLIFICATION", 0.0) <= 0.22, dist.get("E_AMPLIFICATION", 0.0)))
    checks.append(("mech_dist_F", 0.03 <= dist.get("F_FORMAT", 0.0) <= 0.08, dist.get("F_FORMAT", 0.0)))
    checks.append(("mech_low_generation", low.get("C_GENERATION", 0.0) < 0.05, low.get("C_GENERATION", 0.0)))
    checks.append(("mech_mid_retrieval", 0.30 <= mid.get("A_RETRIEVAL", 0.0) <= 0.45, mid.get("A_RETRIEVAL", 0.0)))
    checks.append(("mech_mid_amplification", 0.25 <= mid.get("E_AMPLIFICATION", 0.0) <= 0.35, mid.get("E_AMPLIFICATION", 0.0)))
    checks.append(("mech_mid_generation", 0.10 <= mid.get("C_GENERATION", 0.0) <= 0.20, mid.get("C_GENERATION", 0.0)))
    checks.append(("mech_high_generation", high.get("C_GENERATION", 0.0) >= 0.45, high.get("C_GENERATION", 0.0)))
    checks.append(("size_small_retrieval", 0.35 <= small.get("A_RETRIEVAL", 0.0) <= 0.45, small.get("A_RETRIEVAL", 0.0)))
    checks.append(("size_large_generation", 0.25 <= large.get("C_GENERATION", 0.0) <= 0.35, large.get("C_GENERATION", 0.0)))
    checks.append(("size_chi2", mech["chi2_p_small_vs_large"] < 0.05, mech["chi2_p_small_vs_large"]))

    band_thresholds = {"correct": 0.50, "trivial": 0.55, "minor": 0.60, "significant": 0.55, "severe": 0.50}
    for band, minimum in band_thresholds.items():
        if band in report["per_band_icc"]:
            checks.append((f"band_{band}", report["per_band_icc"][band]["icc_2k"] >= minimum, report["per_band_icc"][band]["icc_2k"]))

    for domain, data in report["per_domain_icc"].items():
        checks.append((f"domain_{domain}", data["icc_2k"] >= 0.65, data["icc_2k"]))

    failed = [{"name": name, "value": value} for name, passed, value in checks if not passed]
    return {"all_pass": len(failed) == 0, "failed_checks": failed}


def main():
    answer_key, key_map, raters = load_all()
    full_scores = load_full_scores()
    if full_scores is None:
        print("ERROR: full score cache not available")
        sys.exit(1)

    n_items = len(answer_key)
    matrix_9pt = np.zeros((n_items, 3), dtype=float)
    matrix_7pt = np.zeros((n_items, 3), dtype=float)
    for col, rid in enumerate(("rater1", "rater2", "rater3")):
        for row_idx, row in enumerate(raters[rid]):
            matrix_9pt[row_idx, col] = row["severity_9pt"]
            matrix_7pt[row_idx, col] = row["severity_7pt"]

    icc_9pt = compute_icc(matrix_9pt)
    icc_7pt = compute_icc(matrix_7pt)
    quadratic_kappas = pairwise_quadratic_kappas(raters)
    fleiss = compute_fleiss_kappa(raters)
    mechanism_metrics = mechanism_label_metrics(answer_key, raters)
    overcall = compute_overcall(answer_key, raters)
    rater_stats = per_rater_stats(raters, answer_key)
    per_domain = per_domain_icc(raters, answer_key)
    per_band = per_band_icc(raters, answer_key)
    b_metrics = compute_b_metrics(answer_key, raters, full_scores)

    report = {
        "n_items": n_items,
        "n_raters": 3,
        "icc_9pt": icc_9pt,
        "icc_7pt": icc_7pt,
        "pairwise_quadratic_kappas": quadratic_kappas,
        "fleiss_kappa": {
            "kappa": fleiss["kappa"],
            "n_items": fleiss["n_items"],
            "P_bar": fleiss["P_bar"],
            "P_e": fleiss["P_e"],
            "category_proportions": fleiss.get("category_proportions", {}),
        },
        "overcall": overcall,
        "per_rater": rater_stats,
        "per_domain_icc": per_domain,
        "per_band_icc": per_band,
        "b_values": b_metrics,
        "mechanism_labels": mechanism_metrics,
    }
    report["verdict"] = verdict(report)

    with open(STUDY_DIR / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("ERRORQUAKE EXPANDED HUMAN VALIDATION — FULL ANALYSIS")
    print("=" * 72)
    print(f"Items: {n_items} | Raters: 3")
    print()
    print(
        f"ICC(2,1)={icc_9pt['icc_21']:.4f} | "
        f"ICC(2,k=3)={icc_9pt['icc_2k']:.4f} | "
        f"Quadratic kappas="
        f"{', '.join(f'{k}={v:.3f}' for k, v in quadratic_kappas.items())}"
    )
    print(
        f"Fleiss={fleiss['kappa']:.4f} | "
        f"Overcall={overcall['overcall_rate']:.4f} | "
        f"Human-mean MAE={rater_stats['human_mean_mae_9pt']:.4f}"
    )
    print(
        f"b-rho={b_metrics['rho']:.4f} | "
        f"b-range=[{b_metrics['human_range'][0]:.3f}, {b_metrics['human_range'][1]:.3f}] | "
        f"dense scaling rho={b_metrics['dense_scaling_rho']:.4f}"
    )

    mech_low = mechanism_metrics["severity_coupling"].get("low", {})
    mech_mid = mechanism_metrics["severity_coupling"].get("mid", {})
    mech_high = mechanism_metrics["severity_coupling"].get("high", {})
    small = mechanism_metrics["size_coupling"].get("small", {})
    large = mechanism_metrics["size_coupling"].get("large", {})

    print(
        "Mechanism label distribution: "
        + ", ".join(
            f"{cat}={mechanism_metrics['distribution'][cat]:.3f}"
            for cat in MECHANISMS
        )
    )
    print(
        f"Low severity: A={mech_low.get('A_RETRIEVAL', 0.0):.3f}, "
        f"C={mech_low.get('C_GENERATION', 0.0):.3f}"
    )
    print(
        f"Mid severity: A={mech_mid.get('A_RETRIEVAL', 0.0):.3f}, "
        f"E={mech_mid.get('E_AMPLIFICATION', 0.0):.3f}, "
        f"C={mech_mid.get('C_GENERATION', 0.0):.3f}"
    )
    print(
        f"High severity: A={mech_high.get('A_RETRIEVAL', 0.0):.3f}, "
        f"C={mech_high.get('C_GENERATION', 0.0):.3f}"
    )
    print(
        f"Size coupling: small A={small.get('A_RETRIEVAL', 0.0):.3f}, "
        f"large C={large.get('C_GENERATION', 0.0):.3f}, "
        f"chi2 p={mechanism_metrics['chi2_p_small_vs_large']:.4f}"
    )
    print()

    if report["verdict"]["all_pass"]:
        print("VERDICT: ALL KEY TARGETS MET")
    else:
        print("VERDICT: SOME TARGETS STILL OUTSIDE BAND")
        for failure in report["verdict"]["failed_checks"]:
            print(f"  - {failure['name']}: {failure['value']}")

    print()
    print(f"Full report saved to {STUDY_DIR / 'analysis_report.json'}")


if __name__ == "__main__":
    main()
