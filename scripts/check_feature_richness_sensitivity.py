"""
check_feature_richness_sensitivity_replicated.py
===================================================

Repeats check_feature_richness_sensitivity.run_one_seed() across 5
seeds instead of trusting a single run. Reports mean/std per variant,
and the paired per-seed difference (rich minus baseline, same seed) --
paired because both variants are fit on identical data within a seed,
so the per-seed difference is a tighter signal than treating baseline
and rich as independent samples.

Run with:
    python scripts/check_feature_richness_sensitivity_replicated.py
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from check_feature_richness_sensitivity import run_one_seed, N_ITEMS, N_SESSIONS, ITEMS_PER_SESSION

N_REPLICATIONS = 5
BASE_SEED = 500  # kept separate from RANDOM_SEED=42 (the single-run script's seed)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                             "feature_richness_sensitivity_replicated.csv")
METRIC_KEYS = ["difficulty_pearson", "discrimination_pearson", "approx_item_grade_pearson"]


def run_all_replications() -> list:
    """Runs one seed per replication. Returns a list of
    (seed, baseline_result, rich_result) tuples."""
    all_runs = []

    for replication_index in range(N_REPLICATIONS):
        seed = BASE_SEED + replication_index * 100
        print(f"\n{'=' * 70}")
        print(f"REPLICATION {replication_index + 1}/{N_REPLICATIONS}  (seed={seed})")
        print(f"{'=' * 70}")

        baseline_result, rich_result = run_one_seed(
            seed, n_items=N_ITEMS, n_sessions=N_SESSIONS,
            items_per_session=ITEMS_PER_SESSION, verbose=False,
        )
        print(f"  baseline: difficulty r={baseline_result['difficulty_pearson']:.4f}  "
              f"discrimination r={baseline_result['discrimination_pearson']:.4f}  "
              f"item-grade r={baseline_result['approx_item_grade_pearson']:.4f}")
        print(f"  rich:     difficulty r={rich_result['difficulty_pearson']:.4f}  "
              f"discrimination r={rich_result['discrimination_pearson']:.4f}  "
              f"item-grade r={rich_result['approx_item_grade_pearson']:.4f}")

        all_runs.append((seed, baseline_result, rich_result))

    return all_runs


def summarize(all_runs: list) -> dict:
    """all_runs: list of (seed, baseline_result, rich_result). Returns
    mean/std per variant per metric, plus paired per-seed diffs
    (rich - baseline) with mean/std/min/max."""
    summary = {}

    # Mean/std per variant, per metric, across all replications.
    for variant_index, variant_name in enumerate(["baseline", "rich"]):
        for key in METRIC_KEYS:
            values = []
            for run in all_runs:
                result = run[1 + variant_index]  # index 1 = baseline_result, 2 = rich_result
                values.append(result[key])
            values = np.array(values)
            summary[f"{variant_name}_{key}_mean"] = values.mean()
            summary[f"{variant_name}_{key}_std"] = values.std()

    # Paired diff per seed (rich - baseline) -- see module docstring for why paired.
    for key in METRIC_KEYS:
        paired_diffs = []
        for run in all_runs:
            seed, baseline_result, rich_result = run
            paired_diffs.append(rich_result[key] - baseline_result[key])
        paired_diffs = np.array(paired_diffs)
        summary[f"paired_diff_{key}_mean"] = paired_diffs.mean()
        summary[f"paired_diff_{key}_std"] = paired_diffs.std()
        summary[f"paired_diff_{key}_min"] = paired_diffs.min()
        summary[f"paired_diff_{key}_max"] = paired_diffs.max()

    return summary


def print_summary_table(all_runs: list, summary: dict):
    print("\n" + "=" * 100)
    print(f"SUMMARY OVER {N_REPLICATIONS} REPLICATIONS "
          f"(n_items={N_ITEMS}, n_sessions={N_SESSIONS})")
    print("=" * 100)

    print(f"\n{'Metric':<24}{'Baseline (mean±std)':>26}{'Rich (mean±std)':>24}")
    for key in METRIC_KEYS:
        baseline_str = f"{summary[f'baseline_{key}_mean']:.4f} ± {summary[f'baseline_{key}_std']:.4f}"
        rich_str = f"{summary[f'rich_{key}_mean']:.4f} ± {summary[f'rich_{key}_std']:.4f}"
        print(f"{key:<24}{baseline_str:>26}{rich_str:>24}")

    print(f"\nPaired per-seed difference (rich - baseline):")
    print(f"{'Metric':<24}{'mean ± std':>20}{'min':>10}{'max':>10}")
    for key in METRIC_KEYS:
        mean_ = summary[f"paired_diff_{key}_mean"]
        std_ = summary[f"paired_diff_{key}_std"]
        min_ = summary[f"paired_diff_{key}_min"]
        max_ = summary[f"paired_diff_{key}_max"]
        print(f"{key:<24}{f'{mean_:+.4f} ± {std_:.4f}':>20}{min_:>+10.4f}{max_:>+10.4f}")

    seeds_used = []
    for run in all_runs:
        seeds_used.append(run[0])
    print(f"\nPer-replication seeds used: {seeds_used}")

    print("\nMin/max not crossing zero -> effect holds across every seed, trust it.")
    print("Min/max straddling zero -> the single-seed result was likely an overestimate.")


def save_summary_csv(all_runs: list, summary: dict):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    fieldnames = ["seed"]
    for key in METRIC_KEYS:
        fieldnames.append(f"baseline_{key}")
    for key in METRIC_KEYS:
        fieldnames.append(f"rich_{key}")
    for key in METRIC_KEYS:
        fieldnames.append(f"paired_diff_{key}")

    with open(RESULTS_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for seed, baseline_result, rich_result in all_runs:
            row = {"seed": seed}
            for key in METRIC_KEYS:
                row[f"baseline_{key}"] = baseline_result[key]
                row[f"rich_{key}"] = rich_result[key]
                row[f"paired_diff_{key}"] = rich_result[key] - baseline_result[key]
            writer.writerow(row)

    print(f"\nSaved per-replication results to: {RESULTS_PATH}")


def main():
    all_runs = run_all_replications()
    summary = summarize(all_runs)
    print_summary_table(all_runs, summary)
    save_summary_csv(all_runs, summary)


if __name__ == "__main__":
    main()