"""
check_feature_richness_sensitivity_replicated.py
===================================================

The single-run feature-richness check (seed 42) showed a large effect:
richer feature representation moved difficulty recovery from -0.226 to
0.880 and the approximate item-grade Pearson from 0.019 to 0.807. That
is a big enough jump to be very unlikely to be pure noise, but it is
still one run, unreplicated -- every other finding in this project
(backend comparison, Cold/Jump-start convergence, the small- and
full-scale DET numbers) has been checked across multiple seeds before
being trusted, and this result should meet the same bar before it goes
into a report.

Runs check_feature_richness_sensitivity.run_one_seed() N_REPLICATIONS
times with different seeds and reports mean +/- std for both variants,
plus (more importantly) the PAIRED per-seed difference (rich minus
baseline) -- paired because baseline and rich are fit on the exact same
simulated items and responses within a given seed, so the difference
per seed is a tighter, more appropriate quantity than treating the two
variants as independent samples across seeds.

Run with:
    python scripts/check_feature_richness_sensitivity_replicated.py

Each replication takes about as long as the single-run script did
(no AutoML search, just RF+XGB+LGBM fits -- a few minutes), so this
should finish in well under an hour for N_REPLICATIONS=5.
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from check_feature_richness_sensitivity import run_one_seed, N_ITEMS, N_SESSIONS, ITEMS_PER_SESSION

N_REPLICATIONS = 5
BASE_SEED = 500  # distinct from check_feature_richness_sensitivity.RANDOM_SEED (42),
                  # spread by 100 per replication, matching the convention used
                  # throughout this project's other _replicated scripts

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                             "feature_richness_sensitivity_replicated.csv")
METRIC_KEYS = ["difficulty_pearson", "discrimination_pearson", "approx_item_grade_pearson"]


def run_all_replications() -> list:
    """Returns a list of (seed, baseline_result, rich_result) tuples,
    one per replication."""
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
    summary = {}
    for variant_index, variant_name in enumerate(["baseline", "rich"]):
        for key in METRIC_KEYS:
            values = np.array([run[1 + variant_index][key] for run in all_runs])
            summary[f"{variant_name}_{key}_mean"] = values.mean()
            summary[f"{variant_name}_{key}_std"] = values.std()

    # Paired per-seed differences (rich minus baseline), not independent-sample
    # differences -- see module docstring for why this is the right quantity.
    for key in METRIC_KEYS:
        paired_diffs = np.array([run[2][key] - run[1][key] for run in all_runs])
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

    print(f"\nPaired per-seed difference (rich - baseline), the number that actually answers")
    print(f"'does this reliably help, and by how much':")
    print(f"{'Metric':<24}{'mean ± std':>20}{'min':>10}{'max':>10}")
    for key in METRIC_KEYS:
        mean_ = summary[f"paired_diff_{key}_mean"]
        std_ = summary[f"paired_diff_{key}_std"]
        min_ = summary[f"paired_diff_{key}_min"]
        max_ = summary[f"paired_diff_{key}_max"]
        print(f"{key:<24}{f'{mean_:+.4f} ± {std_:.4f}':>20}{min_:>+10.4f}{max_:>+10.4f}")

    print(f"\nPer-replication seeds used: {[run[0] for run in all_runs]}")
    print("\nIf the paired difference's mean is large relative to its std (and min/max don't")
    print("cross zero), the effect is consistent across seeds, not a single lucky draw --")
    print("safe to report as a real finding. If std is large relative to the mean, or min/max")
    print("straddle zero, treat the original single-seed result as an overestimate and say so.")


def save_summary_csv(all_runs: list, summary: dict):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    fieldnames = (["seed"] +
                  [f"baseline_{k}" for k in METRIC_KEYS] +
                  [f"rich_{k}" for k in METRIC_KEYS] +
                  [f"paired_diff_{k}" for k in METRIC_KEYS])
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
