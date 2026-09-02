"""
run_yn_vocab_nlp_experiment_replicated.py
============================================

Repeats run_yn_vocab_nlp_experiment.run_one_seed() across 5 seeds
instead of trusting the single run. Reports mean/std per condition,
plus the per-seed gain against the FIXED synthetic-feature baseline
(det_replication_summary.csv, 10 replications -- a fixed reference
point, not resampled here, so this is a one-sample comparison, not a
paired difference).

COST: this runs the full MCEM pipeline (up to 20 EM steps) for all 6
conditions per replication -- the single run took 19.3 minutes, so 5
replications is roughly 1.5-2 hours. Run unattended.

Run with:
    python scripts/run_yn_vocab_nlp_experiment_replicated.py

Individual condition rows are appended to results/yn_vocab_nlp_experiment.csv
(same file the single-run script uses) so nothing is lost if
interrupted. The aggregated summary is saved separately to
results/yn_vocab_nlp_experiment_replicated_summary.csv.
"""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from run_yn_vocab_nlp_experiment import (
    run_one_seed, append_result_row, PAPER_PEARSON, SYNTHETIC_BASELINE_PEARSON,
)

N_REPLICATIONS = 5
BASE_SEED = 3000  # distinct from every other seed used in this project (42, 500, 1000, 9000)

SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                             "yn_vocab_nlp_experiment_replicated_summary.csv")
METRIC_KEYS = ["test_loss_nll", "item_grade_pearson_r", "item_grade_spearman_r",
               "ability_recovery_pearson_r", "n_steps_run"]
CONDITION_ORDER = ["Cold", "Jump 20", "Jump 40", "Jump 80", "Warm 05-08", "Warm 05-15"]


def run_all_replications() -> dict:
    """Runs N_REPLICATIONS seeds. Returns {split_name: [row_dict, ...]}."""
    by_condition = {}
    run_start = time.time()

    for replication_index in range(N_REPLICATIONS):
        seed = BASE_SEED + replication_index * 100
        elapsed_so_far = (time.time() - run_start) / 60
        print(f"\n{'=' * 70}")
        print(f"REPLICATION {replication_index + 1}/{N_REPLICATIONS}  (seed={seed}, "
              f"{elapsed_so_far:.1f} min elapsed so far)")
        print(f"{'=' * 70}")

        rows = run_one_seed(seed, verbose=False)
        for row in rows:
            print(f"  {row['split']:<12} item_grade_pearson_r={row['item_grade_pearson_r']:.4f}  "
                  f"n_steps={row['n_steps_run']}  ({row['elapsed_seconds']:.0f}s)")

            gap_vs_paper = PAPER_PEARSON[row["split"]] - row["item_grade_pearson_r"]
            gain_vs_synthetic = row["item_grade_pearson_r"] - SYNTHETIC_BASELINE_PEARSON[row["split"]]
            append_result_row({
                **row,
                "paper_pearson": PAPER_PEARSON[row["split"]],
                "synthetic_baseline_pearson": SYNTHETIC_BASELINE_PEARSON[row["split"]],
                "gap_vs_paper": round(gap_vs_paper, 4),
                "gain_vs_synthetic": round(gain_vs_synthetic, 4),
            })

            split_name = row["split"]
            if split_name not in by_condition:
                by_condition[split_name] = []
            by_condition[split_name].append(row)

    return by_condition


def summarize(by_condition: dict) -> list:
    """by_condition: {split_name: [row_dict, ...]}. Returns one summary
    row per condition with mean/std per metric, plus gain vs. the fixed
    synthetic baseline (mean/min/max across replications)."""
    summary_rows = []

    for split_name, rows in by_condition.items():
        summary_row = {"split": split_name, "n_replications": len(rows)}

        for key in METRIC_KEYS:
            values = []
            for r in rows:
                values.append(r[key])
            values = np.array(values, dtype=float)
            summary_row[f"{key}_mean"] = values.mean()
            summary_row[f"{key}_std"] = values.std()

        pearson_values = []
        for r in rows:
            pearson_values.append(r["item_grade_pearson_r"])
        pearson_values = np.array(pearson_values)

        per_replication_rounded = []
        for p in pearson_values:
            per_replication_rounded.append(round(p, 4))
        summary_row["item_grade_pearson_r_per_replication"] = per_replication_rounded

        baseline = SYNTHETIC_BASELINE_PEARSON[split_name]
        summary_row["gain_vs_synthetic_mean"] = float(pearson_values.mean() - baseline)
        summary_row["gain_vs_synthetic_min"] = float(pearson_values.min() - baseline)
        summary_row["gain_vs_synthetic_max"] = float(pearson_values.max() - baseline)
        summary_row["gap_vs_paper_mean"] = float(PAPER_PEARSON[split_name] - pearson_values.mean())

        summary_rows.append(summary_row)

    return summary_rows


def print_summary_table(summary_rows: list):
    # Sort into Cold -> Jump 20/40/80 -> Warm order for display.
    def condition_sort_key(row):
        return CONDITION_ORDER.index(row["split"])

    summary_rows_sorted = sorted(summary_rows, key=condition_sort_key)

    print("\n" + "=" * 110)
    print(f"SUMMARY OVER {N_REPLICATIONS} REPLICATIONS (real NLP features, full MCEM pipeline)")
    print("=" * 110)
    header = (f"{'Split':<12}{'Pearson mean':>14}{'Pearson std':>13}{'Synthetic':>11}"
               f"{'Gain (mean)':>13}{'Gain (min)':>12}{'Gain (max)':>12}{'Gap vs paper':>14}")
    print(header)
    for row in summary_rows_sorted:
        print(f"{row['split']:<12}{row['item_grade_pearson_r_mean']:>14.4f}"
              f"{row['item_grade_pearson_r_std']:>13.4f}{SYNTHETIC_BASELINE_PEARSON[row['split']]:>11.3f}"
              f"{row['gain_vs_synthetic_mean']:>+13.4f}{row['gain_vs_synthetic_min']:>+12.4f}"
              f"{row['gain_vs_synthetic_max']:>+12.4f}{row['gap_vs_paper_mean']:>14.4f}")

    seeds_used = []
    for i in range(N_REPLICATIONS):
        seeds_used.append(BASE_SEED + i * 100)
    print(f"\nPer-replication seeds used: {seeds_used}")

    print("\nGain (min) positive -> every replication beat the baseline, a real effect.")
    print("Gain min/max straddling zero -> the single-run gain was likely just noise.")


def save_summary_csv(summary_rows: list):
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)

    fieldnames = ["split", "n_replications"]
    for key in METRIC_KEYS:
        fieldnames.append(f"{key}_mean")
    for key in METRIC_KEYS:
        fieldnames.append(f"{key}_std")
    fieldnames += ["gain_vs_synthetic_mean", "gain_vs_synthetic_min", "gain_vs_synthetic_max",
                   "gap_vs_paper_mean", "item_grade_pearson_r_per_replication"]

    with open(SUMMARY_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            row_out = dict(row)
            row_out["item_grade_pearson_r_per_replication"] = str(row_out["item_grade_pearson_r_per_replication"])
            writer.writerow(row_out)

    print(f"\nSaved replication summary to: {SUMMARY_PATH}")


def main():
    by_condition = run_all_replications()
    summary_rows = summarize(by_condition)
    print_summary_table(summary_rows)
    save_summary_csv(summary_rows)


if __name__ == "__main__":
    main()
