"""
run_det_experiment_replicated.py
=================================

Runs the DET offline analysis N_REPLICATIONS times with different seeds
and reports mean +/- std per (item_type, split) -- Dr. Zheng's request
to check whether the Cold < Jump < Warm pattern (and the relative
ordering vs. the paper) holds up on average, or was partly an artifact
of a single run's random data.

Uses a SMALLER item bank / session count than the full-scale DET run
(config.N_YN_VOCAB_ITEMS / config.N_VIC_ITEMS / config.N_DET_SESSIONS),
by the same logic as check_cold_start_noise.py: the full-scale run
takes ~150 minutes for ONE pass (see det_offline_analysis.csv), so 10+
replications at that scale would take the better part of a day. The
smaller settings below (150 Y/N Vocab / 80 ViC items, 6000 sessions)
were checked against the same post-split-responses-per-item formula
used in config.py and clear R=80 with a comfortable margin for both
item types, so this settles the "is the R=80 condition even valid"
question before spending replication time on it. This is a pattern
check, not an attempt to match the paper's numbers at full DET scale --
run_det_experiment.py at the settings in config.py is still what
produces the numbers to compare against Table 1/Table 2 directly.

Uses the ensemble backend by default (fast; see config.DET_BACKEND_OVERRIDE)
so N_REPLICATIONS replications x 12 conditions finishes in minutes, not
hours.

Each replication's individual condition results are also appended to
results/det_offline_analysis.csv via the reused run_one_item_type()
(same as a normal run_det_experiment.py pass would do) -- this script's
own output is the aggregated mean/std, saved separately to
results/det_replication_summary.csv so it isn't mixed with single-run
data.

Run with:
    python scripts/run_det_experiment_replicated.py
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from run_det_experiment import run_one_item_type
import config as _config

N_REPLICATIONS = 10

REPLICATION_N_YN_VOCAB_ITEMS = 150
REPLICATION_N_VIC_ITEMS = 80
REPLICATION_N_DET_SESSIONS = 6000
YN_ITEMS_PER_SESSION = 18
VIC_ITEMS_PER_SESSION = 9
YN_CHANCE = 0.25
VIC_CHANCE = 0.0
BASE_SEED = 1000  # kept far from config.DET_RANDOM_SEED (42) so this run's
                   # data doesn't accidentally coincide with a single-run pass

# Separate summary file per item-selection mode, same reasoning as
# run_det_experiment.py's RESULTS_PATH -- otherwise re-running this
# script with config.DET_ITEM_SELECTION="adaptive" silently overwrites
# the "random"-mode summary this file used to contain.
_selection_mode = getattr(_config, "DET_ITEM_SELECTION", "random")
_suffix = "" if _selection_mode == "random" else f"_{_selection_mode}"
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                             f"det_replication_summary{_suffix}.csv")

METRIC_KEYS = ["test_loss_nll", "item_grade_pearson_r", "item_grade_spearman_r",
               "ability_recovery_pearson_r", "n_steps_run"]


def run_all_replications():
    """Returns {(item_type, split): [row_dict, row_dict, ...]} across all
    N_REPLICATIONS runs, one row_dict per replication."""
    by_condition = {}

    for replication_index in range(N_REPLICATIONS):
        seed = BASE_SEED + replication_index * 100  # spread seeds apart per replication
        print(f"\n{'=' * 70}")
        print(f"REPLICATION {replication_index + 1}/{N_REPLICATIONS}  (seed={seed})")
        print(f"{'=' * 70}")

        yn_results = run_one_item_type(
            "Y/N Vocab", REPLICATION_N_YN_VOCAB_ITEMS, YN_CHANCE,
            YN_ITEMS_PER_SESSION, REPLICATION_N_DET_SESSIONS, seed,
        )
        vic_results = run_one_item_type(
            "ViC", REPLICATION_N_VIC_ITEMS, VIC_CHANCE,
            VIC_ITEMS_PER_SESSION, REPLICATION_N_DET_SESSIONS, seed + 100,
        )

        for row in yn_results + vic_results:
            key = (row["item_type"], row["split"])
            by_condition.setdefault(key, []).append(row)

    return by_condition


def summarize(by_condition: dict) -> list:
    """Turns {(item_type, split): [rows]} into one summary row per
    condition with mean/std for each metric, plus the raw per-replication
    values (so a suspicious std can be traced back to which replication
    drove it, not just flagged as "high variance")."""
    summary_rows = []
    for (item_type, split), rows in by_condition.items():
        summary_row = {"item_type": item_type, "split": split, "n_replications": len(rows)}
        for key in METRIC_KEYS:
            values = np.array([r[key] for r in rows], dtype=float)
            summary_row[f"{key}_mean"] = values.mean()
            summary_row[f"{key}_std"] = values.std()
        summary_row["item_grade_pearson_r_per_replication"] = [
            round(r["item_grade_pearson_r"], 3) for r in rows
        ]
        summary_rows.append(summary_row)
    return summary_rows


def print_summary_table(summary_rows: list):
    condition_order = ["Cold", "Jump 20", "Jump 40", "Jump 80", "Warm 05-08", "Warm 05-15"]
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (r["item_type"], condition_order.index(r["split"])),
    )

    print("\n" + "=" * 100)
    print(f"SUMMARY OVER {N_REPLICATIONS} REPLICATIONS "
          f"(n_items={REPLICATION_N_YN_VOCAB_ITEMS}/{REPLICATION_N_VIC_ITEMS}, "
          f"n_sessions={REPLICATION_N_DET_SESSIONS})")
    print("=" * 100)
    header = f"{'Type':<10}{'Split':<12}{'Pearson (mean)':>16}{'Pearson (std)':>16}{'Spearman (mean)':>18}"
    print(header)
    for row in summary_rows_sorted:
        print(f"{row['item_type']:<10}{row['split']:<12}"
              f"{row['item_grade_pearson_r_mean']:>16.3f}"
              f"{row['item_grade_pearson_r_std']:>16.3f}"
              f"{row['item_grade_spearman_r_mean']:>18.3f}")

    print("\nPattern check: within each item type, does mean Pearson increase")
    print("monotonically Cold -> Jump 20 -> Jump 40 -> Jump 80 -> Warm? "
          "(matches the paper's ordering if so)")
    for item_type in ["Y/N Vocab", "ViC"]:
        means = [
            next(r["item_grade_pearson_r_mean"] for r in summary_rows_sorted
                 if r["item_type"] == item_type and r["split"] == split)
            for split in condition_order
        ]
        is_monotonic = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
        print(f"  {item_type}: {[round(m, 3) for m in means]}  "
              f"-> {'monotonic (matches paper pattern)' if is_monotonic else 'NOT monotonic -- inspect per-replication values'}")


def save_summary_csv(summary_rows: list):
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    fieldnames = ["item_type", "split", "n_replications"] + \
        [f"{k}_mean" for k in METRIC_KEYS] + [f"{k}_std" for k in METRIC_KEYS] + \
        ["item_grade_pearson_r_per_replication"]
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