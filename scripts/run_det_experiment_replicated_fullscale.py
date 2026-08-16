"""
run_det_experiment_replicated_fullscale.py
============================================

Same idea as run_det_experiment_replicated.py, but at the REAL DET item
bank scale (config.N_YN_VOCAB_ITEMS / config.N_VIC_ITEMS -- 3290/585 --
and config.N_DET_SESSIONS -- 25000), instead of the smaller stand-in
settings used there.

Why this exists as a separate script: comparing the single full-scale
run in det_offline_analysis.csv against the 10x-replicated small-scale
run conflated two different things -- wrong item-bank scale (confounds
response density per item, see config.py's N_DET_SESSIONS comment) AND
no averaging (a single run's gap-by-condition pattern isn't more
trustworthy than any one of the small-scale run's 10 replications was).
This script fixes the second problem at the correct scale, so the two
effects aren't tangled together anymore.

N_REPLICATIONS is much lower here (3, not 10) purely because of cost:
each full-scale replication takes ~150 minutes with the ensemble
backend (see det_offline_analysis.csv's elapsed_seconds column), so 3
replications is already ~7.5 hours. This won't give as tight a std
estimate as the small-scale run's 10 replications, but it's a large
improvement over trusting a single run, and is what's feasible to run
overnight.

The summary is re-saved after EVERY replication (not just at the end),
so if this is interrupted partway through a multi-hour run, whatever
replications completed are not lost -- the saved file will just reflect
n_replications < N_REPLICATIONS until the run finishes.

Run with:
    python scripts/run_det_experiment_replicated_fullscale.py
"""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from run_det_experiment import run_one_item_type

N_REPLICATIONS = 3

BASE_SEED = 9000  # kept distinct from config.DET_RANDOM_SEED (42, the single
                   # full-scale run) and the small-scale replication's
                   # BASE_SEED (1000), so none of the three runs' data overlaps

# Separate summary file per item-selection mode, same reasoning as
# run_det_experiment.py's RESULTS_PATH -- otherwise re-running this
# script with config.DET_ITEM_SELECTION="adaptive" silently overwrites
# the "random"-mode summary this file used to contain.
_selection_mode = getattr(config, "DET_ITEM_SELECTION", "random")
_suffix = "" if _selection_mode == "random" else f"_{_selection_mode}"
SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                             f"det_replication_summary_fullscale{_suffix}.csv")

METRIC_KEYS = ["test_loss_nll", "item_grade_pearson_r", "item_grade_spearman_r",
               "ability_recovery_pearson_r", "n_steps_run"]

CONDITION_ORDER = ["Cold", "Jump 20", "Jump 40", "Jump 80", "Warm 05-08", "Warm 05-15"]

# Paper's Table 1 / Table 2 AutoIRT-row Pearson values, for the gap
# column in the printed summary -- same numbers already used in the
# emails, kept here so the script's own output can be checked against
# them directly instead of computing the gap by hand again afterward.
PAPER_PEARSON = {
    "Y/N Vocab": {"Cold": 0.785, "Jump 20": 0.936, "Jump 40": 0.955,
                  "Jump 80": 0.971, "Warm 05-08": 0.987, "Warm 05-15": 0.987},
    "ViC": {"Cold": 0.742, "Jump 20": 0.954, "Jump 40": 0.978,
            "Jump 80": 0.989, "Warm 05-08": 0.998, "Warm 05-15": 0.998},
}


def summarize(by_condition: dict) -> list:
    """Turns {(item_type, split): [rows]} into one summary row per
    condition with mean/std for each metric, plus the raw per-replication
    Pearson values and the gap against the paper's number."""
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
        summary_row["gap_vs_paper"] = (
            PAPER_PEARSON[item_type][split] - summary_row["item_grade_pearson_r_mean"]
        )
        summary_rows.append(summary_row)
    return summary_rows


def print_summary_table(summary_rows: list, n_completed_replications: int, n_sessions: int):
    summary_rows_sorted = sorted(
        summary_rows,
        key=lambda r: (r["item_type"], CONDITION_ORDER.index(r["split"])),
    )

    print("\n" + "=" * 110)
    print(f"SUMMARY OVER {n_completed_replications}/{N_REPLICATIONS} COMPLETED REPLICATIONS "
          f"(n_items={config.N_YN_VOCAB_ITEMS}/{config.N_VIC_ITEMS}, "
          f"n_sessions={n_sessions})")
    if n_completed_replications < N_REPLICATIONS:
        print("(partial -- run still in progress or was interrupted before finishing)")
    print("=" * 110)
    header = (f"{'Type':<10}{'Split':<12}{'Pearson (mean)':>16}{'Pearson (std)':>16}"
               f"{'Gap vs paper':>14}")
    print(header)
    for row in summary_rows_sorted:
        print(f"{row['item_type']:<10}{row['split']:<12}"
              f"{row['item_grade_pearson_r_mean']:>16.3f}"
              f"{row['item_grade_pearson_r_std']:>16.3f}"
              f"{row['gap_vs_paper']:>14.3f}")

    print("\nWhich condition has the largest gap vs. the paper, averaged over "
          f"{n_completed_replications} replication(s)?")
    for item_type in ["Y/N Vocab", "ViC"]:
        rows_for_type = [r for r in summary_rows_sorted if r["item_type"] == item_type]
        worst = max(rows_for_type, key=lambda r: r["gap_vs_paper"])
        best = min(rows_for_type, key=lambda r: r["gap_vs_paper"])
        print(f"  {item_type}: largest gap = {worst['split']} ({worst['gap_vs_paper']:.3f}), "
              f"smallest gap = {best['split']} ({best['gap_vs_paper']:.3f})")


def save_summary_csv(summary_rows: list):
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    fieldnames = ["item_type", "split", "n_replications"] + \
        [f"{k}_mean" for k in METRIC_KEYS] + [f"{k}_std" for k in METRIC_KEYS] + \
        ["gap_vs_paper", "item_grade_pearson_r_per_replication"]
    with open(SUMMARY_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            row_out = dict(row)
            row_out["item_grade_pearson_r_per_replication"] = str(row_out["item_grade_pearson_r_per_replication"])
            writer.writerow(row_out)


def main():
    by_condition = {}
    run_start_time = time.time()

    # Adaptive selection needs a different (larger) session count than
    # random selection to clear the same Jump-start data-volume bar --
    # see config.py's N_DET_SESSIONS vs N_DET_SESSIONS_ADAPTIVE comments.
    item_selection = getattr(config, "DET_ITEM_SELECTION", "random")
    n_sessions = (config.N_DET_SESSIONS_ADAPTIVE if item_selection == "adaptive"
                  else config.N_DET_SESSIONS)

    for replication_index in range(N_REPLICATIONS):
        seed = BASE_SEED + replication_index * 100
        elapsed_so_far = (time.time() - run_start_time) / 60
        print(f"\n{'=' * 70}")
        print(f"REPLICATION {replication_index + 1}/{N_REPLICATIONS}  (seed={seed}, "
              f"{elapsed_so_far:.1f} min elapsed so far)")
        print(f"{'=' * 70}")

        yn_results = run_one_item_type(
            "Y/N Vocab", config.N_YN_VOCAB_ITEMS, config.YN_CHANCE,
            config.YN_ITEMS_PER_SESSION, n_sessions, seed,
        )
        vic_results = run_one_item_type(
            "ViC", config.N_VIC_ITEMS, config.VIC_CHANCE,
            config.VIC_ITEMS_PER_SESSION, n_sessions, seed + 100,
        )

        for row in yn_results + vic_results:
            key = (row["item_type"], row["split"])
            by_condition.setdefault(key, []).append(row)

        # Re-save after every replication, not just at the end, so a
        # multi-hour run interrupted partway through doesn't lose the
        # replications that did complete.
        summary_rows = summarize(by_condition)
        print_summary_table(summary_rows, n_completed_replications=replication_index + 1,
                             n_sessions=n_sessions)
        save_summary_csv(summary_rows)
        print(f"\n(Saved progress after replication {replication_index + 1}/{N_REPLICATIONS} "
              f"to: {SUMMARY_PATH})")

    total_elapsed_hours = (time.time() - run_start_time) / 3600
    print(f"\nAll {N_REPLICATIONS} replications complete. Total time: {total_elapsed_hours:.1f} hours.")


if __name__ == "__main__":
    main()