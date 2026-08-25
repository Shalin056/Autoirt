"""
run_det_backend_comparison.py
==============================

Every DET-phase run so far (Week 3's report, both replication passes,
the adaptive-selection work) used config.DET_BACKEND_OVERRIDE =
"ensemble" for speed. Whether the ensemble backend itself explains part
of the jump-start/warm-start gap against the paper's numbers was flagged
back in the Week 3 report as an open lead ("I'd like your take on
whether that's worth chasing with more simulation tuning or a backend
change") and never actually tested at DET scale -- run_backend_comparision.py
only compares backends on the abstract single-item-type simulation
(run_experiment.py's setting), not the two-item-type DET setup with its
six split conditions.

This fills that gap directly: runs all 12 DET conditions (2 item types x
6 splits) through BOTH backends on IDENTICAL simulated data and splits,
so any difference in the results isolates the backend choice specifically
-- nothing else differs between the two passes.

Deliberately uses the SAME small-scale DET settings as
run_det_experiment_replicated.py (150 Y/N Vocab / 80 ViC items, 6,000
sessions) rather than the full 3,290/585-item DET scale. Two reasons:
  1. AutoGluon is meaningfully slower per fit than the ensemble backend
     (see run_backend_comparision.py's own note on this), and running
     12 conditions x AutoGluon at full DET scale would be a multi-hour
     to multi-day commitment just to test a single hypothesis.
  2. The small-scale settings are already validated as a reliable stand-in
     for the Cold/Jump/Warm *pattern* (run_det_experiment_replicated.py's
     10-replication check confirmed this), even though absolute gap size
     against the paper needs full scale to judge -- this script is asking
     "does the backend change the pattern/gap at all", not "what is the
     exact full-scale gap", so the small scale is the right tool here.

Item selection is hardcoded to uniform random regardless of
config.DET_ITEM_SELECTION, since adaptive-selection questions are a
separate, already-answered line of investigation (see Week 4) and mixing
that variable into a backend comparison would confound the result this
script is trying to isolate.

Needs autogluon.tabular installed first:
    pip install autogluon.tabular

Run with:
    python scripts/run_det_backend_comparison.py

Takes noticeably longer than a normal small-scale DET pass because of
the AutoGluon half -- results are saved incrementally (same pattern as
run_det_experiment.py's append_result_row) so a long run interrupted
partway through doesn't lose completed conditions.
"""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.simulate_det import simulate_det_items, simulate_det_sessions, simulate_det_responses, \
    COLD_JUMP_SPLIT_DAY, WARM_SPLIT_DAY_1, WARM_SPLIT_DAY_2
from src.autoirt_model import run_autoirt_calibration
from src.evaluate import evaluate_calibration
from run_det_experiment import (
    select_subset, choose_pilot_and_operational_items,
    split_cold_start, split_jump_start, split_warm_start,
)

COMPARISON_N_YN_VOCAB_ITEMS = 150
COMPARISON_N_VIC_ITEMS = 80
COMPARISON_N_DET_SESSIONS = 6000
COMPARISON_SEED = 42  # distinct from config.DET_RANDOM_SEED (42 is also the
                       # default there, but this script's data is independently
                       # simulated, not reused from any other run's CSV)

BACKENDS_TO_COMPARE = ["ensemble", "autogluon"]

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "det_backend_comparison.csv")
RESULT_FIELDNAMES = ["backend", "item_type", "split", "n_steps_run", "stopped_reason",
                      "elapsed_seconds", "test_loss_nll", "item_grade_pearson_r",
                      "item_grade_spearman_r", "ability_recovery_pearson_r"]


def append_result_row(row: dict):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_conditions(item_type_name: str, n_items: int, chance_value: float,
                      items_per_session: int, random_seed: int):
    """Simulates items/sessions/responses ONCE (uniform random selection,
    fixed seed) and returns the 6 (split_name, train, test) tuples shared
    by both backend passes, so the two backends see byte-identical data."""
    items = simulate_det_items(n_items=n_items, chance_value=chance_value, random_seed=random_seed)
    sessions = simulate_det_sessions(n_sessions=COMPARISON_N_DET_SESSIONS, random_seed=random_seed + 1)
    theta_key = "yn_theta" if item_type_name == "Y/N Vocab" else "vic_theta"
    theta = sessions[theta_key]

    all_responses = simulate_det_responses(
        items, theta, sessions["session_day"], items_per_session, random_seed=random_seed + 2,
    )

    pilot_item_ids, operational_item_ids = choose_pilot_and_operational_items(
        n_items, random_seed=random_seed + 3,
    )

    conditions = []
    train, test = split_cold_start(all_responses, operational_item_ids, COLD_JUMP_SPLIT_DAY)
    conditions.append(("Cold", train, test))
    for r in config.JUMP_START_R_VALUES:
        train, test = split_jump_start(all_responses, operational_item_ids, pilot_item_ids,
                                        COLD_JUMP_SPLIT_DAY, r)
        conditions.append((f"Jump {r}", train, test))
    train, test = split_warm_start(all_responses, WARM_SPLIT_DAY_1)
    conditions.append(("Warm 05-08", train, test))
    train, test = split_warm_start(all_responses, WARM_SPLIT_DAY_2)
    conditions.append(("Warm 05-15", train, test))

    return items, theta, conditions


def run_one_backend_one_item_type(backend: str, item_type_name: str, n_items: int,
                                   items, theta, conditions: list) -> list:
    print(f"\n{'#' * 70}")
    print(f"# BACKEND: {backend}  |  ITEM TYPE: {item_type_name}")
    print(f"{'#' * 70}")

    results = []
    for condition_index, (split_name, train_responses, test_responses) in enumerate(conditions, start=1):
        print(f"\n--- [{condition_index}/{len(conditions)}] {backend} / {item_type_name} / {split_name} "
              f"(train n={len(train_responses['grade'])}, test n={len(test_responses['grade'])}) ---")
        start_time = time.time()
        calibration_result = run_autoirt_calibration(
            train_responses, items, n_items=n_items, random_seed=COMPARISON_SEED,
            backend=backend, max_em_steps=config.DET_MAX_EM_STEPS,
        )
        elapsed_seconds = time.time() - start_time
        metrics = evaluate_calibration(
            test_responses, theta,
            calibration_result["discrimination"], calibration_result["difficulty"],
            n_items=n_items,
        )
        print(f"{backend} / {item_type_name} / {split_name} metrics ({elapsed_seconds:.0f}s):", metrics)
        row = {
            "backend": backend,
            "item_type": item_type_name,
            "split": split_name,
            "n_steps_run": calibration_result["n_steps_run"],
            "stopped_reason": calibration_result["stopped_reason"],
            "elapsed_seconds": round(elapsed_seconds, 1),
            **metrics,
        }
        append_result_row(row)
        results.append(row)

    return results


def print_summary_table(all_results: list):
    print("\n" + "=" * 100)
    print("SUMMARY: ensemble vs. AutoGluon on IDENTICAL DET-phase data and splits")
    print("=" * 100)
    header = f"{'Backend':<12}{'Type':<10}{'Split':<12}{'Pearson':>10}{'Spearman':>10}{'Loss':>8}{'Steps':>8}{'Secs':>8}"
    print(header)
    for row in all_results:
        print(f"{row['backend']:<12}{row['item_type']:<10}{row['split']:<12}"
              f"{row['item_grade_pearson_r']:>10.3f}{row['item_grade_spearman_r']:>10.3f}"
              f"{row['test_loss_nll']:>8.3f}{row['n_steps_run']:>8}{row['elapsed_seconds']:>8.0f}")

    print("\nPer-condition Pearson difference (AutoGluon minus ensemble) -- positive means")
    print("AutoGluon calibrated better on this condition:")
    by_key = {(r["backend"], r["item_type"], r["split"]): r["item_grade_pearson_r"] for r in all_results}
    for item_type in ["Y/N Vocab", "ViC"]:
        for split in ["Cold", "Jump 20", "Jump 40", "Jump 80", "Warm 05-08", "Warm 05-15"]:
            ensemble_val = by_key.get(("ensemble", item_type, split))
            autogluon_val = by_key.get(("autogluon", item_type, split))
            if ensemble_val is not None and autogluon_val is not None:
                diff = autogluon_val - ensemble_val
                print(f"  {item_type:<10}{split:<12}{diff:+.3f}")


def main():
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)  # fresh file for this run; rows appended as each condition finishes

    run_start_time = time.time()
    all_results = []

    for item_type_name, n_items, chance_value, items_per_session, seed_offset in [
        ("Y/N Vocab", COMPARISON_N_YN_VOCAB_ITEMS, config.YN_CHANCE, config.YN_ITEMS_PER_SESSION, 0),
        ("ViC", COMPARISON_N_VIC_ITEMS, config.VIC_CHANCE, config.VIC_ITEMS_PER_SESSION, 100),
    ]:
        # Data/splits built ONCE per item type and reused for both backends,
        # so the comparison isolates the backend and nothing else.
        items, theta, conditions = build_conditions(
            item_type_name, n_items, chance_value, items_per_session, COMPARISON_SEED + seed_offset,
        )
        for backend in BACKENDS_TO_COMPARE:
            all_results += run_one_backend_one_item_type(
                backend, item_type_name, n_items, items, theta, conditions,
            )

    total_elapsed_minutes = (time.time() - run_start_time) / 60
    print_summary_table(all_results)
    print(f"\nTotal run time: {total_elapsed_minutes:.1f} minutes")
    print(f"Results saved incrementally to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()