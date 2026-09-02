"""
run_det_backend_comparison.py
==============================

Compares ensemble vs. AutoGluon backends on all 12 DET conditions
(2 item types x 6 splits), using IDENTICAL simulated data/splits for
both, so any difference isolates the backend choice. Every prior DET
run used the ensemble backend for speed; this checks whether that
choice explains any of the gap vs. the paper.

Small-scale settings (150 Y/N Vocab / 80 ViC items, 6,000 sessions),
not full DET scale -- AutoGluon is much slower per fit, and small scale
already validated as a reliable stand-in for the Cold/Jump/Warm
*pattern* (see run_det_experiment_replicated.py).

Item selection is hardcoded to random regardless of
config.DET_ITEM_SELECTION -- adaptive selection is a separate, already-
answered question, and mixing it in would confound this comparison.

Needs autogluon.tabular installed first:
    pip install autogluon.tabular

Run with:
    python scripts/run_det_backend_comparison.py
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
COMPARISON_SEED = 42  # independent of config.DET_RANDOM_SEED -- this script's
                       # data is freshly simulated, not reused from another run

BACKENDS_TO_COMPARE = ["ensemble", "autogluon"]

# One entry per item type: (label, n_items, chance_value, items_per_session, seed_offset).
# seed_offset keeps Y/N Vocab and ViC's simulated data independent of each other.
ITEM_TYPE_SETTINGS = [
    ("Y/N Vocab", COMPARISON_N_YN_VOCAB_ITEMS, config.YN_CHANCE, config.YN_ITEMS_PER_SESSION, 0),
    ("ViC", COMPARISON_N_VIC_ITEMS, config.VIC_CHANCE, config.VIC_ITEMS_PER_SESSION, 100),
]

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
    """Builds items/sessions/responses ONCE and returns the 6
    (split_name, train, test) tuples, reused by both backends so they
    see byte-identical data."""
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

    # Look up each (backend, item_type, split)'s Pearson so the diff
    # loop below can find the matching ensemble/autogluon pair.
    by_key = {}
    for r in all_results:
        key = (r["backend"], r["item_type"], r["split"])
        by_key[key] = r["item_grade_pearson_r"]

    print("\nPer-condition Pearson difference (AutoGluon minus ensemble) -- positive means")
    print("AutoGluon calibrated better on this condition:")
    for item_type in ["Y/N Vocab", "ViC"]:
        for split in ["Cold", "Jump 20", "Jump 40", "Jump 80", "Warm 05-08", "Warm 05-15"]:
            ensemble_val = by_key.get(("ensemble", item_type, split))
            autogluon_val = by_key.get(("autogluon", item_type, split))
            if ensemble_val is not None and autogluon_val is not None:
                diff = autogluon_val - ensemble_val
                print(f"  {item_type:<10}{split:<12}{diff:+.3f}")


def main():
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)  # fresh file; rows appended as each condition finishes

    run_start_time = time.time()
    all_results = []

    for item_type_name, n_items, chance_value, items_per_session, seed_offset in ITEM_TYPE_SETTINGS:
        # Data/splits built ONCE per item type, reused for both backends,
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