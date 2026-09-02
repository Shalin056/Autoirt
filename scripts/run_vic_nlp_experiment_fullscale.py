"""
run_vic_nlp_experiment_fullscale.py
=======================================

ViC counterpart to run_yn_vocab_nlp_experiment_fullscale.py. Real DET
scale from config.py (585 items, 25,000 sessions), single run first,
not replicated -- full-scale replication is expensive (see that
script's docstring for the cost note).

Run with:
    python scripts/run_vic_nlp_experiment_fullscale.py

Expect this to take a long time. Run unattended.
"""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.simulate_det import simulate_det_sessions, simulate_det_responses, \
    COLD_JUMP_SPLIT_DAY, WARM_SPLIT_DAY_1, WARM_SPLIT_DAY_2
from src.simulate_vic_nlp import simulate_vic_items_nlp
from src.autoirt_model import run_autoirt_calibration
from src.evaluate import evaluate_calibration
from run_det_experiment import choose_pilot_and_operational_items, \
    split_cold_start, split_jump_start, split_warm_start

N_ITEMS = config.N_VIC_ITEMS
N_SESSIONS = config.N_DET_SESSIONS
ITEMS_PER_SESSION = config.VIC_ITEMS_PER_SESSION
RANDOM_SEED = config.DET_RANDOM_SEED
JUMP_START_R_VALUES = config.JUMP_START_R_VALUES
MAX_EM_STEPS = config.DET_MAX_EM_STEPS

VIC_FEATURE_NAMES = ["num_missing_chars", "proportion_vowels_missing", "target_zipf_frequency",
                      "sentence_mean_log_frequency", "position_normalized", "completion_predictability"]

# Paper's Table 2 AutoIRT-row Pearson values (ViC).
PAPER_PEARSON = {"Cold": 0.742, "Jump 20": 0.954, "Jump 40": 0.978,
                  "Jump 80": 0.989, "Warm 05-08": 0.998, "Warm 05-15": 0.998}

# Full-scale SYNTHETIC-feature baseline for ViC, read directly from
# det_replication_summary_fullscale.csv (3 replications, already established).
SYNTHETIC_BASELINE_PEARSON = {"Cold": 0.661, "Jump 20": 0.852, "Jump 40": 0.889,
                               "Jump 80": 0.907, "Warm 05-08": 0.926, "Warm 05-15": 0.929}

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                             "vic_nlp_experiment_fullscale.csv")
FIELDNAMES = ["split", "n_steps_run", "stopped_reason", "elapsed_seconds",
              "test_loss_nll", "item_grade_pearson_r", "item_grade_spearman_r",
              "ability_recovery_pearson_r", "paper_pearson", "synthetic_baseline_pearson",
              "gap_vs_paper", "gain_vs_synthetic"]


def append_result_row(row: dict):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_conditions(random_seed: int):
    print(f"Generating {N_ITEMS} real-content ViC items...")
    t0 = time.time()
    items = simulate_vic_items_nlp(n_items=N_ITEMS, random_seed=random_seed)
    print(f"  done in {time.time() - t0:.1f}s")

    sessions = simulate_det_sessions(n_sessions=N_SESSIONS, random_seed=random_seed + 1)
    theta = sessions["vic_theta"]

    all_responses = simulate_det_responses(
        items, theta, sessions["session_day"], ITEMS_PER_SESSION, random_seed=random_seed + 2,
    )
    pilot_item_ids, operational_item_ids = choose_pilot_and_operational_items(
        N_ITEMS, random_seed=random_seed + 3,
    )

    conditions = []
    train, test = split_cold_start(all_responses, operational_item_ids, COLD_JUMP_SPLIT_DAY)
    conditions.append(("Cold", train, test))
    for r in JUMP_START_R_VALUES:
        train, test = split_jump_start(all_responses, operational_item_ids, pilot_item_ids,
                                        COLD_JUMP_SPLIT_DAY, r)
        conditions.append((f"Jump {r}", train, test))
    train, test = split_warm_start(all_responses, WARM_SPLIT_DAY_1)
    conditions.append(("Warm 05-08", train, test))
    train, test = split_warm_start(all_responses, WARM_SPLIT_DAY_2)
    conditions.append(("Warm 05-15", train, test))

    return items, theta, conditions


def main():
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    print(f"Full DET scale: n_items={N_ITEMS}, n_sessions={N_SESSIONS}, seed={RANDOM_SEED}")
    run_start = time.time()

    items, theta, conditions = build_conditions(RANDOM_SEED)

    results = []
    for condition_index, (split_name, train_responses, test_responses) in enumerate(conditions, start=1):
        print(f"\n{'#' * 70}")
        print(f"# [{condition_index}/{len(conditions)}] {split_name}  "
              f"(train n={len(train_responses['grade'])}, test n={len(test_responses['grade'])})")
        print(f"{'#' * 70}")

        start_time = time.time()
        calibration_result = run_autoirt_calibration(
            train_responses, items, n_items=N_ITEMS, random_seed=RANDOM_SEED,
            backend="ensemble", max_em_steps=MAX_EM_STEPS,
            item_feature_names=VIC_FEATURE_NAMES,
        )
        elapsed = time.time() - start_time

        metrics = evaluate_calibration(
            test_responses, theta,
            calibration_result["discrimination"], calibration_result["difficulty"],
            n_items=N_ITEMS,
        )
        print(f"{split_name} metrics ({elapsed:.0f}s): {metrics}")

        paper_p = PAPER_PEARSON[split_name]
        synth_p = SYNTHETIC_BASELINE_PEARSON[split_name]
        gap_vs_paper = paper_p - metrics["item_grade_pearson_r"]
        gain_vs_synthetic = metrics["item_grade_pearson_r"] - synth_p
        print(f"  gap vs. paper ({paper_p:.3f}): {gap_vs_paper:+.3f}   "
              f"gain vs. synthetic-feature baseline ({synth_p:.3f}): {gain_vs_synthetic:+.3f}")

        row = {
            "split": split_name,
            "n_steps_run": calibration_result["n_steps_run"],
            "stopped_reason": calibration_result["stopped_reason"],
            "elapsed_seconds": round(elapsed, 1),
            "paper_pearson": paper_p,
            "synthetic_baseline_pearson": synth_p,
            "gap_vs_paper": round(gap_vs_paper, 4),
            "gain_vs_synthetic": round(gain_vs_synthetic, 4),
            **metrics,
        }
        append_result_row(row)
        results.append(row)

    total_elapsed_minutes = (time.time() - run_start) / 60
    print("\n" + "=" * 110)
    print("SUMMARY: real ViC NLP features (full DET scale) vs. paper vs. synthetic-feature baseline")
    print("=" * 110)
    header = f"{'Split':<12}{'This run':>10}{'Paper':>10}{'Synthetic':>11}{'Gap vs paper':>14}{'Gain vs synthetic':>19}"
    print(header)
    for row in results:
        print(f"{row['split']:<12}{row['item_grade_pearson_r']:>10.3f}{row['paper_pearson']:>10.3f}"
              f"{row['synthetic_baseline_pearson']:>11.3f}{row['gap_vs_paper']:>14.3f}"
              f"{row['gain_vs_synthetic']:>19.3f}")

    print(f"\nTotal run time: {total_elapsed_minutes:.1f} minutes")
    print(f"Results saved to: {RESULTS_PATH}")
    print("\nSingle run, not replicated. Same rule as everywhere else in this project: treat this")
    print("as a first look, not a final answer, until replicated.")


if __name__ == "__main__":
    main()
