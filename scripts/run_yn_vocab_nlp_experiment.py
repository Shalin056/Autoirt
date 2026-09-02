"""
run_yn_vocab_nlp_experiment.py
================================

Runs the Y/N Vocab NLP item generator through the real calibration
pipeline (full MCEM loop, not the single-M-step diagnostic shortcuts
used elsewhere) and evaluates with the project's official metrics
(evaluate_calibration), the same ones compared against the paper's
Table 1 throughout this project.

Small-scale convention (150 items, 6,000 sessions), single seed, not
replicated -- replicate only if this looks promising.

Reuses the existing DET split logic unchanged; only the item generator
(simulate_yn_vocab_items_nlp) and item_feature_names differ from the
synthetic-feature pipeline.

Y/N Vocab only -- ViC has its own script (needs a different generator).

Run with:
    python scripts/run_yn_vocab_nlp_experiment.py

Prints each condition against two reference points: the paper's Table 1
numbers, and this project's small-scale synthetic-feature baseline
(det_replication_summary.csv, 10 replications).
"""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_det import simulate_det_sessions, simulate_det_responses, \
    COLD_JUMP_SPLIT_DAY, WARM_SPLIT_DAY_1, WARM_SPLIT_DAY_2
from src.simulate_yn_vocab_nlp import simulate_yn_vocab_items_nlp
from src.autoirt_model import run_autoirt_calibration
from src.evaluate import evaluate_calibration
from run_det_experiment import choose_pilot_and_operational_items, \
    split_cold_start, split_jump_start, split_warm_start

N_ITEMS = 150
N_SESSIONS = 6000
ITEMS_PER_SESSION = 18
CHANCE_VALUE = 0.25
RANDOM_SEED = 42
JUMP_START_R_VALUES = [20, 40, 80]
MAX_EM_STEPS = 20

NLP_FEATURE_NAMES = ["is_real", "length", "zipf_frequency",
                      "mean_bigram_log_freq", "mean_trigram_log_freq"]

# Paper's Table 1 AutoIRT-row Pearson values (Y/N Vocab).
PAPER_PEARSON = {"Cold": 0.785, "Jump 20": 0.936, "Jump 40": 0.955,
                  "Jump 80": 0.971, "Warm 05-08": 0.987, "Warm 05-15": 0.987}

# This project's small-scale synthetic-feature baseline (10 replications,
# already established) -- the "before" comparison for real features.
SYNTHETIC_BASELINE_PEARSON = {"Cold": 0.619, "Jump 20": 0.857, "Jump 40": 0.898,
                               "Jump 80": 0.928, "Warm 05-08": 0.965, "Warm 05-15": 0.965}

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                             "yn_vocab_nlp_experiment.csv")
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
    """Builds items/sessions/responses and returns the 6
    (split_name, train, test) tuples for this seed."""
    items = simulate_yn_vocab_items_nlp(n_items=N_ITEMS, random_seed=random_seed)
    sessions = simulate_det_sessions(n_sessions=N_SESSIONS, random_seed=random_seed + 1)
    theta = sessions["yn_theta"]

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


def run_one_seed(random_seed: int, verbose: bool = True) -> list:
    """Runs all 6 conditions for one seed. Returns result rows without
    the paper/synthetic comparison columns -- the caller adds those, so
    a replication script applies the same comparison to every seed."""
    items, theta, conditions = build_conditions(random_seed)
    if verbose:
        print(f"Real word bank ready: {int(items['is_real'].sum())} real / "
              f"{int((1 - items['is_real']).sum())} fake words")

    rows = []
    for condition_index, (split_name, train_responses, test_responses) in enumerate(conditions, start=1):
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# [{condition_index}/{len(conditions)}] {split_name}  "
                  f"(train n={len(train_responses['grade'])}, test n={len(test_responses['grade'])})")
            print(f"{'#' * 70}")

        start_time = time.time()
        calibration_result = run_autoirt_calibration(
            train_responses, items, n_items=N_ITEMS, random_seed=random_seed,
            backend="ensemble", max_em_steps=MAX_EM_STEPS,
            item_feature_names=NLP_FEATURE_NAMES,
        )
        elapsed = time.time() - start_time

        metrics = evaluate_calibration(
            test_responses, theta,
            calibration_result["discrimination"], calibration_result["difficulty"],
            n_items=N_ITEMS,
        )
        if verbose:
            print(f"{split_name} metrics ({elapsed:.0f}s): {metrics}")

        rows.append({
            "split": split_name,
            "n_steps_run": calibration_result["n_steps_run"],
            "stopped_reason": calibration_result["stopped_reason"],
            "elapsed_seconds": round(elapsed, 1),
            **metrics,
        })

    return rows


def main():
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    print(f"Building items and responses (n_items={N_ITEMS}, n_sessions={N_SESSIONS}, "
          f"seed={RANDOM_SEED})...")
    run_start = time.time()
    rows = run_one_seed(RANDOM_SEED, verbose=True)

    results = []
    for row in rows:
        split_name = row["split"]
        paper_p = PAPER_PEARSON[split_name]
        synth_p = SYNTHETIC_BASELINE_PEARSON[split_name]
        gap_vs_paper = paper_p - row["item_grade_pearson_r"]
        gain_vs_synthetic = row["item_grade_pearson_r"] - synth_p

        print(f"  gap vs. paper ({paper_p:.3f}): {gap_vs_paper:+.3f}   "
              f"gain vs. synthetic-feature baseline ({synth_p:.3f}): {gain_vs_synthetic:+.3f}")

        full_row = {
            **row,
            "paper_pearson": paper_p,
            "synthetic_baseline_pearson": synth_p,
            "gap_vs_paper": round(gap_vs_paper, 4),
            "gain_vs_synthetic": round(gain_vs_synthetic, 4),
        }
        append_result_row(full_row)
        results.append(full_row)

    total_elapsed_minutes = (time.time() - run_start) / 60
    print("\n" + "=" * 110)
    print("SUMMARY: real NLP features vs. paper vs. synthetic-feature baseline (all Pearson)")
    print("=" * 110)
    header = f"{'Split':<12}{'This run':>10}{'Paper':>10}{'Synthetic':>11}{'Gap vs paper':>14}{'Gain vs synthetic':>19}"
    print(header)
    for row in results:
        print(f"{row['split']:<12}{row['item_grade_pearson_r']:>10.3f}{row['paper_pearson']:>10.3f}"
              f"{row['synthetic_baseline_pearson']:>11.3f}{row['gap_vs_paper']:>14.3f}"
              f"{row['gain_vs_synthetic']:>19.3f}")

    print(f"\nTotal run time: {total_elapsed_minutes:.1f} minutes")
    print(f"Results saved to: {RESULTS_PATH}")
    print("\nConsistently positive 'gain vs synthetic' -> real features close the gap. This is")
    print("the real-pipeline confirmation of what the diagnostic proxy suggested earlier.")


if __name__ == "__main__":
    main()
