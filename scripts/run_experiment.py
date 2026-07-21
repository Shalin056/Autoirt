"""
run_experiment.py
==================

This is the script you actually run. It ties together everything in
`src/` to reproduce the paper's simulation study, in two parts:

  1. WARM-START: calibrate on most sessions, test on held-out sessions
     (same item bank, just unseen test-takers).

  2. COLD-START: calibrate on most ITEMS, then test on entirely new items
     that had ZERO training responses (this is the harder, more
     interesting case, and the main point of the AutoIRT paper).

How to run this
----------------
From the project's top-level folder, run:

    python scripts/run_experiment.py

All the experiment settings (number of items, sessions, etc.) are in
`config.py` at the top level -- edit that file if you want a bigger or
smaller run.
"""

import sys
import os
import csv

# Allow this script to import the `src` package from the project root,
# regardless of which folder you run it from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from src.simulate import simulate_items, simulate_test_taker_abilities, simulate_test_responses
from src.autoirt_model import run_autoirt_calibration
from src.evaluate import evaluate_calibration


def split_out_warm_start_test_sessions(responses: dict, test_fraction: float, random_seed: int):
    """
    Randomly holds out a fraction of SESSIONS (test-takers) for testing.
    The item bank is identical between train and test in this split --
    only the test-takers are new. This is the "warm-start" scenario.
    """
    rng = np.random.default_rng(random_seed)
    n_sessions = responses["session_id"].max() + 1
    n_test_sessions = int(n_sessions * test_fraction)
    test_session_ids = rng.choice(n_sessions, size=n_test_sessions, replace=False)
    is_test_row = np.isin(responses["session_id"], test_session_ids)
    return is_test_row


def split_out_cold_start_items(responses: dict, n_items: int, n_holdout_items: int, random_seed: int):
    """
    Randomly holds out a set of ITEMS entirely: none of their responses
    are used for training. This simulates brand-new pilot questions that
    have never been shown to any test-taker before. This is the
    "cold-start" scenario -- the harder and more useful case, since it
    tests whether the model can calibrate a new item using ONLY its raw
    features, with no response data at all.
    """
    rng = np.random.default_rng(random_seed)
    holdout_item_ids = rng.choice(n_items, size=n_holdout_items, replace=False)
    is_test_row = np.isin(responses["item_id"], holdout_item_ids)
    return is_test_row, holdout_item_ids


def select_subset(responses: dict, row_mask: np.ndarray, take_matching_rows: bool):
    """
    Small helper: returns the subset of `responses` where row_mask is True
    (if take_matching_rows=True) or False (if take_matching_rows=False).
    """
    final_mask = row_mask if take_matching_rows else ~row_mask
    return {
        "session_id": responses["session_id"][final_mask],
        "item_id": responses["item_id"][final_mask],
        "grade": responses["grade"][final_mask],
        "true_theta": responses["true_theta"][final_mask],
    }


def run_warm_start_experiment(items, true_abilities, all_responses):
    print("\n" + "=" * 60)
    print("WARM-START EXPERIMENT")
    print("(same items, unseen test-takers)")
    print("=" * 60)

    is_test_row = split_out_warm_start_test_sessions(
        all_responses, config.WARM_START_TEST_FRACTION, config.RANDOM_SEED + 1,
    )
    train_responses = select_subset(all_responses, is_test_row, take_matching_rows=False)
    test_responses = select_subset(all_responses, is_test_row, take_matching_rows=True)

    calibration_result = run_autoirt_calibration(
        train_responses, items, n_items=config.N_ITEMS,
        n_em_steps=config.N_EM_STEPS, random_seed=config.RANDOM_SEED,
    )

    metrics = evaluate_calibration(
        test_responses, true_abilities,
        calibration_result["discrimination"], calibration_result["difficulty"],
        n_items=config.N_ITEMS,
    )
    print("Warm-start metrics:", metrics)
    return metrics


def run_cold_start_experiment(items, true_abilities, all_responses):
    print("\n" + "=" * 60)
    print("COLD-START EXPERIMENT")
    print("(brand-new items with zero training responses)")
    print("=" * 60)

    is_cold_item_row, holdout_item_ids = split_out_cold_start_items(
        all_responses, config.N_ITEMS, config.N_COLD_START_HOLDOUT_ITEMS, config.RANDOM_SEED + 2,
    )
    train_responses = select_subset(all_responses, is_cold_item_row, take_matching_rows=False)
    test_responses = select_subset(all_responses, is_cold_item_row, take_matching_rows=True)

    calibration_result = run_autoirt_calibration(
        train_responses, items, n_items=config.N_ITEMS,
        n_em_steps=config.N_EM_STEPS, random_seed=config.RANDOM_SEED + 10,
    )

    metrics = evaluate_calibration(
        test_responses, true_abilities,
        calibration_result["discrimination"], calibration_result["difficulty"],
        n_items=config.N_ITEMS,
    )
    print("Cold-start metrics:", metrics)
    return metrics


def print_summary_table(warm_metrics: dict, cold_metrics: dict):
    print("\n" + "=" * 60)
    print("SUMMARY (compare to the paper's Table 1 pattern)")
    print("=" * 60)
    header = f"{'Split':<12}{'Test Loss':>12}{'Pearson':>10}{'Spearman':>10}{'Theta Corr':>12}"
    print(header)
    for split_name, metrics in [("Warm-start", warm_metrics), ("Cold-start", cold_metrics)]:
        print(
            f"{split_name:<12}"
            f"{metrics['test_loss_nll']:>12.3f}"
            f"{metrics['item_grade_pearson_r']:>10.3f}"
            f"{metrics['item_grade_spearman_r']:>10.3f}"
            f"{metrics['ability_recovery_pearson_r']:>12.3f}"
        )


def save_metrics_to_csv(warm_metrics: dict, cold_metrics: dict, output_path: str):
    """Saves the final metrics table to results/metrics.csv for later reference."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["split"] + list(warm_metrics.keys())

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"split": "warm_start", **warm_metrics})
        writer.writerow({"split": "cold_start", **cold_metrics})

    print(f"\nSaved metrics to: {output_path}")


def run_one_full_experiment(base_seed: int):
    """
    Runs one complete pass of the experiment (simulate data, then both the
    warm-start and cold-start calibration experiments) using `base_seed` as
    the starting point for all randomness in this pass.

    Returns (warm_metrics, cold_metrics) for this one run.
    """
    print(f"\n>>> Simulating {config.N_ITEMS} items and {config.N_SESSIONS} "
          f"test-taker sessions (seed={base_seed})...")
    items = simulate_items(n_items=config.N_ITEMS, random_seed=base_seed)
    true_abilities = simulate_test_taker_abilities(
        n_sessions=config.N_SESSIONS, random_seed=base_seed + 1,
    )
    all_responses = simulate_test_responses(
        items, true_abilities, items_per_session=config.ITEMS_PER_SESSION,
        random_seed=base_seed + 2,
    )

    warm_metrics = run_warm_start_experiment(items, true_abilities, all_responses)
    cold_metrics = run_cold_start_experiment(items, true_abilities, all_responses)
    return warm_metrics, cold_metrics


def summarize_repeated_runs(all_warm_metrics: list, all_cold_metrics: list):
    """
    Given a list of metric dictionaries from several independent runs,
    computes the mean and standard deviation of each metric across runs.
    This tells you not just "what number did we get" but "how much does
    that number bounce around by chance alone."
    """
    def mean_and_std(metric_dicts, key):
        values = np.array([d[key] for d in metric_dicts])
        return values.mean(), values.std()

    metric_keys = list(all_warm_metrics[0].keys())

    print("\n" + "=" * 70)
    print(f"AVERAGED OVER {len(all_warm_metrics)} INDEPENDENT RUN(S)")
    print("=" * 70)
    header = f"{'Split':<12}{'Metric':<28}{'Mean':>10}{'Std Dev':>10}"
    print(header)
    for split_name, metric_dicts in [("Warm-start", all_warm_metrics), ("Cold-start", all_cold_metrics)]:
        for key in metric_keys:
            mean_value, std_value = mean_and_std(metric_dicts, key)
            print(f"{split_name:<12}{key:<28}{mean_value:>10.3f}{std_value:>10.3f}")

    warm_summary = {
        key: dict(zip(["mean", "std"], mean_and_std(all_warm_metrics, key)))
        for key in metric_keys
    }
    cold_summary = {
        key: dict(zip(["mean", "std"], mean_and_std(all_cold_metrics, key)))
        for key in metric_keys
    }
    return warm_summary, cold_summary


def save_repeated_metrics_to_csv(all_warm_metrics: list, all_cold_metrics: list, output_path: str):
    """Saves EVERY individual run's metrics (not just the average) to CSV,
    so you can see the full spread of results, not just a summary."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metric_keys = list(all_warm_metrics[0].keys())
    fieldnames = ["split", "run_number"] + metric_keys

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for run_number, metrics in enumerate(all_warm_metrics, start=1):
            writer.writerow({"split": "warm_start", "run_number": run_number, **metrics})
        for run_number, metrics in enumerate(all_cold_metrics, start=1):
            writer.writerow({"split": "cold_start", "run_number": run_number, **metrics})

    print(f"\nSaved per-run metrics to: {output_path}")


def main():
    n_repeats = getattr(config, "N_REPEATS", 1)
    all_warm_metrics = []
    all_cold_metrics = []

    for run_index in range(n_repeats):
        run_seed = config.RANDOM_SEED + (run_index * 1000)  # spread seeds far apart per run
        warm_metrics, cold_metrics = run_one_full_experiment(run_seed)
        all_warm_metrics.append(warm_metrics)
        all_cold_metrics.append(cold_metrics)

    if n_repeats == 1:
        print_summary_table(all_warm_metrics[0], all_cold_metrics[0])
        results_path = os.path.join(os.path.dirname(__file__), "..", "results", "metrics.csv")
        save_metrics_to_csv(all_warm_metrics[0], all_cold_metrics[0], results_path)
    else:
        summarize_repeated_runs(all_warm_metrics, all_cold_metrics)
        results_path = os.path.join(os.path.dirname(__file__), "..", "results", "metrics_all_runs.csv")
        save_repeated_metrics_to_csv(all_warm_metrics, all_cold_metrics, results_path)


if __name__ == "__main__":
    main()
