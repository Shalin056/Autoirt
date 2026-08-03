"""
Runs the same warm-start/cold-start scenario through both ML backends
(the hand-built ensemble and AutoGluon) on IDENTICAL simulated data, so
the comparison isolates just the backend -- nothing else differs between
the two runs.

This isolates whether a gap between this project's metrics and the
paper's reported numbers is coming from the AutoML backend choice or
from something else: comparing "ensemble" against "autogluon" on
identical data and splits answers that directly.

Needs autogluon.tabular installed first:
    pip install autogluon.tabular

Heavy install (multiple GB, a few minutes) and AutoGluon is also slower
per fit than the ensemble, so a full 400-item/20,000-session/5-seed run
here will take considerably longer than run_experiment.py does. Start
with the small smoke-test settings below before scaling up.

Run with:
    python scripts/run_backend_comparison.py
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from src.simulate import simulate_items, simulate_test_taker_abilities, simulate_test_responses
from src.autoirt_model import run_autoirt_calibration
from src.evaluate import evaluate_calibration
from run_experiment import (
    split_out_warm_start_test_sessions, split_out_cold_start_items, select_subset,
)

# Smoke-test settings by default -- small enough to finish in a few
# minutes so you can confirm AutoGluon actually runs before committing to
# the full comparison. Bump these to config.N_ITEMS / config.N_SESSIONS
# once the smoke test works.
COMPARISON_N_ITEMS = 60
COMPARISON_N_SESSIONS = 3000
COMPARISON_N_COLD_HOLDOUT = 15
COMPARISON_SEED = 42


def run_one_backend(backend: str, items, true_abilities, all_responses, n_items: int):
    """Runs warm-start and cold-start for one backend, returns both metric dicts."""
    print(f"\n{'=' * 60}\nBACKEND: {backend}\n{'=' * 60}")

    # Warm-start
    is_test_row = split_out_warm_start_test_sessions(
        all_responses, config.WARM_START_TEST_FRACTION, COMPARISON_SEED + 1,
    )
    train_responses = select_subset(all_responses, is_test_row, take_matching_rows=False)
    test_responses = select_subset(all_responses, is_test_row, take_matching_rows=True)
    warm_result = run_autoirt_calibration(
        train_responses, items, n_items=n_items, random_seed=COMPARISON_SEED, backend=backend,
    )
    warm_metrics = evaluate_calibration(
        test_responses, true_abilities,
        warm_result["discrimination"], warm_result["difficulty"], n_items=n_items,
    )
    warm_metrics["n_steps_run"] = warm_result["n_steps_run"]
    print("Warm-start:", warm_metrics)

    # Cold-start
    is_cold_item_row, _ = split_out_cold_start_items(
        all_responses, n_items, COMPARISON_N_COLD_HOLDOUT, COMPARISON_SEED + 2,
    )
    train_responses = select_subset(all_responses, is_cold_item_row, take_matching_rows=False)
    test_responses = select_subset(all_responses, is_cold_item_row, take_matching_rows=True)
    cold_result = run_autoirt_calibration(
        train_responses, items, n_items=n_items, random_seed=COMPARISON_SEED + 10, backend=backend,
    )
    cold_metrics = evaluate_calibration(
        test_responses, true_abilities,
        cold_result["discrimination"], cold_result["difficulty"], n_items=n_items,
    )
    cold_metrics["n_steps_run"] = cold_result["n_steps_run"]
    print("Cold-start:", cold_metrics)

    return warm_metrics, cold_metrics


def main():
    print(f"Simulating {COMPARISON_N_ITEMS} items, {COMPARISON_N_SESSIONS} sessions "
          f"(seed={COMPARISON_SEED})...")
    items = simulate_items(n_items=COMPARISON_N_ITEMS, random_seed=COMPARISON_SEED)
    true_abilities = simulate_test_taker_abilities(
        n_sessions=COMPARISON_N_SESSIONS, random_seed=COMPARISON_SEED + 1,
    )
    all_responses = simulate_test_responses(
        items, true_abilities, items_per_session=config.ITEMS_PER_SESSION,
        random_seed=COMPARISON_SEED + 2,
    )

    ensemble_warm, ensemble_cold = run_one_backend(
        "ensemble", items, true_abilities, all_responses, COMPARISON_N_ITEMS,
    )
    autogluon_warm, autogluon_cold = run_one_backend(
        "autogluon", items, true_abilities, all_responses, COMPARISON_N_ITEMS,
    )

    print("\n" + "=" * 70)
    print("BACKEND COMPARISON (same data, same splits, only the model differs)")
    print("=" * 70)
    header = f"{'Split':<12}{'Backend':<12}{'Loss':>8}{'Pearson':>10}{'Spearman':>10}{'AbilityRec':>12}{'Steps':>8}"
    print(header)
    rows = [
        ("Warm-start", "ensemble", ensemble_warm),
        ("Warm-start", "autogluon", autogluon_warm),
        ("Cold-start", "ensemble", ensemble_cold),
        ("Cold-start", "autogluon", autogluon_cold),
    ]
    for split_name, backend_name, m in rows:
        print(f"{split_name:<12}{backend_name:<12}{m['test_loss_nll']:>8.3f}"
              f"{m['item_grade_pearson_r']:>10.3f}{m['item_grade_spearman_r']:>10.3f}"
              f"{m['ability_recovery_pearson_r']:>12.3f}{m['n_steps_run']:>8}")

    results_path = os.path.join(os.path.dirname(__file__), "..", "results", "backend_comparison.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "backend"] + list(ensemble_warm.keys()))
        writer.writeheader()
        for split_name, backend_name, m in rows:
            writer.writerow({"split": split_name, "backend": backend_name, **m})
    print(f"\nSaved to: {results_path}")


if __name__ == "__main__":
    main()