"""
run_item_bank_sweep.py
=======================

Recreates the specific finding from the AutoIRT paper's simulation study
(see "Simulation study" section, Figure 4): holding session count fixed,
cold-start calibration quality is driven primarily by the TOTAL SIZE OF
THE ITEM BANK, not by how many test-takers you've simulated.

This is a good sanity check that the calibration logic is sound,
independent of whether the exact correlation numbers match the paper
(they won't match exactly -- different AutoML backend, smaller scale,
different random data -- but the SHAPE of the relationship should).

How to run
----------
    python scripts/run_item_bank_sweep.py

Edit ITEM_BANK_SIZES / N_SESSIONS / N_HOLDOUT_ITEMS below to change the
sweep. Results are saved to results/item_bank_sweep.csv and a plot to
results/item_bank_sweep.png.
"""

import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.simulate import simulate_items, simulate_test_taker_abilities, simulate_test_responses
from src.autoirt_model import run_autoirt_calibration, GradeEnsembleModel, DEFAULT_THETA_GRID, _fit_irt_curve_to_predictions
from src.evaluate import evaluate_calibration

# --- Sweep settings ---
# The paper sweeps item bank size at 100, 400, 1600. We keep those exact
# values (so the SHAPE of the result is comparable to Figure 4) but use a
# smaller, fixed session count than the paper's 10,000-160,000 range, to
# keep total runtime reasonable. Increase N_SESSIONS if you have time.
ITEM_BANK_SIZES = [100, 400, 1600]
N_SESSIONS = 3000
ITEMS_PER_SESSION = 8
N_HOLDOUT_ITEMS = 50
N_EM_STEPS = 4
RANDOM_SEED = 42


def run_cold_start_for_item_bank_size(n_items: int, seed: int) -> dict:
    """Simulate an item bank of size n_items + a disjoint N_HOLDOUT_ITEMS
    cold-start holdout set, calibrate on the training items only, then
    score the holdout items' calibration quality using ONLY their raw
    features (zero training responses for those items -- true cold-start)."""
    rng_items = np.random.default_rng(seed)

    total_items = n_items + N_HOLDOUT_ITEMS
    items = simulate_items(n_items=total_items, random_seed=seed)

    holdout_ids = rng_items.choice(total_items, size=N_HOLDOUT_ITEMS, replace=False)
    is_holdout = np.zeros(total_items, dtype=bool)
    is_holdout[holdout_ids] = True
    train_item_ids = np.where(~is_holdout)[0]

    train_items = {
        "feature_1": items["feature_1"][train_item_ids],
        "feature_2": items["feature_2"][train_item_ids],
        "discrimination": items["discrimination"][train_item_ids],
        "difficulty": items["difficulty"][train_item_ids],
        "chance": items["chance"][train_item_ids],
    }

    abilities = simulate_test_taker_abilities(n_sessions=N_SESSIONS, random_seed=seed + 1)
    responses = simulate_test_responses(
        train_items, abilities, items_per_session=ITEMS_PER_SESSION, random_seed=seed + 2
    )

    calibration_result = run_autoirt_calibration(
        responses, train_items, n_items=n_items,
        n_em_steps=N_EM_STEPS, random_seed=seed,
    )

    # Score the holdout items using the final ML model + their raw
    # features alone (zero training responses -- true cold-start).
    final_model = calibration_result["trained_model"]
    theta_grid = DEFAULT_THETA_GRID
    holdout_a = np.zeros(N_HOLDOUT_ITEMS)
    holdout_d = np.zeros(N_HOLDOUT_ITEMS)
    holdout_feature_1 = items["feature_1"][holdout_ids]
    holdout_feature_2 = items["feature_2"][holdout_ids]

    for i in range(N_HOLDOUT_ITEMS):
        query = np.column_stack([
            theta_grid,
            np.full(len(theta_grid), holdout_feature_1[i]),
            np.full(len(theta_grid), holdout_feature_2[i]),
        ])
        preds = final_model.predict_probability_correct(query)
        a, d = _fit_irt_curve_to_predictions(preds, theta_grid)
        holdout_a[i] = a
        holdout_d[i] = d

    holdout_items_true = {
        "feature_1": holdout_feature_1, "feature_2": holdout_feature_2,
        "discrimination": items["discrimination"][holdout_ids],
        "difficulty": items["difficulty"][holdout_ids],
        "chance": items["chance"][holdout_ids],
    }
    test_abilities = simulate_test_taker_abilities(n_sessions=2000, random_seed=seed + 3)
    test_responses = simulate_test_responses(
        holdout_items_true, test_abilities,
        items_per_session=min(N_HOLDOUT_ITEMS, 10), random_seed=seed + 4,
    )

    metrics = evaluate_calibration(
        test_responses, test_abilities, holdout_a, holdout_d, n_items=N_HOLDOUT_ITEMS,
    )
    return metrics


def main():
    results = []
    for n_items in ITEM_BANK_SIZES:
        print(f"\n=== Item bank size: {n_items} (+{N_HOLDOUT_ITEMS} cold-start holdout) ===")
        metrics = run_cold_start_for_item_bank_size(n_items, RANDOM_SEED)
        print(metrics)
        results.append({"n_items": n_items, **metrics})

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "item_bank_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved sweep results to {csv_path}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            [r["n_items"] for r in results],
            [r["item_grade_pearson_r"] for r in results],
            marker="o",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Number of items in training bank")
        ax.set_ylabel("Cold-start item-grade Pearson r")
        ax.set_title("Cold-start calibration quality vs. item bank size\n(paper's Figure 4 pattern)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(out_dir, "item_bank_sweep.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved plot to {png_path}")
    except ImportError:
        print("matplotlib not installed; skipping plot (CSV results are still saved).")


if __name__ == "__main__":
    main()