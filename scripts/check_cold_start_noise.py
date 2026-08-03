"""
check_cold_start_noise.py
==========================

Measures how much Pearson r bounces around by chance alone at each item
bank size, so a difference between item bank sizes (as seen in
run_item_bank_sweep.py) can be checked against the noise floor before
being treated as a real effect. run_item_bank_sweep.py uses
N_HOLDOUT_ITEMS=50 (paper uses 1000) and a single seed per item count,
both of which inflate how much a given Pearson difference could be pure
sampling variance rather than a real effect.

This uses the FAST ensemble backend (not autogluon) and a bigger holdout
set, repeated across a few seeds, purely to measure the noise floor. It
is NOT trying to reproduce the paper's exact numbers -- it's a
noise-floor check, a few minutes instead of another multi-hour AutoGluon
run.

Run with:
    python scripts/check_cold_start_noise.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.simulate import simulate_items, simulate_test_taker_abilities, simulate_test_responses
from src.autoirt_model import run_autoirt_calibration, DEFAULT_THETA_GRID, _fit_irt_curve_to_predictions
from src.evaluate import evaluate_calibration

N_HOLDOUT_ITEMS = 200   # up from 50 -- still short of the paper's 1000, but
                        # enough to meaningfully shrink sampling noise
N_SESSIONS = 10000
ITEMS_PER_SESSION = 8
N_EM_STEPS = 4
N_SEEDS_PER_SIZE = 3
ITEM_BANK_SIZES = [400, 1600]


def run_once(n_items: int, seed: int) -> float:
    """Same cold-start procedure as run_item_bank_sweep.py, ensemble
    backend, returns just item_grade_pearson_r."""
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
        train_items, abilities, items_per_session=ITEMS_PER_SESSION, random_seed=seed + 2,
    )

    calibration_result = run_autoirt_calibration(
        responses, train_items, n_items=n_items,
        n_em_steps=N_EM_STEPS, random_seed=seed, backend="ensemble",
    )

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
    return metrics["item_grade_pearson_r"]


def main():
    print(f"N_HOLDOUT_ITEMS={N_HOLDOUT_ITEMS}, {N_SEEDS_PER_SIZE} seeds per item bank size, "
          f"ensemble backend (fast, noise-check only -- not a real accuracy comparison)\n")
    for n_items in ITEM_BANK_SIZES:
        pearsons = [run_once(n_items, seed=1000 * n_items + s) for s in range(N_SEEDS_PER_SIZE)]
        pearsons = np.array(pearsons)
        print(f"n_items={n_items:>5}: Pearson r per seed = {np.round(pearsons, 3).tolist()}  "
              f"-> mean={pearsons.mean():.3f}, std={pearsons.std():.3f}")

    print("\nIf the two mean +/- std ranges overlap substantially, the 400-vs-1600")
    print("'inversion' seen with N_HOLDOUT_ITEMS=50 was very likely just noise, not")
    print("a real effect -- worth saying so plainly rather than chasing a cause")
    print("that may not exist. If they're clearly separated even with this larger,")
    print("repeated sample, that's a real, reportable finding worth investigating further.")


if __name__ == "__main__":
    main()