"""
check_item_id_feature_sensitivity.py
======================================

The AutoIRT paper's Method section says item ID is passed to the AutoML
predictor as a feature (like a random effect). Our FEATURE_COLUMNS
(theta, feature_1, feature_2) doesn't include it -- a real gap from the
paper's stated method. This tests whether adding it as a one-hot column
helps.

Item ID can only help items the model saw during training, so this
tests Warm-start recalibration specifically (operational items with
real response history), not Cold-start (held-out items were never
trained on, so their ID has nothing to look up).

Same isolation strategy as check_feature_richness_sensitivity.py: TRUE
theta for training (removes MCEM noise as a confound), fitted vs. TRUE
item parameters compared directly (not the pipeline's usual
item_grade_pearson_r).

One-hot encoded across operational item IDs (not a raw integer column,
which would let the trees split on arbitrary numeric thresholds instead
of treating each ID as its own category).

Run with:
    python scripts/check_item_id_feature_sensitivity.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.simulate import three_parameter_logistic, simulate_test_taker_abilities
from src.simulate_det import simulate_det_items

N_ITEMS = 150
N_SESSIONS = 6000
ITEMS_PER_SESSION = 18
CHANCE_VALUE = 0.25
RANDOM_SEED = 42


def choose_pilot_and_operational_items(n_items: int, random_seed: int):
    """Same 50/50 split used throughout the project, reimplemented
    locally to avoid pulling in run_det_experiment.py's dependencies."""
    rng = np.random.default_rng(random_seed)
    n_pilot = n_items // 2
    pilot_item_ids = rng.choice(n_items, size=n_pilot, replace=False)
    is_pilot = np.zeros(n_items, dtype=bool)
    is_pilot[pilot_item_ids] = True
    operational_item_ids = np.where(~is_pilot)[0]
    return pilot_item_ids, operational_item_ids


def build_ensemble(random_seed: int):
    return [
        RandomForestClassifier(n_estimators=300, max_depth=8,
                                random_state=random_seed, n_jobs=-1),
        XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                      eval_metric="logloss", random_state=random_seed, n_jobs=-1),
        LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                       random_state=random_seed, verbosity=-1, n_jobs=-1),
    ]


def fit_ensemble(models, feature_df: pd.DataFrame, grades: np.ndarray):
    for model in models:
        model.fit(feature_df, grades)
    return models


def predict_ensemble(models, feature_df: pd.DataFrame) -> np.ndarray:
    predictions = []
    for model in models:
        predictions.append(model.predict_proba(feature_df)[:, 1])
    return np.mean(predictions, axis=0)


def build_features(feature_1, feature_2, theta, item_id, operational_ids: np.ndarray,
                    variant: str) -> pd.DataFrame:
    """variant='baseline': theta, feature_1, feature_2 only (what the
    pipeline uses today). variant='with_item_id': baseline + one
    one-hot column per operational item ID. `item_id` can be a single
    value (broadcast to every row) or an array matching theta's length."""
    base = pd.DataFrame({"theta": theta, "feature_1": feature_1, "feature_2": feature_2})
    if variant == "baseline":
        return base

    if variant == "with_item_id":
        n_rows = len(base)
        if np.isscalar(item_id):
            item_id_arr = np.full(n_rows, item_id)
        else:
            item_id_arr = np.asarray(item_id)

        # One column per operational item ID: 1 where that row's item_id
        # matches this column's ID, 0 otherwise.
        one_hot_columns = {}
        for op_id in operational_ids:
            column_name = f"item_{op_id}"
            one_hot_columns[column_name] = (item_id_arr == op_id).astype(int)
        one_hot = pd.DataFrame(one_hot_columns, index=base.index)

        return pd.concat([base, one_hot], axis=1)

    raise ValueError(f"Unknown variant '{variant}'")


def run_variant(variant: str, items: dict, operational_ids: np.ndarray,
                 train_theta: np.ndarray, train_responses_item_idx: np.ndarray,
                 train_responses_session_idx: np.ndarray, train_grades: np.ndarray,
                 random_seed: int) -> dict:
    print(f"\n{'#' * 70}")
    print(f"# Feature variant: {variant}")
    print(f"{'#' * 70}")

    feature_df = build_features(
        items["feature_1"][train_responses_item_idx],
        items["feature_2"][train_responses_item_idx],
        train_theta[train_responses_session_idx],
        train_responses_item_idx,
        operational_ids,
        variant,
    )
    if variant == "with_item_id":
        feature_description = "theta, feature_1, feature_2 + one-hot item id"
    else:
        feature_description = "theta, feature_1, feature_2"
    print(f"Training on {len(feature_df)} responses, {feature_df.shape[1]} feature columns "
          f"({feature_description})")

    models = build_ensemble(random_seed)
    fit_ensemble(models, feature_df, train_grades)

    # Fitted parameters for OPERATIONAL items only -- the already-seen-item
    # recalibration scenario item ID is meant to help with.
    from src.autoirt_model import _fit_irt_curve_to_predictions, DEFAULT_THETA_GRID

    fitted_a = np.zeros(len(operational_ids))
    fitted_d = np.zeros(len(operational_ids))
    for i, item_id in enumerate(operational_ids):
        query_df = build_features(
            np.full(len(DEFAULT_THETA_GRID), items["feature_1"][item_id]),
            np.full(len(DEFAULT_THETA_GRID), items["feature_2"][item_id]),
            DEFAULT_THETA_GRID,
            item_id,
            operational_ids,
            variant,
        )
        predicted_probs = predict_ensemble(models, query_df)
        fitted_a[i], fitted_d[i] = _fit_irt_curve_to_predictions(
            predicted_probs, DEFAULT_THETA_GRID, fixed_chance=CHANCE_VALUE,
        )

    true_a = items["discrimination"][operational_ids]
    true_d = items["difficulty"][operational_ids]
    difficulty_pearson, _ = pearsonr(fitted_d, true_d)
    discrimination_pearson, _ = pearsonr(fitted_a, true_a)

    print(f"Parameter recovery (fitted vs. TRUE item parameters, OPERATIONAL items):")
    print(f"  difficulty Pearson:     {difficulty_pearson:.4f}")
    print(f"  discrimination Pearson: {discrimination_pearson:.4f}")

    return {
        "variant": variant,
        "difficulty_pearson": difficulty_pearson,
        "discrimination_pearson": discrimination_pearson,
    }


def main():
    items = simulate_det_items(n_items=N_ITEMS, chance_value=CHANCE_VALUE, random_seed=RANDOM_SEED)
    pilot_ids, operational_ids = choose_pilot_and_operational_items(N_ITEMS, random_seed=RANDOM_SEED + 3)
    print(f"n_items={N_ITEMS}, operational={len(operational_ids)}, pilot(unused)={len(pilot_ids)}")
    print("(pilot items unused in this check -- item ID can't help unseen items, see docstring)")

    true_theta = simulate_test_taker_abilities(n_sessions=N_SESSIONS, random_seed=RANDOM_SEED + 1)

    # One training row per (session, item) pair: each session answers
    # ITEMS_PER_SESSION randomly chosen operational items.
    rng = np.random.default_rng(RANDOM_SEED + 2)
    session_idx_list = []
    item_idx_list = []
    for session in range(N_SESSIONS):
        sample_size = min(ITEMS_PER_SESSION, len(operational_ids))
        chosen_items = rng.choice(operational_ids, size=sample_size, replace=False)
        for item_id in chosen_items:
            session_idx_list.append(session)
            item_idx_list.append(item_id)
    session_idx = np.array(session_idx_list)
    item_idx = np.array(item_idx_list)

    probs = three_parameter_logistic(
        theta=true_theta[session_idx], discrimination=items["discrimination"][item_idx],
        chance=CHANCE_VALUE, difficulty=items["difficulty"][item_idx],
    )
    grades = rng.binomial(1, probs)
    print(f"Simulated {len(grades)} training responses from operational items only, "
          f"using TRUE theta directly (no MCEM).")

    # Responses per operational item -- context on how much data each
    # item's own recalibration has to work with.
    _, counts = np.unique(item_idx, return_counts=True)
    print(f"Responses per operational item: mean={counts.mean():.0f}, "
          f"min={counts.min()}, max={counts.max()}")

    results = []
    for variant in ["baseline", "with_item_id"]:
        results.append(run_variant(
            variant, items, operational_ids,
            true_theta, item_idx, session_idx, grades, RANDOM_SEED,
        ))

    print("\n" + "=" * 70)
    print("SUMMARY: does item ID (per-item lookup, paper's stated method) help")
    print("recalibrate items the model already has response history for?")
    print("=" * 70)
    print(f"{'Variant':<16}{'Difficulty r':>14}{'Discrimination r':>18}")
    for r in results:
        print(f"{r['variant']:<16}{r['difficulty_pearson']:>14.4f}{r['discrimination_pearson']:>18.4f}")

    baseline, with_id = results[0], results[1]
    print(f"\nDifficulty recovery improvement (with_item_id - baseline): "
          f"{with_id['difficulty_pearson'] - baseline['difficulty_pearson']:+.4f}")
    print(f"Discrimination recovery improvement (with_item_id - baseline): "
          f"{with_id['discrimination_pearson'] - baseline['discrimination_pearson']:+.4f}")
    print("\nMeaningful positive gap -> item ID is a cheap win for Warm-start (and partly")
    print("Jump-start) recalibration. Small/negative gap -> raw features already separate")
    print("operational items well enough, not where the payoff is.")


if __name__ == "__main__":
    main()