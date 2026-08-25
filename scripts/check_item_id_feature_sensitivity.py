"""
check_item_id_feature_sensitivity.py
======================================

Re-reading the AutoIRT paper's Method section closely: "The item ID is
passed to the AutoML predictor as a feature, similar to the use of
random effects terms." Checked our FEATURE_COLUMNS in autoirt_model.py:

    FEATURE_COLUMNS = ["theta", "feature_1", "feature_2"]

No item ID. This is a real, previously-unnoticed discrepancy from the
paper's stated method, confirmed in BOTH papers (BanditCAT and AutoIRT
describe the same mechanism).

Unlike check_feature_richness_sensitivity.py, item ID can ONLY help
items the model has already seen responses for during training -- a
held-out pilot item's ID was never in the training data, so there is
nothing to look up, and one-hot encoding an unseen category contributes
nothing. That means this specifically tests the WARM-START mechanism
(recalibrating OPERATIONAL items you already have real response history
for), not Cold-start. Jump-start would see a partial, diluted version of
this effect (mostly-operational training data plus a little new-item
data); Cold-start is unaffected by design.

Same isolation strategy as the feature-richness check, for the same
reasons (removes MCEM ability-estimation noise as a confound):
  - Uses TRUE theta for training, not an EM-estimated one.
  - Compares FITTED item parameters against TRUE item parameters
    directly (Pearson correlation on difficulty and discrimination),
    evaluated on OPERATIONAL items this time (not held-out pilot items --
    there would be nothing to measure for the item-ID variant otherwise).
  - This is a single supervised-learning problem, not the full MCEM
    loop, and the metric is intentionally different from the project's
    standard item_grade_pearson_r for the same reasons documented in
    check_feature_richness_sensitivity.py.

item_id is one-hot encoded across the operational item set (the only
IDs the model ever sees in training), rather than passed as a raw
integer -- a raw integer column would let RandomForestClassifier split
on arbitrary numeric thresholds of an ID, which is not what "similar to
a random effect / per-item intercept" means and would not fairly test
the paper's actual mechanism. One-hot is unambiguous across all three
ensemble members (RF, XGBoost, LightGBM).

Y/N Vocab only, same as the feature-richness check -- if this looks
promising, ViC is the natural next check, same as there.

Run with:
    python scripts/check_item_id_feature_sensitivity.py

Takes a few minutes, similar to check_feature_richness_sensitivity.py
(same model architecture, no AutoML search, no MCEM loop). One-hot
encoding 75 operational items adds 75 columns -- noticeably more than
the feature-richness check's 8, but still small for tree ensembles.
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
    """Same 50/50 split used throughout this project's DET-phase checks --
    reimplemented locally rather than imported, same reasoning as
    check_feature_richness_sensitivity.py (avoid pulling in
    run_det_experiment.py's src.evaluate dependency for one small helper)."""
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
    predictions = [model.predict_proba(feature_df)[:, 1] for model in models]
    return np.mean(predictions, axis=0)


def build_features(feature_1, feature_2, theta, item_id, operational_ids: np.ndarray,
                    variant: str) -> pd.DataFrame:
    """variant='baseline' -> exactly what the pipeline gives the model today
    (theta, feature_1, feature_2). variant='with_item_id' -> baseline plus
    one one-hot column per operational item ID, matching the paper's
    described mechanism. `item_id` may be a scalar (broadcast) or an
    array the same length as theta/feature_1/feature_2."""
    base = pd.DataFrame({"theta": theta, "feature_1": feature_1, "feature_2": feature_2})
    if variant == "baseline":
        return base

    if variant == "with_item_id":
        n_rows = len(base)
        item_id_arr = np.full(n_rows, item_id) if np.isscalar(item_id) else np.asarray(item_id)
        one_hot = pd.DataFrame(
            {f"item_{op_id}": (item_id_arr == op_id).astype(int) for op_id in operational_ids},
            index=base.index,
        )
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
    print(f"Training on {len(feature_df)} responses, {feature_df.shape[1]} feature columns"
          f"{' (theta, feature_1, feature_2 + one-hot item id)' if variant == 'with_item_id' else ' (theta, feature_1, feature_2)'}")

    models = build_ensemble(random_seed)
    fit_ensemble(models, feature_df, train_grades)

    # Recover parameters for OPERATIONAL items only -- this is the
    # already-seen-item recalibration scenario item ID is meant to help
    # with, unlike the feature-richness check's held-out pilot items.
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

    rng = np.random.default_rng(RANDOM_SEED + 2)
    session_idx_list, item_idx_list = [], []
    for session in range(N_SESSIONS):
        chosen = rng.choice(operational_ids, size=min(ITEMS_PER_SESSION, len(operational_ids)),
                             replace=False)
        session_idx_list.extend([session] * len(chosen))
        item_idx_list.extend(chosen.tolist())
    session_idx = np.array(session_idx_list)
    item_idx = np.array(item_idx_list)

    probs = three_parameter_logistic(
        theta=true_theta[session_idx], discrimination=items["discrimination"][item_idx],
        chance=CHANCE_VALUE, difficulty=items["difficulty"][item_idx],
    )
    grades = rng.binomial(1, probs)
    print(f"Simulated {len(grades)} training responses from operational items only, "
          f"using TRUE theta directly (no MCEM).")

    # Responses per operational item, for context on how much each item's
    # own recalibration has to work with.
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
    print("\nA meaningful positive gap here means adding item ID as a feature is a cheap,")
    print("easy win for Warm-start (and partially Jump-start) recalibration specifically --")
    print("no NLP work required, just a code change to FEATURE_COLUMNS and the M-step's")
    print("training/query feature construction. A small or negative gap means the raw")
    print("features already let the model separate operational items well enough on their")
    print("own, and this specific paper-vs-implementation gap isn't where the payoff is.")


if __name__ == "__main__":
    main()
