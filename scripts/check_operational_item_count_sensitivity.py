"""
check_operational_item_count_sensitivity.py
==============================================

Redesign of the first version. That version (N_HOLDOUT=50, single seed)
came back too noisy to read: both curves bounced non-monotonically, and
one point went slightly negative, which shouldn't happen from real
signal alone. Same failure mode check_cold_start_noise.py hit and
fixed earlier in this project -- a small holdout set gives an unstable
single-run correlation estimate. Applying the same fix here: bigger
fixed holdout (200, up from 50) plus a modest 3-seed replication per
sweep point, not a blind 5x replication of the noisy design.

Tests whether Y/N Vocab's branching difficulty formula (real vs. fake
words follow different rules) needs more operational items to learn
than ViC's additive one (one smooth rule for every item) -- see the
Week 6 report for the full reasoning. Ruled out separately: raw
single-feature signal strength against true difficulty, already
measured as nearly identical (~-0.37) for both types during generator
validation.

Same isolation method as check_feature_richness_sensitivity.py: TRUE
theta for training (no MCEM noise), fitted vs. TRUE difficulty on a
FIXED holdout set (same items at every operational-count step).

Run with:
    python scripts/check_operational_item_count_sensitivity.py
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
from src.simulate_yn_vocab_nlp import simulate_yn_vocab_items_nlp
from src.simulate_vic_nlp import simulate_vic_items_nlp
from src.autoirt_model import _fit_irt_curve_to_predictions, DEFAULT_THETA_GRID

N_ITEMS = 300      # larger bank than the first version (150) -- needed to
                    # support a 200-item holdout plus a large enough pool
N_HOLDOUT = 200     # up from 50, matching check_cold_start_noise.py's fix
                    # for the same instability problem
N_SESSIONS = 6000
ITEMS_PER_SESSION = 12
N_SEEDS = 3         # modest replication, matching check_cold_start_noise.py's
                    # N_SEEDS_PER_SIZE -- not a full 5x rerun of the noisy design
BASE_SEED = 5000    # distinct from every seed used elsewhere in this project

OPERATIONAL_COUNTS_TO_SWEEP = [10, 20, 40, 75, 100]

YN_FEATURE_NAMES = ["theta", "is_real", "length", "zipf_frequency",
                     "mean_bigram_log_freq", "mean_trigram_log_freq"]
VIC_FEATURE_NAMES = ["theta", "num_missing_chars", "proportion_vowels_missing",
                      "target_zipf_frequency", "sentence_mean_log_frequency",
                      "position_normalized", "completion_predictability"]


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


def build_feature_df(items: dict, feature_names: list, item_indices: np.ndarray,
                      theta_values: np.ndarray) -> pd.DataFrame:
    """feature_names[0] must be "theta". item_indices selects which
    item's raw features go in each row; theta_values is that row's
    ability value. Both arrays must be the same length."""
    columns = {"theta": theta_values}
    for name in feature_names[1:]:
        columns[name] = items[name][item_indices]
    return pd.DataFrame(columns)


def run_one_seed_one_count(items: dict, chance_value: float, feature_names: list,
                            pool_ids: np.ndarray, holdout_ids: np.ndarray,
                            operational_count: int, true_theta: np.ndarray,
                            random_seed: int) -> float:
    """Trains on operational_count items drawn from pool_ids, evaluates
    difficulty recovery on the FIXED holdout_ids. Returns the difficulty
    Pearson correlation for this one seed."""
    rng = np.random.default_rng(random_seed)
    operational_ids = rng.choice(pool_ids, size=operational_count, replace=False)

    session_idx_list = []
    item_idx_list = []
    for session in range(N_SESSIONS):
        sample_size = min(ITEMS_PER_SESSION, operational_count)
        chosen_items = rng.choice(operational_ids, size=sample_size, replace=False)
        for item_id in chosen_items:
            session_idx_list.append(session)
            item_idx_list.append(item_id)
    session_idx = np.array(session_idx_list)
    item_idx = np.array(item_idx_list)

    probs = three_parameter_logistic(
        theta=true_theta[session_idx], discrimination=items["discrimination"][item_idx],
        chance=chance_value, difficulty=items["difficulty"][item_idx],
    )
    grades = rng.binomial(1, probs)

    train_df = build_feature_df(items, feature_names, item_idx, true_theta[session_idx])
    models = build_ensemble(random_seed)
    fit_ensemble(models, train_df, grades)

    fitted_d = np.zeros(len(holdout_ids))
    for i, item_id in enumerate(holdout_ids):
        item_index_array = np.full(len(DEFAULT_THETA_GRID), item_id)
        query_df = build_feature_df(items, feature_names, item_index_array, DEFAULT_THETA_GRID)
        predicted_probs = predict_ensemble(models, query_df)
        _, fitted_d[i] = _fit_irt_curve_to_predictions(
            predicted_probs, DEFAULT_THETA_GRID, fixed_chance=chance_value,
        )

    true_d = items["difficulty"][holdout_ids]
    difficulty_pearson, _ = pearsonr(fitted_d, true_d)
    return difficulty_pearson


def run_one_item_type(label: str, generate_items_fn, chance_value: float,
                       feature_names: list) -> dict:
    print(f"\n{'=' * 70}")
    print(f"ITEM TYPE: {label}")
    print(f"{'=' * 70}")

    items = generate_items_fn(n_items=N_ITEMS, random_seed=BASE_SEED)

    rng = np.random.default_rng(BASE_SEED + 1)
    all_item_ids = np.arange(N_ITEMS)
    rng.shuffle(all_item_ids)
    holdout_ids = all_item_ids[:N_HOLDOUT]
    pool_ids = all_item_ids[N_HOLDOUT:]  # operational subsets drawn from here

    true_theta = simulate_test_taker_abilities(n_sessions=N_SESSIONS, random_seed=BASE_SEED + 2)

    results_by_count = {}
    for operational_count in OPERATIONAL_COUNTS_TO_SWEEP:
        seed_results = []
        for seed_index in range(N_SEEDS):
            seed = BASE_SEED + 100 + operational_count * 10 + seed_index
            r = run_one_seed_one_count(items, chance_value, feature_names, pool_ids,
                                        holdout_ids, operational_count, true_theta, seed)
            seed_results.append(r)

        mean_r = float(np.mean(seed_results))
        std_r = float(np.std(seed_results))
        print(f"  operational_count={operational_count:>4}: difficulty Pearson "
              f"mean={mean_r:.4f}, std={std_r:.4f}, per-seed={[round(r, 4) for r in seed_results]}")
        results_by_count[operational_count] = {"mean": mean_r, "std": std_r, "seeds": seed_results}

    return results_by_count


def main():
    yn_results = run_one_item_type("Y/N Vocab", simulate_yn_vocab_items_nlp, 0.25, YN_FEATURE_NAMES)
    vic_results = run_one_item_type("ViC", simulate_vic_items_nlp, 0.0, VIC_FEATURE_NAMES)

    print("\n" + "=" * 80)
    print("SUMMARY: does recovery quality need more operational items for one")
    print("item type than the other?")
    print("=" * 80)
    print(f"{'Operational count':<20}{'Y/N Vocab (mean±std)':>24}{'ViC (mean±std)':>22}")
    for count in OPERATIONAL_COUNTS_TO_SWEEP:
        yn = yn_results[count]
        vic = vic_results[count]
        yn_str = f"{yn['mean']:.4f} ± {yn['std']:.4f}"
        vic_str = f"{vic['mean']:.4f} ± {vic['std']:.4f}"
        print(f"{count:<20}{yn_str:>24}{vic_str:>22}")

    print("\nIf Y/N Vocab's mean rises much more steeply with operational count than ViC's")
    print("(low at small counts, catching up only at large counts), that supports the")
    print("branching-vs-additive explanation. If both curves rise similarly, or Y/N Vocab's")
    print("std stays too large to read a trend at all, the explanation needs rethinking.")


if __name__ == "__main__":
    main()
