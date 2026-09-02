"""
check_response_volume_sensitivity.py
=======================================

Second redesign. The item-count sweep (check_operational_item_count_
sensitivity.py, now retired) tested the wrong axis: in the real
pipeline, Cold/Jump/Warm-start all draw from the SAME fixed operational
item set (~75 items for Y/N Vocab) -- what actually differs between
them is how much RESPONSE VOLUME has accumulated for those items by
calibration time, not how many distinct items exist. Cold-start's
evaluated items have zero responses; Warm-start's operational items
have the most. This sweeps that instead: operational item count held
FIXED, total training response volume varied (via session count),
directly mirroring what actually changes between the real conditions.

Same isolation method as before: TRUE theta (no MCEM noise), fitted
vs. TRUE difficulty on a FIXED holdout set. Same fix as
check_cold_start_noise.py for stability: N_HOLDOUT=200, N_SEEDS=3 per
sweep point.

If Y/N Vocab's branching difficulty rule needs more total data than
ViC's additive one, Y/N Vocab's recovery should be poor at low session
counts and catch up only at high ones; ViC's should be comparatively
flat-and-good throughout.

Run with:
    python scripts/check_response_volume_sensitivity.py
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

N_ITEMS = 300
N_HOLDOUT = 200
OPERATIONAL_COUNT = 75  # fixed -- matches the real Y/N Vocab operational
                         # item count convention used throughout this project
ITEMS_PER_SESSION = 12
N_SEEDS = 3
BASE_SEED = 6000  # distinct from every seed used elsewhere in this project

# Total training rows at each point is roughly SESSION_COUNTS[i] * ITEMS_PER_SESSION.
# Chosen to bracket the real pipeline's actual regimes, not just round numbers:
# real Y/N Vocab Cold-start trains on ~34,478 rows, Warm-start on ~50,724-59,436 rows
# (both verified from earlier runs this project). This sweep spans below Cold-start,
# through both real regimes, to above Warm-start.
SESSION_COUNTS_TO_SWEEP = [1200, 3000, 4200, 6000, 8500]

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


def run_one_seed_one_session_count(items: dict, chance_value: float, feature_names: list,
                                    operational_ids: np.ndarray, holdout_ids: np.ndarray,
                                    n_sessions: int, random_seed: int) -> float:
    """Trains on n_sessions worth of responses from the FIXED operational
    set, evaluates difficulty recovery on the FIXED holdout set. Returns
    the difficulty Pearson correlation for this one seed."""
    rng = np.random.default_rng(random_seed)
    true_theta = simulate_test_taker_abilities(n_sessions=n_sessions, random_seed=random_seed + 1)

    session_idx_list = []
    item_idx_list = []
    for session in range(n_sessions):
        sample_size = min(ITEMS_PER_SESSION, len(operational_ids))
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
    operational_ids = all_item_ids[N_HOLDOUT:N_HOLDOUT + OPERATIONAL_COUNT]  # FIXED set,
                                                                              # same at every sweep point

    results_by_sessions = {}
    for n_sessions in SESSION_COUNTS_TO_SWEEP:
        total_rows = n_sessions * min(ITEMS_PER_SESSION, OPERATIONAL_COUNT)
        seed_results = []
        for seed_index in range(N_SEEDS):
            seed = BASE_SEED + 100 + n_sessions // 100 + seed_index
            r = run_one_seed_one_session_count(items, chance_value, feature_names,
                                                operational_ids, holdout_ids, n_sessions, seed)
            seed_results.append(r)

        mean_r = float(np.mean(seed_results))
        std_r = float(np.std(seed_results))
        print(f"  n_sessions={n_sessions:>6} (~{total_rows:>6} training rows): difficulty Pearson "
              f"mean={mean_r:.4f}, std={std_r:.4f}, per-seed={[round(r, 4) for r in seed_results]}")
        results_by_sessions[n_sessions] = {"mean": mean_r, "std": std_r, "seeds": seed_results}

    return results_by_sessions


def main():
    yn_results = run_one_item_type("Y/N Vocab", simulate_yn_vocab_items_nlp, 0.25, YN_FEATURE_NAMES)
    vic_results = run_one_item_type("ViC", simulate_vic_items_nlp, 0.0, VIC_FEATURE_NAMES)

    print("\n" + "=" * 80)
    print("SUMMARY: does recovery quality need more response volume for one")
    print("item type than the other?")
    print("=" * 80)
    print(f"{'Sessions':<12}{'Y/N Vocab (mean±std)':>24}{'ViC (mean±std)':>22}")
    for n_sessions in SESSION_COUNTS_TO_SWEEP:
        yn = yn_results[n_sessions]
        vic = vic_results[n_sessions]
        yn_str = f"{yn['mean']:.4f} ± {yn['std']:.4f}"
        vic_str = f"{vic['mean']:.4f} ± {vic['std']:.4f}"
        print(f"{n_sessions:<12}{yn_str:>24}{vic_str:>22}")

    print("\nIf Y/N Vocab starts low and rises to meet or pass ViC as session count grows,")
    print("that supports the branching-vs-additive explanation (Y/N Vocab's conditional")
    print("rule needs more total data). If the gap stays roughly constant across the sweep,")
    print("or both curves move together, the explanation is wrong and should be dropped.")


if __name__ == "__main__":
    main()
