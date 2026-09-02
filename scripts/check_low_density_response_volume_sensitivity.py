"""
check_low_density_response_volume_sensitivity.py
====================================================

Follows the full-scale single run (run_yn_vocab_nlp_experiment_fullscale.py),
which showed real features performing WORSE than the synthetic baseline
in every condition -- a reversal from small scale. Candidate
explanation: full scale spreads far less data per item than small
scale does (~87 responses/item at full scale vs. ~460/item at small
scale, since item count grew ~22x while sessions only grew ~4x).

This tests that directly. check_response_volume_sensitivity.py already
swept response density, but only from ~192 to ~1,360 responses/item --
never reaching full scale's actual ~87/item. That earlier "response
volume ruled out" finding does NOT apply here; this is a different,
lower density regime that was never tested. This sweep is built to
actually bracket both real regimes (see range below).

Same isolation method as the other checks in this line: TRUE theta, no
MCEM noise, fitted vs. TRUE difficulty on a fixed 200-item holdout,
3 seeds per point. Y/N Vocab only for now -- it showed the larger
full-scale reversal, most urgent to understand first.

Run with:
    python scripts/check_low_density_response_volume_sensitivity.py
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
from src.autoirt_model import _fit_irt_curve_to_predictions, DEFAULT_THETA_GRID

N_ITEMS = 300
N_HOLDOUT = 200
OPERATIONAL_COUNT = 100  # fixed
ITEMS_PER_SESSION = 12
N_SEEDS = 3
BASE_SEED = 8000  # distinct from every seed used elsewhere in this project

CHANCE_VALUE = 0.25

# Chosen to bracket both real regimes: full scale's actual ~87
# responses/item (near point 2) and small scale's actual ~460/item
# (near point 5) -- verified before building, not assumed.
SESSION_COUNTS_TO_SWEEP = [400, 850, 1700, 2900, 4200]

FEATURE_NAMES = ["theta", "is_real", "length", "zipf_frequency",
                  "mean_bigram_log_freq", "mean_trigram_log_freq"]


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
    columns = {"theta": theta_values}
    for name in feature_names[1:]:
        columns[name] = items[name][item_indices]
    return pd.DataFrame(columns)


def run_one_seed_one_session_count(items: dict, operational_ids: np.ndarray,
                                    holdout_ids: np.ndarray, n_sessions: int,
                                    random_seed: int) -> float:
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
        chance=CHANCE_VALUE, difficulty=items["difficulty"][item_idx],
    )
    grades = rng.binomial(1, probs)

    train_df = build_feature_df(items, FEATURE_NAMES, item_idx, true_theta[session_idx])
    models = build_ensemble(random_seed)
    fit_ensemble(models, train_df, grades)

    fitted_d = np.zeros(len(holdout_ids))
    for i, item_id in enumerate(holdout_ids):
        item_index_array = np.full(len(DEFAULT_THETA_GRID), item_id)
        query_df = build_feature_df(items, FEATURE_NAMES, item_index_array, DEFAULT_THETA_GRID)
        predicted_probs = predict_ensemble(models, query_df)
        _, fitted_d[i] = _fit_irt_curve_to_predictions(
            predicted_probs, DEFAULT_THETA_GRID, fixed_chance=CHANCE_VALUE,
        )

    true_d = items["difficulty"][holdout_ids]
    difficulty_pearson, _ = pearsonr(fitted_d, true_d)
    return difficulty_pearson


def main():
    print(f"Building Y/N Vocab item bank (n_items={N_ITEMS})...")
    items = simulate_yn_vocab_items_nlp(n_items=N_ITEMS, random_seed=BASE_SEED)

    rng = np.random.default_rng(BASE_SEED + 1)
    all_item_ids = np.arange(N_ITEMS)
    rng.shuffle(all_item_ids)
    holdout_ids = all_item_ids[:N_HOLDOUT]
    operational_ids = all_item_ids[N_HOLDOUT:N_HOLDOUT + OPERATIONAL_COUNT]

    results_by_sessions = {}
    for n_sessions in SESSION_COUNTS_TO_SWEEP:
        responses_per_item = n_sessions * ITEMS_PER_SESSION / OPERATIONAL_COUNT
        seed_results = []
        for seed_index in range(N_SEEDS):
            seed = BASE_SEED + 100 + n_sessions // 10 + seed_index
            r = run_one_seed_one_session_count(items, operational_ids, holdout_ids, n_sessions, seed)
            seed_results.append(r)

        mean_r = float(np.mean(seed_results))
        std_r = float(np.std(seed_results))
        print(f"  n_sessions={n_sessions:>5} (~{responses_per_item:>4.0f} responses/item): "
              f"difficulty Pearson mean={mean_r:.4f}, std={std_r:.4f}, "
              f"per-seed={[round(r, 4) for r in seed_results]}")
        results_by_sessions[n_sessions] = {"mean": mean_r, "std": std_r,
                                            "responses_per_item": responses_per_item}

    print("\n" + "=" * 80)
    print("SUMMARY: does real-feature recovery genuinely degrade at low response density?")
    print("=" * 80)
    print(f"{'Responses/item':<18}{'Pearson mean ± std':>22}")
    for n_sessions in SESSION_COUNTS_TO_SWEEP:
        r = results_by_sessions[n_sessions]
        print(f"{r['responses_per_item']:<18.0f}{r['mean']:.4f} ± {r['std']:.4f}")

    print("\nIf recovery is clearly worse near 87 responses/item (full scale's actual density)")
    print("than near 460/item (small scale's actual density), that confirms per-item response")
    print("density as the likely explanation for the full-scale reversal. If recovery stays")
    print("roughly flat across this range too, density is not the explanation either, and the")
    print("full-scale reversal needs a different diagnosis.")


if __name__ == "__main__":
    main()
