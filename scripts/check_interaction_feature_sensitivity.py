"""
check_interaction_feature_sensitivity.py
============================================

Third test in this line of investigation. Response volume is ruled
out (check_response_volume_sensitivity.py: flat curves, gap constant
from 14k to 102k training rows). This tests a different hypothesis:
not "needs more data," but "the tree ensemble can't easily DISCOVER
Y/N Vocab's implicit branching rule (frequency effect only applies to
real words, wordlikeness effect only applies to fake words) from the
5 raw features alone, regardless of how much data it gets."

If that's right, handing the model an EXPLICIT interaction feature
that spells out the branching directly (frequency * is_real,
wordlikeness * is_fake) should close much of the gap toward ViC's
recovery level, since the model no longer has to discover the
conditional structure itself, just use a feature that already encodes
it. If it doesn't help, this hypothesis is wrong too and the
investigation should stop here rather than keep guessing.

Y/N Vocab only -- ViC has no branching structure for this test to
apply to. Single representative training-data point (~36,000 rows,
close to the real Cold-start volume), not a sweep -- the response-
volume check already showed the gap is constant across volume, so one
point is enough to test whether an explicit interaction feature closes
it.

Same isolation method as the other checks in this line: TRUE theta, no
MCEM noise, fitted vs. TRUE difficulty on a fixed 200-item holdout,
3 seeds.

Run with:
    python scripts/check_interaction_feature_sensitivity.py
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
OPERATIONAL_COUNT = 75
ITEMS_PER_SESSION = 12
N_SESSIONS = 3000  # ~36,000 training rows, close to the real Cold-start volume (~34,478)
N_SEEDS = 3
BASE_SEED = 7000  # distinct from every seed used elsewhere in this project

CHANCE_VALUE = 0.25

BASELINE_FEATURE_NAMES = ["theta", "is_real", "length", "zipf_frequency",
                           "mean_bigram_log_freq", "mean_trigram_log_freq"]
# Same as baseline, plus two explicit interaction columns that directly
# encode the branching rule the true difficulty formula actually uses:
# frequency only matters for real words, wordlikeness only for fake words.
INTERACTION_FEATURE_NAMES = BASELINE_FEATURE_NAMES + ["freq_x_is_real", "wordlikeness_x_is_fake"]


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


def add_interaction_columns(items: dict) -> dict:
    """Returns a new items dict with two extra columns: freq_x_is_real
    (zipf_frequency where is_real==1, else 0) and wordlikeness_x_is_fake
    (mean wordlikeness where is_real==0, else 0). These directly encode
    the branching rule the true difficulty formula uses, rather than
    leaving the model to discover it from is_real + the raw features
    separately."""
    items = dict(items)  # shallow copy, don't mutate the caller's dict
    wordlikeness = (items["mean_bigram_log_freq"] + items["mean_trigram_log_freq"]) / 2
    items["freq_x_is_real"] = items["zipf_frequency"] * items["is_real"]
    items["wordlikeness_x_is_fake"] = wordlikeness * (1 - items["is_real"])
    return items


def build_feature_df(items: dict, feature_names: list, item_indices: np.ndarray,
                      theta_values: np.ndarray) -> pd.DataFrame:
    columns = {"theta": theta_values}
    for name in feature_names[1:]:
        columns[name] = items[name][item_indices]
    return pd.DataFrame(columns)


def run_one_variant_one_seed(items: dict, feature_names: list, operational_ids: np.ndarray,
                              holdout_ids: np.ndarray, random_seed: int) -> float:
    rng = np.random.default_rng(random_seed)
    true_theta = simulate_test_taker_abilities(n_sessions=N_SESSIONS, random_seed=random_seed + 1)

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

    train_df = build_feature_df(items, feature_names, item_idx, true_theta[session_idx])
    models = build_ensemble(random_seed)
    fit_ensemble(models, train_df, grades)

    fitted_d = np.zeros(len(holdout_ids))
    for i, item_id in enumerate(holdout_ids):
        item_index_array = np.full(len(DEFAULT_THETA_GRID), item_id)
        query_df = build_feature_df(items, feature_names, item_index_array, DEFAULT_THETA_GRID)
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
    items = add_interaction_columns(items)

    rng = np.random.default_rng(BASE_SEED + 1)
    all_item_ids = np.arange(N_ITEMS)
    rng.shuffle(all_item_ids)
    holdout_ids = all_item_ids[:N_HOLDOUT]
    operational_ids = all_item_ids[N_HOLDOUT:N_HOLDOUT + OPERATIONAL_COUNT]

    results = {}
    for variant_name, feature_names in [("baseline", BASELINE_FEATURE_NAMES),
                                         ("with_interaction", INTERACTION_FEATURE_NAMES)]:
        print(f"\n{'#' * 70}")
        print(f"# Variant: {variant_name}  ({len(feature_names)} feature columns)")
        print(f"{'#' * 70}")

        seed_results = []
        for seed_index in range(N_SEEDS):
            seed = BASE_SEED + 100 + seed_index
            r = run_one_variant_one_seed(items, feature_names, operational_ids, holdout_ids, seed)
            seed_results.append(r)
            print(f"  seed {seed_index + 1}/{N_SEEDS}: difficulty Pearson = {r:.4f}")

        mean_r = float(np.mean(seed_results))
        std_r = float(np.std(seed_results))
        print(f"  mean={mean_r:.4f}, std={std_r:.4f}")
        results[variant_name] = {"mean": mean_r, "std": std_r, "seeds": seed_results}

    print("\n" + "=" * 70)
    print("SUMMARY: does an explicit interaction feature close Y/N Vocab's gap?")
    print("=" * 70)
    baseline_mean = results["baseline"]["mean"]
    interaction_mean = results["with_interaction"]["mean"]
    improvement = interaction_mean - baseline_mean
    print(f"baseline:          {baseline_mean:.4f} ± {results['baseline']['std']:.4f}")
    print(f"with_interaction:  {interaction_mean:.4f} ± {results['with_interaction']['std']:.4f}")
    print(f"improvement:       {improvement:+.4f}")
    print("\nFor reference, ViC's recovery at comparable data volume (from")
    print("check_response_volume_sensitivity.py, ~36,000 rows) was ~0.218.")
    print("\nA large positive improvement, closing most of the gap to ViC's ~0.218, confirms the")
    print("model-discoverability explanation. A small or no improvement means this hypothesis")
    print("is wrong too, and the asymmetry should be reported as unexplained rather than guessed at.")


if __name__ == "__main__":
    main()
