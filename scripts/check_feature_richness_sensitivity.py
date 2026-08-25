"""
check_feature_richness_sensitivity.py
=======================================

Backend is ruled out (run_det_backend_comparison.py). EM step count is
ruled out for both Cold-start and Jump-start, at both small and full
scale (check_cold_start_extended_convergence.py,
check_jump_start_extended_convergence.py, plus the full-scale n_steps_run
means already in det_replication_summary_fullscale.csv). What's left
pointing at the residual gap is feature quality: the simulation gives the
model two raw uniform numbers (feature_1, feature_2) and asks it to
recover a difficulty function built from sin(feature_1 - feature_2) /
|feature_1 - feature_2| and a discrimination function built from
cos(0.1 * (feature_1 - feature_2)^2) -- a genuinely hard function for a
tree-based model to reconstruct from 2 raw inputs with a small item bank,
versus the paper's real features (BERT embeddings, COCA frequency
statistics, CEFR wordlist membership) which are presumably far more
directly informative about actual item difficulty.

This does NOT attempt to simulate real NLP features (that's next week's
work). It asks a narrower, cheaper question first: holding the TRUE item
parameters completely fixed, does giving the model an easier-to-use
representation of the SAME information (explicit z = feature_1 -
feature_2, z^2, sin(z), cos(z), |z| columns, rather than making it infer
that subtraction and those nonlinear transforms itself from 2 raw
numbers) meaningfully improve cold-start item parameter recovery? If
richer feature *representation* alone (with zero new information added)
closes a meaningful chunk of the gap, that is a strong signal that real
NLP features -- which add both richer representation AND more actual
information than 2 synthetic numbers -- are likely to help a lot,
justifying the time investment in that phase. If it does not help much,
that tempers expectations and suggests structural limits (item bank
size, response volume) matter more than feature engineering.

Deliberately simplified relative to the full pipeline, to isolate ONE
variable (feature representation) instead of also carrying over MCEM's
ability-estimation noise:
  - Uses the TRUE simulated ability (theta) for training, not an
    EM-estimated one. This is a single supervised-learning problem
    (theta, item features) -> grade, not the iterative MCEM loop, so
    ability-estimation noise (already shown to be substantial for
    Cold-start) cannot muddy the comparison.
  - Compares FITTED item parameters against the TRUE (ground-truth)
    item parameters directly (Pearson correlation on difficulty and on
    discrimination separately), rather than the project's standard
    item_grade_pearson_r (which is computed from noisy held-out
    responses via evaluate_calibration). This is a cleaner, more
    direct measure of "how well did this feature representation let
    the model recover the true item parameters" -- but it is a
    DIFFERENT metric than every other script in this project reports,
    is not directly comparable to those numbers, and should not be
    quoted as an official gap-vs-paper figure.
  - A secondary, more familiar-flavored number is also computed
    (mean-grade-vs-mean-predicted-grade correlation on fresh held-out
    test sessions, using TRUE theta rather than the pipeline's
    posterior-mean estimate) purely as an intuition check, clearly
    labeled as an approximation.

Y/N Vocab only, to keep this fast -- if the result looks promising,
the same idea extends to ViC.

Run with:
    python scripts/check_feature_richness_sensitivity.py

Takes a few minutes (two ensemble fits, no AutoML search, no MCEM loop).
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

import config
from src.simulate import three_parameter_logistic, simulate_test_taker_abilities
from src.simulate_det import simulate_det_items
from src.autoirt_model import _fit_irt_curve_to_predictions, DEFAULT_THETA_GRID

N_ITEMS = 150            # matches the small-scale Y/N Vocab convention used throughout
N_SESSIONS = 6000
ITEMS_PER_SESSION = 18
CHANCE_VALUE = 0.25       # Y/N Vocab's chance parameter
RANDOM_SEED = 42


def choose_pilot_and_operational_items(n_items: int, random_seed: int):
    """Same 50/50 split as run_det_experiment.py's function of the same
    name, reimplemented here directly rather than imported, so this
    script doesn't need to pull in the rest of run_det_experiment.py
    (and its src.evaluate dependency) just for this one small helper."""
    rng = np.random.default_rng(random_seed)
    n_pilot = n_items // 2
    pilot_item_ids = rng.choice(n_items, size=n_pilot, replace=False)
    is_pilot = np.zeros(n_items, dtype=bool)
    is_pilot[pilot_item_ids] = True
    operational_item_ids = np.where(~is_pilot)[0]
    return pilot_item_ids, operational_item_ids


def build_ensemble(random_seed: int):
    """Same architecture as autoirt_model.GradeEnsembleModel (RF + XGB +
    LGBM, averaged), reimplemented locally rather than reusing that class
    directly, since it hardcodes a fixed FEATURE_COLUMNS list and this
    script needs to swap feature sets without touching shared state."""
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


def build_features(feature_1, feature_2, theta, variant: str) -> pd.DataFrame:
    """variant='baseline' -> exactly what the pipeline gives the model today.
    variant='rich' -> same underlying information (feature_1, feature_2),
    plus the engineered transforms the TRUE DIFFICULTY function is built
    from (sin(z), |z|), so the model doesn't have to reconstruct
    z = feature_1 - feature_2 and that nonlinear transform from scratch.
    variant='richer' -> rich, plus cos(0.1 * z^2) -- the exact composite
    term the TRUE DISCRIMINATION function (0.5*cos(0.1*z^2)+1.0) is built
    from. 'rich' deliberately does NOT include this: it gives z_squared
    and cos(z) as separate, uncombined columns, which is a different and
    still-hard problem from being given cos(z^2) directly -- discovered
    as the likely explanation for 'rich' recovering difficulty (0.88-0.98)
    much better than discrimination (0.53-0.56) across every seed tested.
    'richer' tests that explanation directly."""
    z = feature_1 - feature_2
    if variant == "baseline":
        return pd.DataFrame({
            "theta": theta, "feature_1": feature_1, "feature_2": feature_2,
        })
    elif variant == "rich":
        return pd.DataFrame({
            "theta": theta, "feature_1": feature_1, "feature_2": feature_2,
            "z": z, "z_squared": z ** 2, "sin_z": np.sin(z), "cos_z": np.cos(z),
            "abs_z": np.abs(z),
        })
    elif variant == "richer":
        return pd.DataFrame({
            "theta": theta, "feature_1": feature_1, "feature_2": feature_2,
            "z": z, "z_squared": z ** 2, "sin_z": np.sin(z), "cos_z": np.cos(z),
            "abs_z": np.abs(z), "cos_z_sq_scaled": np.cos(0.1 * z ** 2),
        })
    raise ValueError(f"Unknown variant '{variant}'")


def run_variant(variant: str, items: dict, operational_ids: np.ndarray, holdout_ids: np.ndarray,
                 train_theta: np.ndarray, train_responses_item_idx: np.ndarray,
                 train_responses_session_idx: np.ndarray, train_grades: np.ndarray,
                 random_seed: int, chance_value: float = CHANCE_VALUE) -> dict:
    print(f"\n{'#' * 70}")
    print(f"# Feature variant: {variant}")
    print(f"{'#' * 70}")

    feature_df = build_features(
        items["feature_1"][train_responses_item_idx],
        items["feature_2"][train_responses_item_idx],
        train_theta[train_responses_session_idx],
        variant,
    )
    print(f"Training on {len(feature_df)} responses, {feature_df.shape[1]} feature columns: "
          f"{list(feature_df.columns)}")

    models = build_ensemble(random_seed)
    fit_ensemble(models, feature_df, train_grades)

    # Cold-start: fit an IRT curve for every held-out item from its
    # features alone, across the theta grid, exactly like
    # calibrate_items_given_abilities does for pilot items.
    fitted_a = np.zeros(len(holdout_ids))
    fitted_d = np.zeros(len(holdout_ids))
    for i, item_id in enumerate(holdout_ids):
        query_df = build_features(
            np.full(len(DEFAULT_THETA_GRID), items["feature_1"][item_id]),
            np.full(len(DEFAULT_THETA_GRID), items["feature_2"][item_id]),
            DEFAULT_THETA_GRID,
            variant,
        )
        predicted_probs = predict_ensemble(models, query_df)
        fitted_a[i], fitted_d[i] = _fit_irt_curve_to_predictions(
            predicted_probs, DEFAULT_THETA_GRID, fixed_chance=chance_value,
        )

    true_a = items["discrimination"][holdout_ids]
    true_d = items["difficulty"][holdout_ids]
    difficulty_pearson, _ = pearsonr(fitted_d, true_d)
    discrimination_pearson, _ = pearsonr(fitted_a, true_a)

    # Secondary, more familiar-flavored check: fresh held-out test
    # sessions at TRUE theta, comparing mean true grade vs. mean grade
    # predicted from the FITTED item parameters, per item. Seeded off
    # `random_seed` (the same one used for this replication's training
    # data), not a module-level constant -- otherwise every replication
    # would reuse the identical held-out test sessions here regardless
    # of which seed generated the training data, silently defeating
    # the point of replicating this metric at all.
    rng = np.random.default_rng(random_seed + 999)
    test_theta = simulate_test_taker_abilities(n_sessions=2000, random_seed=random_seed + 1000)
    reps_per_item = max(1, 2000 // len(holdout_ids))
    session_theta_per_item = rng.choice(test_theta, size=(len(holdout_ids), reps_per_item))
    true_probs = three_parameter_logistic(
        theta=session_theta_per_item, discrimination=true_a[:, None],
        chance=chance_value, difficulty=true_d[:, None],
    )
    true_grades = rng.binomial(1, true_probs)
    predicted_probs_at_test = three_parameter_logistic(
        theta=session_theta_per_item, discrimination=fitted_a[:, None],
        chance=chance_value, difficulty=fitted_d[:, None],
    )
    mean_true_grade = true_grades.mean(axis=1)
    mean_predicted_grade = predicted_probs_at_test.mean(axis=1)
    approx_item_grade_pearson, _ = pearsonr(mean_true_grade, mean_predicted_grade)

    print(f"Parameter recovery (fitted vs. TRUE item parameters, held-out items only):")
    print(f"  difficulty Pearson:     {difficulty_pearson:.4f}")
    print(f"  discrimination Pearson: {discrimination_pearson:.4f}")
    print(f"Approx. item-grade Pearson (true theta, not posterior-mean -- see docstring): "
          f"{approx_item_grade_pearson:.4f}")

    return {
        "variant": variant,
        "difficulty_pearson": difficulty_pearson,
        "discrimination_pearson": discrimination_pearson,
        "approx_item_grade_pearson": approx_item_grade_pearson,
    }


def run_one_seed(random_seed: int, n_items: int = N_ITEMS, n_sessions: int = N_SESSIONS,
                  items_per_session: int = ITEMS_PER_SESSION, chance_value: float = CHANCE_VALUE,
                  variants: tuple = ("baseline", "rich"), verbose: bool = True) -> list:
    """Runs each variant in `variants` for one random seed and returns the
    result dicts in the same order. Factored out of main() so a
    replication script can call this in a loop with different seeds
    without copying the simulation/training logic. chance_value defaults
    to Y/N Vocab's 0.25 -- pass 0.0 for ViC. `variants` defaults to
    exactly what it always has been (baseline, rich), so existing callers
    (check_feature_richness_sensitivity_replicated.py, which unpacks the
    result as `baseline_result, rich_result = run_one_seed(...)`) are
    unaffected -- pass a longer tuple (e.g. adding "richer") explicitly
    to test more variants."""
    _print = print if verbose else (lambda *a, **k: None)

    items = simulate_det_items(n_items=n_items, chance_value=chance_value, random_seed=random_seed)
    pilot_ids, operational_ids = choose_pilot_and_operational_items(n_items, random_seed=random_seed + 3)
    _print(f"n_items={n_items}, operational={len(operational_ids)}, holdout(pilot)={len(pilot_ids)}")

    true_theta = simulate_test_taker_abilities(n_sessions=n_sessions, random_seed=random_seed + 1)

    rng = np.random.default_rng(random_seed + 2)
    session_idx_list, item_idx_list = [], []
    for session in range(n_sessions):
        chosen = rng.choice(operational_ids, size=min(items_per_session, len(operational_ids)),
                             replace=False)
        session_idx_list.extend([session] * len(chosen))
        item_idx_list.extend(chosen.tolist())
    session_idx = np.array(session_idx_list)
    item_idx = np.array(item_idx_list)

    probs = three_parameter_logistic(
        theta=true_theta[session_idx], discrimination=items["discrimination"][item_idx],
        chance=chance_value, difficulty=items["difficulty"][item_idx],
    )
    grades = rng.binomial(1, probs)
    _print(f"Simulated {len(grades)} training responses from operational items only, "
           f"using TRUE theta directly (no MCEM).")

    results = []
    for variant in variants:
        if verbose:
            results.append(run_variant(
                variant, items, operational_ids, pilot_ids,
                true_theta, item_idx, session_idx, grades, random_seed, chance_value,
            ))
        else:
            # Same call, just skip the run_variant()-internal prints for
            # a quieter multi-seed loop -- run_variant always prints
            # its own progress, so this temporarily silences stdout
            # around just that call rather than threading a verbose
            # flag through every function.
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                results.append(run_variant(
                    variant, items, operational_ids, pilot_ids,
                    true_theta, item_idx, session_idx, grades, random_seed, chance_value,
                ))

    return results


def print_summary(results: list):
    print("\n" + "=" * 70)
    print("SUMMARY: does feature REPRESENTATION alone (same info, easier form) help?")
    print("=" * 70)
    print(f"{'Variant':<12}{'Difficulty r':>14}{'Discrimination r':>18}{'Approx item-grade r':>22}")
    for r in results:
        print(f"{r['variant']:<12}{r['difficulty_pearson']:>14.4f}"
              f"{r['discrimination_pearson']:>18.4f}{r['approx_item_grade_pearson']:>22.4f}")

    baseline = results[0]
    for r in results[1:]:
        print(f"\n{r['variant']} vs. baseline:")
        print(f"  Difficulty recovery improvement:     "
              f"{r['difficulty_pearson'] - baseline['difficulty_pearson']:+.4f}")
        print(f"  Discrimination recovery improvement: "
              f"{r['discrimination_pearson'] - baseline['discrimination_pearson']:+.4f}")
    print("\nA large positive gap here means feature representation is a real lever and the")
    print("NLP feature phase is likely to pay off substantially. A small or negative gap means")
    print("the model was already extracting most of what's extractable from 2 raw features, and")
    print("the gap to the paper is more likely structural (item bank size, response volume) --")
    print("worth tempering expectations for how much the NLP phase alone will close it.")


def main():
    results = run_one_seed(RANDOM_SEED)
    print_summary(results)


if __name__ == "__main__":
    main()
