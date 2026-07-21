"""
autoirt_model.py
================

This module implements the AutoIRT calibration algorithm from:

    Sharpnack et al. (2024), "AutoIRT: Calibrating Item Response Theory
    Models with Automated Machine Learning"
    (see their Algorithm 1)

The big idea, explained simply
--------------------------------
Traditional IRT calibration needs hundreds of responses PER ITEM to
reliably estimate that item's difficulty and discrimination. AutoIRT
speeds this up by:

  1. Training a flexible machine learning model (here: an ensemble of
     RandomForest + XGBoost + LightGBM) to predict "will this student get
     this item correct?" using the student's estimated ability and the
     item's raw features (not the clean IRT parameters, just the raw
     numeric features like word length, frequency, etc.).

  2. "Translating" that flexible ML model back into the classic,
     interpretable IRT parameters (discrimination `a` and difficulty `d`)
     by finding the (a, d) pair whose theoretical IRT curve looks as
     close as possible to what the ML model predicts, across a range of
     ability levels. This is a curve-fitting step (least squares).

  3. Repeating this a few times in a loop, alternating between:
       - re-estimating each student's ability given the current item
         parameters (this is called the "E-step"), and
       - re-fitting the ML model and re-deriving item parameters given
         the updated abilities (this is called the "M-step").
     This loop is called "Monte Carlo Expectation-Maximization" (MCEM).

Why not just use AutoGluon like the original paper?
------------------------------------------------------
The paper uses a tool called AutoGluon-tabular, which automatically
tries many models and combines them. AutoGluon has a large number of
dependencies and can be slow to install. Here we build a similar
"stacked ensemble of tree models" by hand using three well-known,
lightweight libraries (scikit-learn's RandomForest, XGBoost, LightGBM)
that are already commonly used in industry, and simply average their
predictions. This captures the same spirit (multiple different tree
models voting together) without requiring the heavier dependency.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .simulate import three_parameter_logistic


# Column names used for every feature table passed to the ML ensemble.
# Giving the models a labeled pandas DataFrame (instead of a bare numpy
# array) avoids a harmless but noisy warning that some libraries print
# when a model trained on named columns is later asked to predict on
# unnamed data.
FEATURE_COLUMNS = ["theta", "feature_1", "feature_2"]


# A fixed grid of possible ability (theta) values, used to approximate
# continuous probability distributions with a discrete set of points.
# This mirrors the paper's approach (they also use a grid, since theta
# is a single number and grids work well in one dimension).
DEFAULT_THETA_GRID = np.linspace(-8, 8, 161)


class GradeEnsembleModel:
    """
    A simple stand-in for the paper's AutoML tool (AutoGluon-tabular).

    This trains three different tree-based classifiers on the SAME data
    and averages their predicted probabilities. Using several different
    model types and averaging them tends to give more stable, accurate
    predictions than any single model alone -- this is called
    "ensembling," and it is the same basic idea AutoGluon uses internally
    (just with more models and more automation).
    """

    def __init__(self, random_seed: int = 0):
        self.models = [
            RandomForestClassifier(n_estimators=300, max_depth=8,
                                    random_state=random_seed, n_jobs=-1),
            XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          eval_metric="logloss", random_state=random_seed, n_jobs=-1),
            LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           random_state=random_seed, verbosity=-1, n_jobs=-1),
        ]

    def fit(self, features: np.ndarray, grades: np.ndarray) -> "GradeEnsembleModel":
        """Train all three underlying models on the same (features, grades) data."""
        features_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        for model in self.models:
            model.fit(features_df, grades)
        return self

    def predict_probability_correct(self, features: np.ndarray) -> np.ndarray:
        """Average the three models' predicted probability of a correct answer."""
        features_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        predictions = [model.predict_proba(features_df)[:, 1] for model in self.models]
        return np.mean(predictions, axis=0)


def _fit_irt_curve_to_predictions(predicted_probabilities: np.ndarray,
                                   theta_grid: np.ndarray,
                                   fixed_chance: float = 0.25) -> tuple:
    """
    Given the ML model's predicted probability-correct curve for ONE item
    (one value per theta on the grid), find the (discrimination, difficulty)
    pair whose theoretical 3PL curve is the closest match, in a
    least-squares sense.

    This is a 2-parameter curve-fitting problem. We use the L-BFGS-B
    optimizer with explicit bounds so that:
      - discrimination (a) stays positive (0.05 to 10.0)
      - difficulty (d) stays within a sensible ability range (-6.0 to 6.0)
    Using bounds prevents degenerate solutions (e.g. negative discrimination)
    during fitting, which could otherwise cause cold-start instability
    unrelated to genuine sample-size effects.
    """
    def sum_of_squared_errors(params):
        discrimination, difficulty = params
        predicted_irt_curve = three_parameter_logistic(
            theta=theta_grid, discrimination=discrimination,
            chance=fixed_chance, difficulty=difficulty,
        )
        return np.sum((predicted_irt_curve - predicted_probabilities) ** 2)

    result = minimize(
        sum_of_squared_errors,
        x0=[1.0, 0.0],
        method="L-BFGS-B",
        bounds=[(0.05, 10.0), (-6.0, 6.0)],
    )
    fitted_discrimination, fitted_difficulty = result.x
    return fitted_discrimination, fitted_difficulty


def calibrate_items_given_abilities(responses: dict, items: dict, n_items: int,
                                     current_ability_estimates: np.ndarray,
                                     theta_grid: np.ndarray = DEFAULT_THETA_GRID,
                                     random_seed: int = 0) -> tuple:
    """
    This is the "M-step": given our current best guess at each
    test-taker's ability, fit the ML ensemble and translate it back into
    IRT item parameters for EVERY item (including items with no training
    responses at all -- this is what makes "cold-start" calibration
    possible).

    Parameters
    ----------
    responses : dict
        Output of `simulate_test_responses(...)` (or a subset of it).
    items : dict
        Output of `simulate_items(...)` -- provides the raw item features.
    n_items : int
        Total number of items in the full item bank (including any items
        with zero responses in `responses`).
    current_ability_estimates : np.ndarray
        Current best guess of each session's theta, indexed by session_id.
    theta_grid : np.ndarray
        Grid of theta values used for the curve-fitting step.
    random_seed : int
        Random seed for the underlying ML models.

    Returns
    -------
    (fitted_discrimination, fitted_difficulty, trained_model) : tuple
        Arrays of length n_items with the newly estimated IRT parameters
        for every item, plus the trained ML ensemble itself (useful for
        inspection/debugging).
    """
    # Build the training data: for every observed response, look up that
    # session's current ability estimate and that item's raw features.
    session_thetas = current_ability_estimates[responses["session_id"]]
    item_feature_1 = items["feature_1"][responses["item_id"]]
    item_feature_2 = items["feature_2"][responses["item_id"]]

    training_features = np.column_stack([session_thetas, item_feature_1, item_feature_2])
    training_grades = responses["grade"]

    model = GradeEnsembleModel(random_seed=random_seed).fit(training_features, training_grades)

    # For EVERY item (even ones never seen in training), ask the model
    # "what is the predicted probability-correct at each theta on the grid?"
    # This is what lets AutoIRT calibrate brand-new ("cold-start") items:
    # the model generalizes from item FEATURES, not from item IDs.
    fitted_discrimination = np.zeros(n_items)
    fitted_difficulty = np.zeros(n_items)

    for theta_index, theta_value in enumerate(theta_grid):
        query_features = np.column_stack([
            np.full(n_items, theta_value),
            items["feature_1"],
            items["feature_2"],
        ])
        predicted_probs_at_this_theta = model.predict_probability_correct(query_features)

        # We accumulate a (n_items x n_grid) table across the loop; simplest
        # to build it up column by column here.
        if theta_index == 0:
            predicted_probability_grid = np.zeros((n_items, len(theta_grid)))
        predicted_probability_grid[:, theta_index] = predicted_probs_at_this_theta

    for item_index in range(n_items):
        a_fit, d_fit = _fit_irt_curve_to_predictions(
            predicted_probability_grid[item_index], theta_grid,
        )
        fitted_discrimination[item_index] = a_fit
        fitted_difficulty[item_index] = d_fit

    # Discrimination should always be positive and reasonably sized;
    # clip away any numerical fitting artifacts.
    fitted_discrimination = np.clip(fitted_discrimination, 0.05, 10.0)

    return fitted_discrimination, fitted_difficulty, model


def _compute_session_log_posterior(item_ids_in_session, grades_in_session,
                                    fitted_discrimination, fitted_difficulty,
                                    theta_grid, fixed_chance=0.25, log_prior=None):
    """
    Internal helper: computes the (unnormalized) log-posterior probability
    of each theta value on the grid, for ONE test-taking session, given
    the grades they received and the current item parameter estimates.

    This uses Bayes' rule: posterior ∝ likelihood x prior.
    """
    a = fitted_discrimination[item_ids_in_session][:, None]
    d = fitted_difficulty[item_ids_in_session][:, None]
    probability_correct = three_parameter_logistic(
        theta=theta_grid[None, :], discrimination=a, chance=fixed_chance, difficulty=d,
    )
    probability_correct = np.clip(probability_correct, 1e-6, 1 - 1e-6)

    grades_column = grades_in_session[:, None]
    log_likelihood = np.sum(
        grades_column * np.log(probability_correct) +
        (1 - grades_column) * np.log(1 - probability_correct),
        axis=0,
    )

    if log_prior is None:
        log_prior = 0.0

    log_posterior = log_likelihood + log_prior
    log_posterior -= log_posterior.max()  # numerical stability
    return log_posterior


def resample_abilities(responses: dict, fitted_discrimination: np.ndarray,
                        fitted_difficulty: np.ndarray, fixed_chance: float = 0.25,
                        theta_grid: np.ndarray = DEFAULT_THETA_GRID,
                        ability_prior_std: float = np.sqrt(2.5),
                        random_seed: int = 0) -> tuple:
    """
    This is the "E-step": given the current item parameters, draw a new
    random sample of each session's ability from its posterior
    distribution. This randomness is what makes the algorithm "Monte
    Carlo" EM rather than plain EM.

    Returns
    -------
    (session_ids, resampled_theta) : tuple of np.ndarray
        The session ids present in `responses`, and a freshly sampled
        theta value for each one.
    """
    rng = np.random.default_rng(random_seed)
    log_prior = -0.5 * (theta_grid / ability_prior_std) ** 2

    sort_order = np.argsort(responses["session_id"])
    sorted_session_ids = responses["session_id"][sort_order]
    sorted_item_ids = responses["item_id"][sort_order]
    sorted_grades = responses["grade"][sort_order]

    unique_session_ids, group_start_positions = np.unique(sorted_session_ids, return_index=True)
    group_boundaries = list(group_start_positions) + [len(sorted_session_ids)]

    resampled_theta = np.zeros(len(unique_session_ids))

    for group_index in range(len(unique_session_ids)):
        this_group = slice(group_boundaries[group_index], group_boundaries[group_index + 1])
        items_this_session = sorted_item_ids[this_group]
        grades_this_session = sorted_grades[this_group]

        log_posterior = _compute_session_log_posterior(
            items_this_session, grades_this_session,
            fitted_discrimination, fitted_difficulty, theta_grid,
            fixed_chance=fixed_chance, log_prior=log_prior,
        )
        posterior_probabilities = np.exp(log_posterior)
        posterior_probabilities /= posterior_probabilities.sum()

        resampled_theta[group_index] = rng.choice(theta_grid, p=posterior_probabilities)

    return unique_session_ids, resampled_theta


def compute_posterior_mean_abilities(responses: dict, fitted_discrimination: np.ndarray,
                                      fitted_difficulty: np.ndarray, fixed_chance: float = 0.25,
                                      theta_grid: np.ndarray = DEFAULT_THETA_GRID,
                                      ability_prior_std: float = np.sqrt(2.5)) -> tuple:
    """
    Computes the posterior MEAN ability for each session (as opposed to a
    random sample). This is what you would report as the student's final
    SCORE (see equation 5 in the paper), since a mean is a more stable
    point estimate than a single random draw.

    Returns
    -------
    (session_ids, posterior_mean_theta) : tuple of np.ndarray
    """
    log_prior = -0.5 * (theta_grid / ability_prior_std) ** 2

    sort_order = np.argsort(responses["session_id"])
    sorted_session_ids = responses["session_id"][sort_order]
    sorted_item_ids = responses["item_id"][sort_order]
    sorted_grades = responses["grade"][sort_order]

    unique_session_ids, group_start_positions = np.unique(sorted_session_ids, return_index=True)
    group_boundaries = list(group_start_positions) + [len(sorted_session_ids)]

    posterior_means = np.zeros(len(unique_session_ids))

    for group_index in range(len(unique_session_ids)):
        this_group = slice(group_boundaries[group_index], group_boundaries[group_index + 1])
        items_this_session = sorted_item_ids[this_group]
        grades_this_session = sorted_grades[this_group]

        log_posterior = _compute_session_log_posterior(
            items_this_session, grades_this_session,
            fitted_discrimination, fitted_difficulty, theta_grid,
            fixed_chance=fixed_chance, log_prior=log_prior,
        )
        posterior_probabilities = np.exp(log_posterior)
        posterior_probabilities /= posterior_probabilities.sum()

        posterior_means[group_index] = np.sum(theta_grid * posterior_probabilities)

    return unique_session_ids, posterior_means


def run_autoirt_calibration(responses: dict, items: dict, n_items: int,
                             n_em_steps: int = 4, random_seed: int = 0) -> dict:
    """
    Runs the full AutoIRT calibration loop (Algorithm 1 in the paper):
    alternates between the M-step (fit ML model, derive IRT parameters)
    and the E-step (resample abilities), for a fixed number of rounds.

    Parameters
    ----------
    responses : dict
        The training data (see `simulate_test_responses`).
    items : dict
        The item bank (see `simulate_items`). Note this can include items
        with ZERO responses in `responses` -- those are "cold-start"
        items, and AutoIRT will still produce parameter estimates for
        them using only their raw features.
    n_items : int
        Total number of items in the bank (including cold-start items).
    n_em_steps : int
        How many rounds of M-step / E-step to run (paper uses 4).
    random_seed : int
        Random seed, for reproducibility.

    Returns
    -------
    dict with keys:
        "discrimination" : final fitted discrimination for every item
        "difficulty"      : final fitted difficulty for every item
        "trained_model"   : the final trained ML ensemble
    """
    n_sessions = int(responses["session_id"].max()) + 1

    # Initialize ability estimates by drawing from the prior distribution
    # Normal(0, sqrt(2.5)), which is what the paper assumes for theta.
    # This is more faithful to the paper than starting everyone at zero,
    # and gives the M-step a better starting point on the first iteration.
    rng = np.random.default_rng(random_seed)
    ability_estimates = rng.normal(loc=0.0, scale=np.sqrt(2.5), size=n_sessions)

    fitted_discrimination = None
    fitted_difficulty = None
    trained_model = None

    for step in range(n_em_steps):
        fitted_discrimination, fitted_difficulty, trained_model = calibrate_items_given_abilities(
            responses, items, n_items, ability_estimates, random_seed=random_seed + step,
        )

        is_last_step = (step == n_em_steps - 1)
        if not is_last_step:
            session_ids, new_thetas = resample_abilities(
                responses, fitted_discrimination, fitted_difficulty, random_seed=random_seed + step,
            )
            ability_estimates[session_ids] = new_thetas

        print(f"  [AutoIRT] Completed EM step {step + 1} of {n_em_steps}.")

    return {
        "discrimination": fitted_discrimination,
        "difficulty": fitted_difficulty,
        "trained_model": trained_model,
    }