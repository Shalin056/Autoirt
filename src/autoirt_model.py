"""
Implements the AutoIRT calibration algorithm from Sharpnack et al. (2024),
"AutoIRT: Calibrating Item Response Theory Models with Automated Machine
Learning" (Algorithm 1).

Traditional IRT calibration needs hundreds of responses per item to get a
reliable difficulty/discrimination estimate. AutoIRT gets around that by
training a flexible ML model on (ability, item features) -> correct/wrong,
then translating that model back into interpretable IRT parameters by
curve-fitting: finding the (a, d) pair whose theoretical IRT curve best
matches what the ML model predicts across a range of ability levels. This
repeats a few times, alternating between re-estimating each student's
ability given the current item parameters (E-step) and re-fitting the ML
model / re-deriving item parameters given the updated abilities (M-step) --
Monte Carlo EM.

The paper uses AutoGluon-tabular for the ML step. AutoGluon is a heavy
dependency, so this uses a hand-built stacked ensemble instead
(RandomForest + XGBoost + LightGBM, predictions averaged) -- same basic
idea, lighter footprint.
"""

import shutil
import tempfile
import weakref

import numpy as np
import pandas as pd
from collections import deque
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .simulate import three_parameter_logistic


# Giving the models a labeled DataFrame instead of a bare array avoids a
# harmless warning some libraries print when a model trained on named
# columns later predicts on unnamed data.
FEATURE_COLUMNS = ["theta", "feature_1", "feature_2"]

# Grid of possible ability values, used to approximate continuous
# posterior distributions with a discrete set of points -- same approach
# the paper uses, since theta is one-dimensional and grids work fine there.
DEFAULT_THETA_GRID = np.linspace(-8, 8, 161)


class GradeEnsembleModel:
    """Stand-in for the paper's AutoGluon-tabular step: trains three
    different tree-based classifiers on the same data and averages their
    predicted probabilities. Same idea as AutoGluon's internal stacking,
    just fewer models and no automated search."""

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
        features_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        for model in self.models:
            model.fit(features_df, grades)
        return self

    def predict_probability_correct(self, features: np.ndarray) -> np.ndarray:
        features_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        predictions = [model.predict_proba(features_df)[:, 1] for model in self.models]
        return np.mean(predictions, axis=0)


class AutoGluonGradeModel:
    """The paper's actual backend, for a direct comparison against
    GradeEnsembleModel. Same fit/predict_probability_correct interface so
    it's a drop-in swap -- see `backend=` in calibrate_items_given_abilities
    and run_autoirt_calibration.

    Needs `pip install autogluon.tabular` (not in requirements.txt by
    default since it's a heavy install -- multiple GB of dependencies).
    Also worth running `pip install catboost` separately: the paper's
    ensemble includes CatBoost, but the base autogluon.tabular install
    skips it (along with a couple of neural-net models it can't use
    without extra installs) and just prints warnings and continues
    without it. AutoGluon still runs fine either way -- this is only
    about getting the same model set the paper actually used.
    Raises a clear error if autogluon itself isn't installed rather than
    failing on some unrelated import error deeper in.
    """

    def __init__(self, random_seed: int = 0, time_limit: int = 60):
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as e:
            raise ImportError(
                "AutoGluonGradeModel needs autogluon.tabular, which isn't "
                "installed. Run `pip install autogluon.tabular` first (it's "
                "a heavy install, expect several GB and a few minutes)."
            ) from e
        self._TabularPredictor = TabularPredictor
        self.random_seed = random_seed
        # Per-fit time budget passed straight to AutoGluon's `fit(time_limit=...)`.
        # Higher values let AutoGluon try more/heavier model configurations
        # per M-step at the cost of longer runtime; see run_backend_comparision.py
        # and run_item_bank_sweep.py for scripts that vary this.
        self.time_limit = time_limit
        self.predictor = None
        # KNN and the neural net variants are excluded: slowest/heaviest
        # for negligible gain on this small, purely numeric feature set
        # (2-3 raw features), so skipping them saves both training time
        # and on-disk footprint (fewer .pkl files per fit).
        self.excluded_model_types = ["KNN", "NN_TORCH", "FASTAI"]
        self._model_dir = tempfile.mkdtemp(prefix="autogluon_tmp_")
        # Cleanup is tied to this object's lifetime rather than to the end
        # of fit(): a stacked ensemble's base learners (e.g. CatBoost) can
        # get lazy-loaded from disk on a later predict call, so the model
        # directory needs to stay around as long as the model might still
        # be used. weakref.finalize removes _model_dir only once nothing
        # holds a reference to this model anymore -- in practice that's
        # right after the next EM step creates a new model and this one
        # becomes unreachable, so disk usage stays bounded to one or two
        # live model folders at a time.
        #
        # NOTE (Windows): rmtree with ignore_errors=True will silently
        # swallow a WinError 32 ("file in use") if some library
        # (LightGBM/CatBoost/joblib) hasn't released its file handle the
        # instant this object is garbage collected. If that happens,
        # folders pile up in %TEMP% anyway -- check %TEMP% for leftover
        # autogluon_tmp_* folders if disk space is tight.
        self._cleanup = weakref.finalize(
            self, shutil.rmtree, self._model_dir, ignore_errors=True,
        )

    def fit(self, features: np.ndarray, grades: np.ndarray) -> "AutoGluonGradeModel":
        train_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        train_df["grade"] = grades
        self.predictor = self._TabularPredictor(
            label="grade", problem_type="binary", eval_metric="log_loss", verbosity=0,
            path=self._model_dir,
        ).fit(
            train_df, time_limit=self.time_limit, presets="medium_quality",
            excluded_model_types=self.excluded_model_types,
        )
        return self

    def predict_probability_correct(self, features: np.ndarray) -> np.ndarray:
        features_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        return self.predictor.predict_proba(features_df)[1].to_numpy()


def _make_grade_model(backend: str, random_seed: int, autogluon_time_limit: int = 60):
    """Picks the ML backend by name -- "ensemble" (default, RF+XGB+LGBM,
    no extra install) or "autogluon" (paper's actual tool, needs
    autogluon.tabular installed separately). autogluon_time_limit only
    applies to the "autogluon" backend."""
    if backend == "ensemble":
        return GradeEnsembleModel(random_seed=random_seed)
    elif backend == "autogluon":
        return AutoGluonGradeModel(random_seed=random_seed, time_limit=autogluon_time_limit)
    raise ValueError(f"Unknown backend '{backend}': use 'ensemble' or 'autogluon'.")


def _fit_irt_curve_to_predictions(predicted_probabilities: np.ndarray,
                                   theta_grid: np.ndarray,
                                   fixed_chance: float = 0.25) -> tuple:
    """Fits (discrimination, difficulty) to the ML model's predicted
    probability-correct curve for one item, least-squares style.

    Discrimination has to stay positive -- a negative-slope 3PL curve
    isn't a real IRT model -- and difficulty needs to stay within the
    theta grid's range. scipy.optimize.curve_fit is given explicit bounds
    on both parameters so the fit stays in range by construction, which
    matters on noisy predicted curves (common early in the EM loop,
    before ability estimates are any good).
    """
    def model(theta, discrimination, difficulty):
        return three_parameter_logistic(
            theta=theta, discrimination=discrimination,
            chance=fixed_chance, difficulty=difficulty,
        )

    lower_bounds = [0.05, theta_grid.min()]
    upper_bounds = [10.0, theta_grid.max()]
    try:
        fitted_params, _ = curve_fit(
            model, theta_grid, predicted_probabilities,
            p0=[1.0, 0.0], bounds=(lower_bounds, upper_bounds), maxfev=5000,
        )
        fitted_discrimination, fitted_difficulty = fitted_params
    except RuntimeError:
        # curve_fit didn't converge (rare) -- fall back to a coarse grid
        # search over the same bounded region instead of returning junk.
        a_candidates = np.linspace(lower_bounds[0], upper_bounds[0], 20)
        d_candidates = np.linspace(lower_bounds[1], upper_bounds[1], 20)
        best_a, best_d, best_sse = a_candidates[0], d_candidates[0], np.inf
        for a in a_candidates:
            curve = three_parameter_logistic(theta=theta_grid, discrimination=a,
                                              chance=fixed_chance, difficulty=d_candidates[:, None])
            sse = np.sum((curve - predicted_probabilities[None, :]) ** 2, axis=1)
            idx = np.argmin(sse)
            if sse[idx] < best_sse:
                best_sse, best_a, best_d = sse[idx], a, d_candidates[idx]
        fitted_discrimination, fitted_difficulty = best_a, best_d

    return fitted_discrimination, fitted_difficulty


def calibrate_items_given_abilities(responses: dict, items: dict, n_items: int,
                                     current_ability_estimates: np.ndarray,
                                     theta_grid: np.ndarray = DEFAULT_THETA_GRID,
                                     random_seed: int = 0,
                                     backend: str = "ensemble",
                                     autogluon_time_limit: int = 60) -> tuple:
    """The M-step: given our current guess at each test-taker's ability,
    fit the ML model and translate it back into IRT parameters for every
    item -- including items with zero training responses, which is what
    makes cold-start calibration possible (the model generalizes from item
    features, not item IDs).

    responses: output of simulate_test_responses (or a subset).
    items: output of simulate_items -- the raw item features.
    n_items: total items in the bank, including any with zero responses.
    current_ability_estimates: current theta guess per session, indexed
        by session_id.
    theta_grid: grid used for the curve-fitting step.
    random_seed: seed for the underlying ML models.
    backend: "ensemble" (default, RF+XGB+LGBM, no extra install) or
        "autogluon" (the paper's actual tool -- needs autogluon.tabular
        installed separately; see AutoGluonGradeModel).

    Returns (fitted_discrimination, fitted_difficulty, trained_model,
    training_loss_nonparametric, training_loss_parametric) -- item
    parameter arrays, the trained model, and two training-loss numbers
    (raw ML predictions vs. the 3PL fit derived from them) used to check
    EM convergence, the same diagnostic the paper uses in Figure 6.
    """
    # Build the training table: for every observed response, look up that
    # session's current ability estimate and that item's raw features.
    session_thetas = current_ability_estimates[responses["session_id"]]
    item_feature_1 = items["feature_1"][responses["item_id"]]
    item_feature_2 = items["feature_2"][responses["item_id"]]

    training_features = np.column_stack([session_thetas, item_feature_1, item_feature_2])
    training_grades = responses["grade"]

    model = _make_grade_model(backend, random_seed, autogluon_time_limit).fit(training_features, training_grades)

    # Ask the model for a predicted probability-correct at every theta on
    # the grid, for every item (even ones never seen in training).
    fitted_discrimination = np.zeros(n_items)
    fitted_difficulty = np.zeros(n_items)

    for theta_index, theta_value in enumerate(theta_grid):
        query_features = np.column_stack([
            np.full(n_items, theta_value),
            items["feature_1"],
            items["feature_2"],
        ])
        predicted_probs_at_this_theta = model.predict_probability_correct(query_features)

        if theta_index == 0:
            predicted_probability_grid = np.zeros((n_items, len(theta_grid)))
        predicted_probability_grid[:, theta_index] = predicted_probs_at_this_theta

    for item_index in range(n_items):
        a_fit, d_fit = _fit_irt_curve_to_predictions(
            predicted_probability_grid[item_index], theta_grid,
        )
        fitted_discrimination[item_index] = a_fit
        fitted_difficulty[item_index] = d_fit

    # Belt-and-suspenders clip in case of any numerical fitting artifacts.
    fitted_discrimination = np.clip(fitted_discrimination, 0.05, 10.0)

    # Training-loss diagnostic (mirrors the paper's Figure 6): compare the
    # raw ML ensemble's training loss against the loss after projecting it
    # onto the 3PL curve, so the EM loop can check convergence directly
    # instead of trusting a fixed step count.
    nonparametric_predictions = model.predict_probability_correct(training_features)
    parametric_predictions = three_parameter_logistic(
        theta=session_thetas,
        discrimination=fitted_discrimination[responses["item_id"]],
        chance=0.25,
        difficulty=fitted_difficulty[responses["item_id"]],
    )
    nonparametric_predictions = np.clip(nonparametric_predictions, 1e-6, 1 - 1e-6)
    parametric_predictions = np.clip(parametric_predictions, 1e-6, 1 - 1e-6)

    def _mean_nll(p):
        return float(-(
            training_grades * np.log(p) + (1 - training_grades) * np.log(1 - p)
        ).mean())

    training_loss_nonparametric = _mean_nll(nonparametric_predictions)
    training_loss_parametric = _mean_nll(parametric_predictions)

    return (fitted_discrimination, fitted_difficulty, model,
            training_loss_nonparametric, training_loss_parametric)


def _compute_session_log_posterior(item_ids_in_session, grades_in_session,
                                    fitted_discrimination, fitted_difficulty,
                                    theta_grid, fixed_chance=0.25, log_prior=None):
    """Unnormalized log-posterior over the theta grid for one session,
    given their grades and the current item parameters (posterior ∝
    likelihood x prior)."""
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
    """The E-step: given the current item parameters, draw a fresh random
    sample of each session's ability from its posterior. The randomness
    here is the "Monte Carlo" part of Monte Carlo EM.

    Returns (session_ids, resampled_theta)."""
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
    """Posterior MEAN ability per session (as opposed to a random sample)
    -- this is what gets reported as a test-taker's score (paper's
    equation 5), since a mean is a more stable point estimate than one
    random draw.

    Returns (session_ids, posterior_mean_theta)."""
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
                             n_em_steps: int = 4, random_seed: int = 0,
                             convergence_tolerance: float = 0.005,
                             max_em_steps: int = 20,
                             convergence_window: int = 6,
                             min_hits_in_window: int = 3,
                             backend: str = "ensemble",
                             autogluon_time_limit: int = 60) -> dict:
    """Runs the full AutoIRT calibration loop (Algorithm 1 in the paper):
    alternates M-step (fit ML model, derive IRT parameters) and E-step
    (resample abilities) until the training loss looks converged.

    responses: training data (simulate_test_responses).
    items: item bank (simulate_items) -- can include items with zero
        responses in `responses`; those are the cold-start items, and
        AutoIRT still produces estimates for them from features alone.
    n_items: total items in the bank, including cold-start items.
    n_em_steps: minimum EM steps to run before checking convergence.
        Default 4, matching the paper's own number, but used here as a
        floor rather than a hard stop -- see the note below on why.
    random_seed: seed for reproducibility.
    convergence_tolerance: a step counts as "small" once the relative
        change in training loss from the previous step drops below this.
        Default 0.005 (0.5%).
    convergence_window / min_hits_in_window: convergence is declared once
        at least min_hits_in_window of the last convergence_window steps
        were "small" -- defaults 3 of the last 6. Requiring a majority
        rather than an exact run of consecutive small steps makes this
        more tolerant of one noisy step in the middle of an otherwise
        flat trajectory.
    max_em_steps: hard cap in case the loss never settles. Default 20.
    backend: "ensemble" (default) or "autogluon" -- see AutoGluonGradeModel
        above. Used to test how much of any remaining gap vs. the paper is
        the calibration procedure vs. the specific AutoML tool.

    Returns a dict with "discrimination", "difficulty", "trained_model",
    "training_loss_history" (per-step nonparametric/parametric loss),
    "stopped_reason" ("converged" or "hit_max_em_steps"), and
    "n_steps_run".

    On the termination rule: Algorithm 1 in the paper (top of p. 4) is
    just "for each EM iteration do" -- no stopping condition. Their choice
    of 4 steps comes from a separate empirical note in the Results section
    (loss was roughly flat within 4 steps for their specific settings),
    not a rule that's guaranteed to hold at a different scale or with a
    different AutoML backend -- hence checking it here instead of copying
    their number. The standard reference for an automated MCEM stopping
    rule is Booth & Hobert (1999), "Maximizing generalized linear mixed
    model likelihoods with an automated Monte Carlo EM algorithm," JRSS B,
    61(1), 265-285: they build a confidence interval for each iteration's
    parameter update from its Monte Carlo error, and stop once the update
    is statistically indistinguishable from zero. Their version needs
    machinery this project doesn't have (a standard error per parameter
    from repeated MC draws), so what's here is a simplified version in
    the same spirit, using the training loss that's already being tracked.
    """
    n_sessions = int(responses["session_id"].max()) + 1
    # Start from a random draw of the ability prior (theta ~ N(0, sqrt(2.5)))
    # per session instead of theta=0 for everyone -- if every session starts
    # at the same value, the "theta" column has zero variance in the first
    # M-step's training data, so the model can't learn any theta-dependence
    # at all in that first round.
    init_rng = np.random.default_rng(random_seed)
    ability_estimates = init_rng.normal(0.0, np.sqrt(2.5), size=n_sessions)

    fitted_discrimination = None
    fitted_difficulty = None
    trained_model = None
    training_loss_history = []
    recent_small_change_flags = deque(maxlen=convergence_window)
    stopped_reason = "hit_max_em_steps"

    for step in range(max_em_steps):
        (fitted_discrimination, fitted_difficulty, trained_model,
         loss_nonparametric, loss_parametric) = calibrate_items_given_abilities(
            responses, items, n_items, ability_estimates, random_seed=random_seed + step,
            backend=backend, autogluon_time_limit=autogluon_time_limit,
        )
        training_loss_history.append({
            "step": step + 1,
            "loss_nonparametric": loss_nonparametric,
            "loss_parametric": loss_parametric,
        })

        # Convergence check
        past_minimum_steps = (step + 1) >= n_em_steps
        if len(training_loss_history) >= 2:
            previous_loss = training_loss_history[-2]["loss_parametric"]
            relative_change = abs(loss_parametric - previous_loss) / max(abs(previous_loss), 1e-8)
            recent_small_change_flags.append(relative_change < convergence_tolerance)
        else:
            relative_change = None

        window_full = len(recent_small_change_flags) == convergence_window
        has_converged = (
            past_minimum_steps and window_full
            and sum(recent_small_change_flags) >= min_hits_in_window
        )
        is_last_step = has_converged or (step == max_em_steps - 1)

        rel_change_str = f"{relative_change:.4f}" if relative_change is not None else "n/a"
        print(f"  [AutoIRT] Completed EM step {step + 1}"
              f"{' (of max ' + str(max_em_steps) + ')' if not has_converged else ''}. "
              f"(train loss: nonparametric={loss_nonparametric:.4f}, "
              f"parametric={loss_parametric:.4f}, rel. change={rel_change_str})")

        if not is_last_step:
            session_ids, new_thetas = resample_abilities(
                responses, fitted_discrimination, fitted_difficulty, random_seed=random_seed + step,
            )
            ability_estimates[session_ids] = new_thetas
        else:
            stopped_reason = "converged" if has_converged else "hit_max_em_steps"
            break

    if stopped_reason == "hit_max_em_steps":
        print(f"  [AutoIRT] WARNING: reached max_em_steps={max_em_steps} without the "
              f"relative loss change staying below {convergence_tolerance} for at least "
              f"{min_hits_in_window} of the last {convergence_window} steps. Training loss "
              f"may not have fully converged -- check training_loss_history before trusting "
              f"these item parameters.")

    return {
        "discrimination": fitted_discrimination,
        "difficulty": fitted_difficulty,
        "trained_model": trained_model,
        "training_loss_history": training_loss_history,
        "stopped_reason": stopped_reason,
        "n_steps_run": len(training_loss_history),
    }