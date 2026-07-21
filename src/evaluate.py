"""
evaluate.py
===========

This module measures how good a calibration is, using the same three
metrics reported in the AutoIRT paper's results tables:

  1. Test loss (mean negative log-likelihood)
     "On average, how surprised was our model by the actual right/wrong
      outcomes?" Lower is better. A perfect model would give a very low
      number here; a model that is no better than a coin flip would give
      a higher number.

  2. Item-level grade correlation (Pearson and Spearman)
     For each item, compare "the actual fraction of people who got it
     right" to "the fraction our model predicted would get it right."
     If these two line up well across all items, the correlation is
     close to 1.0.

  3. Ability (theta) recovery correlation
     Only possible because this is SIMULATED data, where we secretly know
     the true ability of every test-taker. We check: does our model's
     estimated ability line up with the true, hidden ability? This tells
     us how well the whole system (not just individual items) is working.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .simulate import three_parameter_logistic
from .autoirt_model import compute_posterior_mean_abilities


def compute_test_loss(responses: dict, session_id_to_theta: dict,
                       fitted_discrimination: np.ndarray, fitted_difficulty: np.ndarray,
                       fixed_chance: float = 0.25) -> float:
    """
    Metric 1: mean negative log-likelihood ("test loss").

    Lower values mean the model's predicted probabilities matched the
    actual correct/incorrect outcomes more closely.
    """
    session_thetas = np.array([session_id_to_theta[s] for s in responses["session_id"]])
    item_discrimination = fitted_discrimination[responses["item_id"]]
    item_difficulty = fitted_difficulty[responses["item_id"]]

    predicted_probability = three_parameter_logistic(
        theta=session_thetas, discrimination=item_discrimination,
        chance=fixed_chance, difficulty=item_difficulty,
    )
    predicted_probability = np.clip(predicted_probability, 1e-6, 1 - 1e-6)

    actual_grade = responses["grade"]
    negative_log_likelihood = -(
        actual_grade * np.log(predicted_probability) +
        (1 - actual_grade) * np.log(1 - predicted_probability)
    )
    return float(negative_log_likelihood.mean())


def compute_item_grade_correlation(responses: dict, session_id_to_theta: dict,
                                    fitted_discrimination: np.ndarray, fitted_difficulty: np.ndarray,
                                    n_items: int, fixed_chance: float = 0.25) -> tuple:
    """
    Metric 2: item-level grade correlation (Pearson and Spearman).

    For every item, computes:
      - the actual average grade (fraction of correct answers), and
      - the average PREDICTED probability of being correct,
    then correlates these two lists across all items.
    """
    session_thetas = np.array([session_id_to_theta[s] for s in responses["session_id"]])
    item_discrimination = fitted_discrimination[responses["item_id"]]
    item_difficulty = fitted_difficulty[responses["item_id"]]

    predicted_probability = three_parameter_logistic(
        theta=session_thetas, discrimination=item_discrimination,
        chance=fixed_chance, difficulty=item_difficulty,
    )

    sum_actual_grade = np.zeros(n_items)
    sum_predicted_probability = np.zeros(n_items)
    response_count = np.zeros(n_items)

    np.add.at(sum_actual_grade, responses["item_id"], responses["grade"])
    np.add.at(sum_predicted_probability, responses["item_id"], predicted_probability)
    np.add.at(response_count, responses["item_id"], 1)

    items_with_data = response_count > 0
    mean_actual_grade = sum_actual_grade[items_with_data] / response_count[items_with_data]
    mean_predicted_probability = sum_predicted_probability[items_with_data] / response_count[items_with_data]

    pearson_correlation, _ = pearsonr(mean_actual_grade, mean_predicted_probability)
    spearman_correlation, _ = spearmanr(mean_actual_grade, mean_predicted_probability)
    return pearson_correlation, spearman_correlation


def compute_ability_recovery_correlation(true_theta: np.ndarray, estimated_theta: np.ndarray) -> float:
    """
    Metric 3: ability (theta) recovery correlation.

    Only meaningful in a simulation study, where the true ability of each
    test-taker is known (in a real study, nobody's true ability is ever
    directly observable).
    """
    correlation, _ = pearsonr(true_theta, estimated_theta)
    return float(correlation)


def evaluate_calibration(responses: dict, true_theta_by_session: np.ndarray,
                          fitted_discrimination: np.ndarray, fitted_difficulty: np.ndarray,
                          n_items: int, fixed_chance: float = 0.25) -> dict:
    """
    Convenience function: runs all three metrics at once and returns them
    together in a dictionary, ready to print or log.

    Parameters
    ----------
    responses : dict
        The (typically held-out / test) responses to evaluate on.
    true_theta_by_session : np.ndarray
        Full array of TRUE ability values, indexed by session_id. Only
        used for the ability-recovery metric.
    fitted_discrimination, fitted_difficulty : np.ndarray
        The item parameters estimated by AutoIRT.
    n_items : int
        Total number of items in the bank.
    fixed_chance : float
        The chance/guessing parameter used throughout (0.25 here).
    """
    session_ids, posterior_mean_theta = compute_posterior_mean_abilities(
        responses, fitted_discrimination, fitted_difficulty, fixed_chance=fixed_chance,
    )
    session_id_to_theta = dict(zip(session_ids.tolist(), posterior_mean_theta.tolist()))

    test_loss = compute_test_loss(
        responses, session_id_to_theta, fitted_discrimination, fitted_difficulty, fixed_chance,
    )
    pearson_correlation, spearman_correlation = compute_item_grade_correlation(
        responses, session_id_to_theta, fitted_discrimination, fitted_difficulty, n_items, fixed_chance,
    )
    ability_recovery = compute_ability_recovery_correlation(
        true_theta_by_session[session_ids], posterior_mean_theta,
    )

    return {
        "test_loss_nll": test_loss,
        "item_grade_pearson_r": pearson_correlation,
        "item_grade_spearman_r": spearman_correlation,
        "ability_recovery_pearson_r": ability_recovery,
    }
