"""
Item and response simulation for the DET-style phase (Phase 2). Where
run_experiment.py replicates the abstract simulation study from the
paper's supplement, this replicates the two-item-type setup used in
the paper's "Offline calibration analysis for DET" section, so the
cold/jump/warm-start protocol can be tested against something closer
to their actual setup.
 
Real Duolingo data hasn't arrived yet, so this generates two item
banks the same way simulate.py does -- same 3PL model, same
two-feature non-linear item generation -- but with chance parameters
set to what the paper uses per item type: c=0.25 for Y/N Vocab
(multiple choice, guessing gets you 25%) and c=0 for ViC (fill-in-the-
blank, not really guessable). Y/N Vocab and ViC each get their own
ability parameter, matching the paper's separate modeling of the two.
 
The paper splits by calendar date (2024-04-01 to 2024-06-18, with
cutoffs at 2024-05-22 for cold/jump-start and 2024-05-08 / 2024-05-15
for the two warm-start splits). Without real timestamps, each session
gets a random simulated "day" over that same 79-day window, with the
day offsets below matching the paper's actual dates so the splits
land in the same relative spot.
"""

import numpy as np

from .simulate import three_parameter_logistic, simulate_test_taker_abilities


TOTAL_SIMULATED_DAYS = 79
COLD_JUMP_SPLIT_DAY = 51   # 2024-05-22
WARM_SPLIT_DAY_1 = 37      # 2024-05-08
WARM_SPLIT_DAY_2 = 44      # 2024-05-15


def simulate_det_items(n_items: int, chance_value: float,
                        effect_noise_std: float = 0.1, random_seed: int = None) -> dict:
    """Same item-generating process as simulate.simulate_items, just with
    the chance parameter as an argument instead of hardcoded to 0.25,
    since Y/N Vocab and ViC use different values."""
    rng = np.random.default_rng(random_seed)

    feature_1 = rng.uniform(-10, 10, size=n_items)
    feature_2 = rng.uniform(-10, 10, size=n_items)
    z = feature_1 - feature_2
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)

    typical_difficulty = 4 * np.sin(z_safe) / np.abs(z_safe)
    typical_discrimination = 0.5 * np.cos(0.1 * z ** 2) + 1.0

    difficulty_noise = rng.normal(0, effect_noise_std, size=n_items)
    discrimination_noise = rng.normal(0, effect_noise_std, size=n_items)

    difficulty = typical_difficulty + difficulty_noise
    discrimination = np.exp(np.log(typical_discrimination) + discrimination_noise)

    chance = np.full(n_items, chance_value)

    return {
        "feature_1": feature_1,
        "feature_2": feature_2,
        "discrimination": discrimination,
        "difficulty": difficulty,
        "chance": chance,
    }


def simulate_det_sessions(n_sessions: int, random_seed: int = None) -> dict:
    """Generates the per-session stuff shared across both item types: a
    Y/N Vocab ability and a ViC ability (drawn independently -- the paper
    doesn't say these should be correlated, so independence is the
    simplest assumption), plus a simulated day each session happened on."""
    rng = np.random.default_rng(random_seed)
    yn_theta = simulate_test_taker_abilities(n_sessions, random_seed=rng.integers(1e9))
    vic_theta = simulate_test_taker_abilities(n_sessions, random_seed=rng.integers(1e9))
    session_day = rng.integers(0, TOTAL_SIMULATED_DAYS, size=n_sessions)
    return {"yn_theta": yn_theta, "vic_theta": vic_theta, "session_day": session_day}


def simulate_det_responses(items: dict, theta_by_session: np.ndarray, session_day: np.ndarray,
                            items_per_session: int, random_seed: int = None) -> dict:
    """Same as simulate.simulate_test_responses, but each response also
    carries the day it happened on and a running sequence number. The
    sequence number is just there to break ties when two responses land
    on the same simulated day, so "first R responses after the split
    date" (needed for jump-start) has an actual order to sort by."""
    rng = np.random.default_rng(random_seed)
    n_items = len(items["discrimination"])
    n_sessions = len(theta_by_session)

    session_id_list = []
    item_id_list = []
    grade_list = []
    day_list = []
    response_seq_list = []
    response_seq_counter = 0

    for session in range(n_sessions):
        administered_items = rng.choice(n_items, size=items_per_session, replace=False)
        probability_correct = three_parameter_logistic(
            theta=theta_by_session[session],
            discrimination=items["discrimination"][administered_items],
            chance=items["chance"][administered_items],
            difficulty=items["difficulty"][administered_items],
        )
        grades = rng.binomial(n=1, p=probability_correct)

        session_id_list.extend([session] * items_per_session)
        item_id_list.extend(administered_items.tolist())
        grade_list.extend(grades.tolist())
        day_list.extend([session_day[session]] * items_per_session)
        response_seq_list.extend(range(response_seq_counter, response_seq_counter + items_per_session))
        response_seq_counter += items_per_session

    return {
        "session_id": np.array(session_id_list),
        "item_id": np.array(item_id_list),
        "grade": np.array(grade_list),
        "day": np.array(day_list),
        "response_seq": np.array(response_seq_list),
        "true_theta": theta_by_session[np.array(session_id_list)],
    }