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

import time

import numpy as np

from .simulate import three_parameter_logistic, simulate_test_taker_abilities


TOTAL_SIMULATED_DAYS = 79
COLD_JUMP_SPLIT_DAY = 51   # 2024-05-22
WARM_SPLIT_DAY_1 = 37      # 2024-05-08
WARM_SPLIT_DAY_2 = 44      # 2024-05-15

# Same grid and prior std as autoirt_model.DEFAULT_THETA_GRID /
# ability_prior_std -- kept as a local copy rather than importing
# autoirt_model here, since that module pulls in sklearn/xgboost/
# lightgbm/autogluon just for one constant, which this purely-data-
# generation module shouldn't need to depend on.
ADAPTIVE_SELECTION_THETA_GRID = np.linspace(-8, 8, 161)
ABILITY_PRIOR_STD = np.sqrt(2.5)


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


def _fisher_information_3pl(theta, discrimination, chance, difficulty):
    """Fisher information for the 3PL model, equation (2) of the
    BanditCAT paper (Sharpnack et al. 2024). `theta` and the item
    parameter arrays are broadcast against each other, so this can score
    many (theta, item) pairs in one call -- used here as
    theta[:, None] x item_params[None, :] to get an (n_theta_draws,
    n_eligible_items) matrix in one shot."""
    p2 = 1 / (1 + np.exp(-discrimination * (theta - difficulty)))
    p = chance + (1 - chance) * p2
    numerator = (discrimination * (1 - chance) * p2 * (1 - p2)) ** 2
    denominator = np.where(p * (1 - p) < 1e-12, 1e-12, p * (1 - p))
    return numerator / denominator


def simulate_det_responses_adaptive(items: dict, theta_by_session: np.ndarray, session_day: np.ndarray,
                                     items_per_session: int, exposure_gamma: float = 0.5,
                                     n_theta_draws: int = 5, random_injection_rate: float = 0.0,
                                     random_seed: int = None) -> dict:
    """BanditCAT V1 item selection (Sharpnack et al. 2024, Section 3.3)
    in place of simulate_det_responses' uniform-random draw, so the
    resulting response data has the same item-difficulty-tracks-ability
    selection bias the real DET has (see the AutoIRT paper's "Offline
    evaluation suffers..." paragraph) instead of assuming random
    administration.

    At each of the items_per_session rounds within a session:
      1. Draw n_theta_draws samples of theta from the session's current
         posterior (a discretized grid over DEFAULT_THETA_GRID, updated
         after every response within the session).
      2. For every item not yet administered this session, compute the
         3PL Fisher information (eq. 2) at each drawn theta, using a
         randomized discrimination -- Gamma(a_i / exposure_gamma,
         exposure_gamma), mean a_i -- so exposure isn't dominated by a
         handful of the most informative items (the paper's exposure
         control, eq. 7/8). Average over the theta draws.

         SIMPLIFICATION vs. the paper: for the 3PL case (Y/N Vocab,
         chance != 0) the paper randomizes the *height* of a
         Gaussian-kernel approximation to the Fisher information curve,
         obtained from (a, c, d) by moment matching -- the exact
         moment-matching formula isn't given in enough detail in the
         paper text available to reproduce here. This instead randomizes
         the discrimination parameter directly inside the exact 3PL
         Fisher information formula, which gives the same qualitative
         exposure-control effect (more randomization -> more even
         exposure) but is not the literal BanditCAT V1 formula for
         chance != 0 items. For ViC (chance = 0, i.e. genuinely 2PL),
         this matches the paper's method (eq. 7) exactly.
      3. Administer the item with the highest average randomized Fisher
         information.
      4. Observe the grade (scored against the item's TRUE parameters,
         not the randomized ones used only for selection) and do a
         Bayesian update of the posterior over the theta grid.

    random_injection_rate: with this probability at each round, skip the
    Fisher-information selection entirely and pick uniformly at random
    among items not yet administered this session instead. This is the
    standard real-world fix for the restricted-range/coverage problem
    documented in check_adaptive_selection.py -- periodically forcing a
    random (non-adaptive) item back into rotation restores both item
    coverage and per-item ability range, at the cost of diluting how
    strongly administration tracks ability. 0.0 (default) reproduces the
    original pure-adaptive behavior exactly.

    Item parameters used for selection are the TRUE simulated values,
    not a live-refit AutoIRT model -- refitting the full calibration
    model after every single response isn't something the real system
    does either (operational items are calibrated from history, then
    used as-is for administration between recalibration passes), so this
    treats the item bank as if its parameters were already known, which
    is the standard simplification for a simulation study like this one.

    Returns the same dict-of-arrays shape as simulate_det_responses, so
    everything downstream (the cold/jump/warm-start splitting logic,
    evaluate_calibration, etc.) works unchanged.

    IMPORTANT INTERPRETIVE CAVEAT: the AutoIRT paper itself warns that
    non-random administration inflates the item-grade correlation metric
    even when item parameters are constant across items ("Evaluating
    Calibrated Item Parameters" section: "the item mean grade correlation
    is typically positive even when the item parameters are constant for
    all items"), because the estimated ability used to compute predicted
    grades is itself derived from the same non-randomly-selected
    responses. So a higher item_grade_pearson_r after switching to this
    function is not on its own evidence of better calibration -- some or
    all of an improvement could be this known metric-inflation effect
    rather than a real gain, and that should be checked (e.g. by also
    looking at test_loss_nll, which isn't subject to the same bias in
    the same way) before reporting a change here as a calibration
    improvement.
    """
    rng = np.random.default_rng(random_seed)
    n_items = len(items["discrimination"])
    n_sessions = len(theta_by_session)

    discrimination = items["discrimination"]
    difficulty = items["difficulty"]
    chance = items["chance"]

    theta_grid = ADAPTIVE_SELECTION_THETA_GRID
    log_prior = -0.5 * (theta_grid / ABILITY_PRIOR_STD) ** 2

    session_id_list = []
    item_id_list = []
    grade_list = []
    day_list = []
    response_seq_list = []
    response_seq_counter = 0

    # This loop has no vectorized shortcut (each round's item choice depends
    # on the posterior update from the previous round), so at real DET scale
    # it can run silently for several minutes with no output at all before
    # the caller sees anything -- print periodic progress so a long run
    # doesn't look identical to a frozen one.
    progress_interval = max(1, n_sessions // 20)  # ~20 progress lines total
    start_time = time.time()

    for session in range(n_sessions):
        if session > 0 and session % progress_interval == 0:
            elapsed = time.time() - start_time
            fraction_done = session / n_sessions
            estimated_remaining = elapsed / fraction_done - elapsed
            print(f"    [simulate_det_responses_adaptive] {session}/{n_sessions} sessions "
                  f"({fraction_done:.0%}), {elapsed:.0f}s elapsed, "
                  f"~{estimated_remaining:.0f}s remaining", flush=True)

        eligible = np.ones(n_items, dtype=bool)
        log_posterior = log_prior.copy()

        for _round in range(items_per_session):
            posterior = np.exp(log_posterior - log_posterior.max())
            posterior /= posterior.sum()
            theta_draws = rng.choice(theta_grid, size=n_theta_draws, p=posterior)

            eligible_ids = np.where(eligible)[0]

            if random_injection_rate > 0 and rng.random() < random_injection_rate:
                chosen_item = int(rng.choice(eligible_ids))
            else:
                randomized_discrimination = rng.gamma(
                    shape=discrimination[eligible_ids] / exposure_gamma, scale=exposure_gamma,
                )
                info = _fisher_information_3pl(
                    theta_draws[:, None],
                    randomized_discrimination[None, :],
                    chance[eligible_ids][None, :],
                    difficulty[eligible_ids][None, :],
                ).mean(axis=0)
                chosen_item = int(eligible_ids[info.argmax()])

            eligible[chosen_item] = False

            true_probability = three_parameter_logistic(
                theta=theta_by_session[session], discrimination=discrimination[chosen_item],
                chance=chance[chosen_item], difficulty=difficulty[chosen_item],
            )
            grade = int(rng.binomial(n=1, p=true_probability))

            # Posterior update uses the item's TRUE parameters (scoring
            # should reflect the real item, not the randomized one used
            # only to control exposure during selection).
            p_grid = three_parameter_logistic(
                theta=theta_grid, discrimination=discrimination[chosen_item],
                chance=chance[chosen_item], difficulty=difficulty[chosen_item],
            )
            p_grid = np.clip(p_grid, 1e-6, 1 - 1e-6)
            log_likelihood = grade * np.log(p_grid) + (1 - grade) * np.log(1 - p_grid)
            log_posterior = log_posterior + log_likelihood

            session_id_list.append(session)
            item_id_list.append(chosen_item)
            grade_list.append(grade)
            day_list.append(session_day[session])
            response_seq_list.append(response_seq_counter)
            response_seq_counter += 1

    return {
        "session_id": np.array(session_id_list),
        "item_id": np.array(item_id_list),
        "grade": np.array(grade_list),
        "day": np.array(day_list),
        "response_seq": np.array(response_seq_list),
        "true_theta": theta_by_session[np.array(session_id_list)],
    }