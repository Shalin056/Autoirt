"""
simulate.py
===========

This module creates FAKE (synthetic) test data that behaves exactly like
data from a real adaptive test, following the "3PL Item Response Theory"
model described in the Supplementary Material of the AutoIRT paper.

Why do we need fake data at all?
---------------------------------
Before testing a calibration method (like AutoIRT) on real student
responses, it is standard practice to first test it on data where we
KNOW the true answer. That way we can check: "did the method recover the
true item difficulty and true student ability, or not?" This is called
a simulation study.

Background concepts (explained simply)
----------------------------------------
- Each simulated "item" is a test question. Every item has three hidden
  properties:
    a  (discrimination) -- how sharply this item separates weak students
                            from strong students. Higher = sharper.
    d  (difficulty)     -- how hard the item is. Higher = harder.
    c  (chance/guessing)-- probability of guessing correctly even with
                            zero ability. Fixed at 0.25 here (like a
                            4-option multiple-choice question).

- Each simulated "test-taker" (a person taking the test) has a hidden
  ability level, called theta. Higher theta = smarter/more skilled.

- The probability a test-taker gets an item correct depends on both
  their theta and the item's (a, c, d), following the "3PL formula":

      P(correct) = c + (1 - c) * sigmoid(a * (theta - d))

  In words: if theta is much bigger than d, this probability approaches
  1 (easy for them). If theta is much smaller than d, it approaches c
  (they are basically guessing).

All of the specific formulas below (for generating a, d, theta, etc.)
are taken directly from the paper's Supplementary Material so that our
simulated data matches theirs.
"""

import numpy as np


def simulate_items(n_items: int, effect_noise_std: float = 0.1, random_seed: int = None) -> dict:
    """
    Create a fake bank of test items with known difficulty (d) and
    discrimination (a) parameters.

    How the item parameters are generated (from the paper's supplement):
      1. Each item gets two random numeric "features" X1 and X2, drawn
         uniformly between -10 and 10. Think of these as standing in for
         real item features like word frequency or sentence length.
      2. We compute Z = X1 - X2.
      3. The item's "typical" difficulty and discrimination are complex,
         wavy functions of Z (using sine and cosine), which makes the
         relationship between item features and item parameters
         realistically non-linear and hard to guess by eye.
      4. A little bit of random noise is added on top, so that items with
         identical Z are not perfectly identical to each other.
      5. The chance parameter (c) is fixed at 0.25 for every item, meaning
         every item behaves like a 4-option multiple-choice question.

    Parameters
    ----------
    n_items : int
        How many test items (questions) to create.
    effect_noise_std : float
        How much random noise to add on top of the "typical" difficulty
        and discrimination (paper uses 0.1).
    random_seed : int, optional
        Fix this to get the exact same fake items every time you run the
        code (useful for reproducibility / debugging).

    Returns
    -------
    dict with keys:
        "feature_1", "feature_2" : the raw item features (X1, X2)
        "discrimination" (a)     : how sharply each item separates ability levels
        "difficulty" (d)         : how hard each item is
        "chance" (c)             : guessing probability (always 0.25 here)
    """
    rng = np.random.default_rng(random_seed)

    feature_1 = rng.uniform(-10, 10, size=n_items)
    feature_2 = rng.uniform(-10, 10, size=n_items)
    z = feature_1 - feature_2

    # Guard against a division-by-zero if z happens to land exactly on 0.
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)

    typical_difficulty = 4 * np.sin(z_safe) / np.abs(z_safe)
    typical_discrimination = 0.5 * np.cos(0.1 * z ** 2) + 1.0

    difficulty_noise = rng.normal(0, effect_noise_std, size=n_items)
    discrimination_noise = rng.normal(0, effect_noise_std, size=n_items)

    difficulty = typical_difficulty + difficulty_noise
    # Discrimination must stay positive, so we add noise in log-space
    # and exponentiate back (this is what the paper does too).
    discrimination = np.exp(np.log(typical_discrimination) + discrimination_noise)

    chance = np.full(n_items, 0.25)

    return {
        "feature_1": feature_1,
        "feature_2": feature_2,
        "discrimination": discrimination,
        "difficulty": difficulty,
        "chance": chance,
    }


def simulate_test_taker_abilities(n_sessions: int, random_seed: int = None) -> np.ndarray:
    """
    Create fake ability levels (theta) for a group of test-takers.

    Each "session" here represents one test-taker taking the test once.
    Abilities are drawn from a Normal distribution centered at 0 with a
    standard deviation of sqrt(2.5), exactly as specified in the paper.

    Parameters
    ----------
    n_sessions : int
        How many test-taking sessions (i.e. students) to simulate.
    random_seed : int, optional
        Fix this for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_sessions,)
        The true (hidden, but known to us because we generated it) ability
        of each simulated test-taker.
    """
    rng = np.random.default_rng(random_seed)
    return rng.normal(loc=0.0, scale=np.sqrt(2.5), size=n_sessions)


def three_parameter_logistic(theta, discrimination, chance, difficulty):
    """
    The core 3PL Item Response Theory formula.

    Computes the probability that a test-taker with ability `theta`
    answers an item with the given (discrimination, chance, difficulty)
    correctly.

    This function is written to work with plain numbers OR with numpy
    arrays (so it can score one student on one item, or many students on
    many items at once, whichever shape you pass in).
    """
    return chance + (1 - chance) * (1 / (1 + np.exp(-discrimination * (theta - difficulty))))


def simulate_test_responses(items: dict, abilities: np.ndarray,
                             items_per_session: int = 10, random_seed: int = None) -> dict:
    """
    Simulate an entire dataset of test responses: for every test-taker
    (session), randomly pick a handful of items to administer, then flip
    a biased coin (based on the 3PL formula) to decide correct or wrong.

    Parameters
    ----------
    items : dict
        Output of `simulate_items(...)`.
    abilities : np.ndarray
        Output of `simulate_test_taker_abilities(...)`.
    items_per_session : int
        How many items each test-taker answers (paper uses 10).
    random_seed : int, optional
        Fix this for reproducibility.

    Returns
    -------
    dict with keys:
        "session_id" : which test-taker this response belongs to
        "item_id"    : which item was answered
        "grade"      : 1 if correct, 0 if incorrect
        "true_theta" : the test-taker's true ability (only used later for
                        checking how well our method recovered it --
                        in a REAL experiment, we would never know this!)
    """
    rng = np.random.default_rng(random_seed)
    n_items = len(items["discrimination"])
    n_sessions = len(abilities)

    session_id_list = []
    item_id_list = []
    grade_list = []

    for session in range(n_sessions):
        # Pick `items_per_session` distinct items for this test-taker.
        administered_items = rng.choice(n_items, size=items_per_session, replace=False)

        probability_correct = three_parameter_logistic(
            theta=abilities[session],
            discrimination=items["discrimination"][administered_items],
            chance=items["chance"][administered_items],
            difficulty=items["difficulty"][administered_items],
        )

        # Flip a biased coin for each item to decide correct (1) or wrong (0).
        grades = rng.binomial(n=1, p=probability_correct)

        session_id_list.extend([session] * items_per_session)
        item_id_list.extend(administered_items.tolist())
        grade_list.extend(grades.tolist())

    session_id = np.array(session_id_list)
    item_id = np.array(item_id_list)
    grade = np.array(grade_list)

    return {
        "session_id": session_id,
        "item_id": item_id,
        "grade": grade,
        "true_theta": abilities[session_id],
    }


if __name__ == "__main__":
    # A quick smoke test you can run directly with:
    #   python -m src.simulate
    # to sanity-check that everything above works before using it elsewhere.
    print("Running a quick smoke test of simulate.py ...")

    items = simulate_items(n_items=100, random_seed=0)
    abilities = simulate_test_taker_abilities(n_sessions=2000, random_seed=1)
    responses = simulate_test_responses(items, abilities, items_per_session=10, random_seed=2)

    print(f"Discrimination (a): mean={items['discrimination'].mean():.3f}, "
          f"std={items['discrimination'].std():.3f}")
    print(f"Difficulty (d):     mean={items['difficulty'].mean():.3f}, "
          f"std={items['difficulty'].std():.3f}")
    print(f"Total responses generated: {len(responses['grade'])}")
    print(f"Overall accuracy across all responses: {responses['grade'].mean():.3f}")
    print("Smoke test passed: no errors were raised.")
