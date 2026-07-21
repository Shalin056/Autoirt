"""
config.py
=========

All the adjustable settings for the experiment live here, in one place,
so you don't have to hunt through the code to change something.

To run a bigger or smaller experiment, just edit the numbers below and
re-run `scripts/run_experiment.py`.
"""

# --- Size of the simulated data ---
N_ITEMS = 400          # total number of test items in the item bank
N_SESSIONS = 20000     # total number of simulated test-takers
ITEMS_PER_SESSION = 10  # how many items each test-taker answers

# --- Cold-start experiment settings ---
N_COLD_START_HOLDOUT_ITEMS = 100  # items with ZERO training responses,
                                   # used to test calibration on brand-new items

# --- Warm-start experiment settings ---
WARM_START_TEST_FRACTION = 0.2   # fraction of SESSIONS held out for testing
                                   # (same items, just unseen test-takers)

# --- AutoIRT algorithm settings ---
N_EM_STEPS = 4  # number of Monte Carlo EM iterations (paper uses 4)

# --- Reproducibility ---
RANDOM_SEED = 42  # change this to get a different random simulation

# --- Repeated runs (for measuring how much results vary by chance) ---
N_REPEATS = 5  # set to e.g. 5 to run the whole experiment 5 times with
                # different seeds and report the average +/- spread instead
                # of a single (noisier) run
