"""
config.py
=========

Central settings for the experiment scripts. Edit values here rather than in the scripts themselves.
"""

# --- Size of the simulated data ---
N_ITEMS = 400          # total number of test items in the item bank
N_SESSIONS = 20000     # total number of simulated test-takers
ITEMS_PER_SESSION = 10  # how many items each test-taker answers

# --- Cold-start experiment settings ---
N_COLD_START_HOLDOUT_ITEMS = 100  # items with zero training responses,
                                   # used to test calibration on new items

# --- Warm-start experiment settings ---
WARM_START_TEST_FRACTION = 0.2   # fraction of sessions held out for testing
                                   # (same items, unseen test-takers)

# --- AutoIRT algorithm settings ---
N_EM_STEPS = 4  # minimum number of Monte Carlo EM iterations (paper uses 4;
                # run_autoirt_calibration treats this as a floor, not a fixed count)

# --- Reproducibility ---
RANDOM_SEED = 42  # change this to get a different random simulation

# --- Repeated runs (for measuring how much results vary by chance) ---
N_REPEATS = 5  # set to e.g. 5 to run the whole experiment 5 times with
                # different seeds and report the average +/- spread instead
                # of a single (noisier) run

# --- DET-phase (Phase 2) settings ---
# Two-item-type setup matching the paper's DET analysis, using simulated
# item content since we don't have real Duolingo data yet. Real DET has
# 3290 Y/N Vocab items and 585 ViC items, scaled down here so a first
# run finishes in a reasonable time (12 full calibration runs total: 6
# split conditions x 2 item types, each running to convergence). Bump
# these up once a small run works.
N_YN_VOCAB_ITEMS = 150
N_VIC_ITEMS = 80
N_DET_SESSIONS = 1500
YN_ITEMS_PER_SESSION = 18   # matches the paper
VIC_ITEMS_PER_SESSION = 9   # matches the paper
YN_CHANCE = 0.25            # paper: c=0.25 for Y/N Vocab (guessable, multiple choice)
VIC_CHANCE = 0.0            # paper: c=0 for ViC (not really guessable)
JUMP_START_R_VALUES = [20, 40, 80]  # matches the paper's Jump 20/40/80
DET_RANDOM_SEED = 42