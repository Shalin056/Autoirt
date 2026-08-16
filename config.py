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

# --- AutoML backend ---
# "ensemble" = hand-built RandomForest/XGBoost/LightGBM stack.
# "autogluon" = AutoGluon-tabular, matching the paper's own backend.
# See run_backend_comparision.py for a script that fits both backends on
# identical data and compares them directly.
BACKEND = "autogluon"

# --- Reproducibility ---
RANDOM_SEED = 42  # change this to get a different random simulation

# --- Repeated runs (for measuring how much results vary by chance) ---
N_REPEATS = 5  # set to e.g. 5 to run the whole experiment 5 times with
                # different seeds and report the average +/- spread instead
                # of a single (noisier) run

# --- DET-phase (Phase 2) settings ---
# Two-item-type setup matching the paper's DET analysis, using simulated
# item content since real Duolingo data isn't available. Real DET has
# 3290 Y/N Vocab items and 585 ViC items; these are set to roughly half
# that. AutoML fits get more expensive with more items/rows, so start
# with one item type if a full run is too slow.
N_YN_VOCAB_ITEMS = 3290
N_VIC_ITEMS = 585

# N_DET_SESSIONS needs to be large enough that pilot items actually
# accumulate more than R post-split responses for the largest Jump value
# (R=80) -- otherwise "first R responses per pilot item" is silently
# capped by how much data exists rather than by R, and the Jump
# conditions stop being distinguishable from each other. Roughly:
#   n_sessions ~= target_responses_per_item * n_items
#                 / (items_per_session * post_split_day_fraction)
# use check_jump_start_data_volume.py to confirm a given value here
# produces distinct train-set sizes across Jump 20/40/80 before running
# the full (multi-hour) calibration in run_det_experiment.py.
# This value is for DET_ITEM_SELECTION="random" specifically -- every
# random-mode result checked and reported so far was generated at 25000.
N_DET_SESSIONS = 25000

# Separate session count for DET_ITEM_SELECTION="adaptive". Exposure is
# non-uniform under adaptive selection, so the random-mode N_DET_SESSIONS
# above doesn't carry over -- check_jump_start_data_volume.py found 25000
# still left Y/N Vocab Jump 40 and Jump 80 (and ViC Jump 80) data-starved
# even with DET_ADAPTIVE_RANDOM_INJECTION_RATE=0.4, and 50000 clears
# everything except Y/N Vocab Jump 80 (13% -> 42% coverage, real
# improvement but still below the 80% threshold). Not chasing this
# further with even more sessions -- Y/N Vocab Jump 80 was the one
# condition that stayed borderline under random selection too, and the
# same approach applies here: report it with the coverage caveat attached
# rather than spending more compute to fully clear one sub-condition.
N_DET_SESSIONS_ADAPTIVE = 50000

# Hard cap on EM steps for the DET run specifically. Each step is a full
# model fit across ~1200 items, so this bounds worst-case runtime per
# condition. Lower this (e.g. to 8-10) if a run is taking too long --
# results may be slightly less converged, worth noting in the report,
# but it puts a ceiling on how long you're waiting.
DET_MAX_EM_STEPS = 20

YN_ITEMS_PER_SESSION = 18   # matches the paper
VIC_ITEMS_PER_SESSION = 9   # matches the paper
YN_CHANCE = 0.25            # paper: c=0.25 for Y/N Vocab (guessable, multiple choice)
VIC_CHANCE = 0.0            # paper: c=0 for ViC (not really guessable)
JUMP_START_R_VALUES = [20, 40, 80]  # matches the paper's Jump 20/40/80
DET_RANDOM_SEED = 42

# Runtime override for run_det_experiment.py specifically. 12 conditions x
# up to 20 EM steps x AutoGluon's 60s/fit is multi-hour at N_DET_SESSIONS
# =25000. Set to "ensemble" to iterate quickly (minutes, not hours), or
# to None to fall back to config.BACKEND for a full overnight run.
DET_BACKEND_OVERRIDE = "ensemble"

# How items are chosen for each session in the DET-phase simulation.
# "random" = simulate_det_responses' uniform random draw (the default
#   used for every DET run so far).
# "adaptive" = simulate_det_responses_adaptive's BanditCAT V1-style
#   selection (item difficulty tracks the session's current ability
#   estimate, preference for high discrimination, randomized for
#   exposure control) -- see simulate_det.py for the exact mechanism
#   and an important caveat about how this affects the item-grade
#   correlation metric.
DET_ITEM_SELECTION = "random"

# Only used when DET_ITEM_SELECTION="adaptive". check_adaptive_selection.py
# found pure adaptive selection (0.0 here) leaves a large share of the item
# bank under-covered and narrows the ability range each item is seen across
# (62%/1.07 for Y/N Vocab, 82%/1.16 for ViC vs. random's ~100%/1.5+), which
# is a real, literature-grounded reason item calibration can get WORSE under
# adaptive administration, not better. This fraction of rounds fall back to
# picking uniformly at random among not-yet-administered items instead of
# the Fisher-information selection, to recover coverage/range at the cost of
# diluting how strongly administration tracks ability -- run
# check_adaptive_selection.py's coverage sweep to pick a value before
# committing to a full run.
DET_ADAPTIVE_RANDOM_INJECTION_RATE = 0.2