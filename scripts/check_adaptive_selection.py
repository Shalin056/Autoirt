"""
check_adaptive_selection.py
============================

Fast sanity check for simulate_det_responses_adaptive -- NO model
fitting happens here, just item/session simulation, so this takes
seconds/minutes instead of the hours a full run_det_experiment.py pass
with DET_ITEM_SELECTION="adaptive" would take. Checks four things
before trusting a full run:

  1. Does selection actually track ability? For each response, compares
     the administered item's difficulty to the test-taker's TRUE theta.
     Adaptive selection should land much closer to |difficulty - theta|
     = 0 on average than the random baseline does.
  2. Is exposure reasonably spread out, or does a handful of items get
     picked constantly? Reports the most-administered item's share of
     all administrations, for both item types.
  3. Item coverage and per-item ability range: for items with at least
     MIN_EXPOSURES_FOR_RANGE_CHECK responses, what fraction of the item
     bank clears that bar, and how wide a spread of true theta values
     does each item's respondents actually cover? This is the
     restricted-range diagnostic -- adaptive selection concentrates each
     item's exposures near test-takers whose ability is close to that
     item's own difficulty (that's the point, for efficient scoring),
     but item CALIBRATION from a naive batch fit benefits from each item
     being seen across a WIDE range of abilities, not a narrow band near
     its own difficulty. A much narrower per-item theta spread and/or
     much lower coverage under "adaptive" than "random" is a real,
     literature-grounded mechanism (not a bug) for why adaptive
     selection could make item-grade Pearson AND test_loss_nll both get
     worse at once, which rules out the paper's separate warning about
     item-grade correlation being inflated by non-random administration
     (that artifact would move Pearson and test loss in OPPOSITE
     directions, not the same one).
  4. Runtime at a moderate scale, extrapolated to config.N_DET_SESSIONS,
     so there's a runtime estimate before committing to a real run.

Run with:
    python scripts/check_adaptive_selection.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from src.simulate_det import simulate_det_items, simulate_det_sessions, \
    simulate_det_responses, simulate_det_responses_adaptive

CHECK_N_SESSIONS = 300     # small on purpose -- this is a behavior check, not a timing benchmark at scale
RANGE_CHECK_N_SESSIONS = 3000  # needs to be bigger than CHECK_N_SESSIONS so enough items clear
                                # MIN_EXPOSURES_FOR_RANGE_CHECK to compute a meaningful per-item std
MIN_EXPOSURES_FOR_RANGE_CHECK = 5


def check_one_item_type(item_type_name, n_items, chance_value, items_per_session, random_seed):
    print(f"\n=== {item_type_name}: n_items={n_items}, chance={chance_value}, "
          f"{CHECK_N_SESSIONS} sessions ===")
    items = simulate_det_items(n_items=n_items, chance_value=chance_value, random_seed=random_seed)
    sessions = simulate_det_sessions(n_sessions=CHECK_N_SESSIONS, random_seed=random_seed + 1)
    theta_key = "yn_theta" if item_type_name == "Y/N Vocab" else "vic_theta"
    theta = sessions[theta_key]

    random_responses = simulate_det_responses(
        items, theta, sessions["session_day"], items_per_session, random_seed=random_seed + 2,
    )

    start_time = time.time()
    adaptive_responses = simulate_det_responses_adaptive(
        items, theta, sessions["session_day"], items_per_session, random_seed=random_seed + 2,
    )
    elapsed_seconds = time.time() - start_time

    for label, responses in [("random   ", random_responses), ("adaptive ", adaptive_responses)]:
        item_difficulty = items["difficulty"][responses["item_id"]]
        session_theta = theta[responses["session_id"]]
        mean_abs_gap = np.mean(np.abs(item_difficulty - session_theta))

        item_id, counts = np.unique(responses["item_id"], return_counts=True)
        max_exposure_share = counts.max() / len(responses["item_id"])

        print(f"  {label}: mean |difficulty - true theta| = {mean_abs_gap:.3f}   "
              f"max single-item exposure share = {max_exposure_share:.1%}")

    n_sessions_full = getattr(config, "N_DET_SESSIONS", None)
    if n_sessions_full:
        estimated_full_seconds = elapsed_seconds * (n_sessions_full / CHECK_N_SESSIONS)
        print(f"  adaptive selection took {elapsed_seconds:.1f}s for {CHECK_N_SESSIONS} sessions -> "
              f"est. {estimated_full_seconds / 60:.1f} min for config.N_DET_SESSIONS="
              f"{n_sessions_full} (item/session generation only, not model fitting)")


def _per_item_theta_std(responses, theta, min_exposures):
    """For every item exposed at least min_exposures times, the std of
    true theta among the sessions that got it -- a narrow std means
    that item was almost always administered to test-takers at nearly
    the same ability level."""
    stds = []
    item_ids, counts = np.unique(responses["item_id"], return_counts=True)
    for item_id, count in zip(item_ids, counts):
        if count >= min_exposures:
            mask = responses["item_id"] == item_id
            stds.append(theta[responses["session_id"][mask]].std())
    return np.array(stds)


INJECTION_RATES_TO_SWEEP = [0.0, 0.2, 0.4]  # 0.0 = pure adaptive (original behavior)


def check_restricted_range(item_type_name, n_items, chance_value, items_per_session, random_seed):
    print(f"\n=== {item_type_name} item coverage / per-item ability range "
          f"({RANGE_CHECK_N_SESSIONS} sessions) ===")
    items = simulate_det_items(n_items=n_items, chance_value=chance_value, random_seed=random_seed)
    sessions = simulate_det_sessions(n_sessions=RANGE_CHECK_N_SESSIONS, random_seed=random_seed + 1)
    theta_key = "yn_theta" if item_type_name == "Y/N Vocab" else "vic_theta"
    theta = sessions[theta_key]

    random_responses = simulate_det_responses(
        items, theta, sessions["session_day"], items_per_session, random_seed=random_seed + 2,
    )
    stds = _per_item_theta_std(random_responses, theta, MIN_EXPOSURES_FOR_RANGE_CHECK)
    coverage = len(stds) / n_items
    print(f"  random             : {len(stds)}/{n_items} items ({coverage:.0%}) reached "
          f"{MIN_EXPOSURES_FOR_RANGE_CHECK}+ exposures   "
          f"median per-item theta std = {np.median(stds):.3f}" if len(stds) else
          f"  random             : 0/{n_items} items reached {MIN_EXPOSURES_FOR_RANGE_CHECK}+ exposures")

    for injection_rate in INJECTION_RATES_TO_SWEEP:
        adaptive_responses = simulate_det_responses_adaptive(
            items, theta, sessions["session_day"], items_per_session,
            random_injection_rate=injection_rate, random_seed=random_seed + 2,
        )
        stds = _per_item_theta_std(adaptive_responses, theta, MIN_EXPOSURES_FOR_RANGE_CHECK)
        coverage = len(stds) / n_items
        label = f"adaptive (inject={injection_rate:.0%})"
        print(f"  {label:<19}: {len(stds)}/{n_items} items ({coverage:.0%}) reached "
              f"{MIN_EXPOSURES_FOR_RANGE_CHECK}+ exposures   "
              f"median per-item theta std = {np.median(stds):.3f}" if len(stds) else
              f"  {label:<19}: 0/{n_items} items reached {MIN_EXPOSURES_FOR_RANGE_CHECK}+ exposures")


def main():
    check_one_item_type("Y/N Vocab", config.N_YN_VOCAB_ITEMS, config.YN_CHANCE,
                         config.YN_ITEMS_PER_SESSION, config.DET_RANDOM_SEED)
    check_one_item_type("ViC", config.N_VIC_ITEMS, config.VIC_CHANCE,
                         config.VIC_ITEMS_PER_SESSION, config.DET_RANDOM_SEED + 100)

    print("\nWhat to look for:")
    print("- 'adaptive' mean |difficulty - true theta| should be clearly lower than 'random's --")
    print("  if it isn't, the selection isn't actually tracking ability and something's wrong")
    print("  before spending hours on a full run.")
    print("- max single-item exposure share should stay well under 100% for 'adaptive' --")
    print("  a share near 1/items_per_session or higher means exposure control (exposure_gamma)")
    print("  needs to be tightened (lower exposure_gamma = more randomization).")

    check_restricted_range("Y/N Vocab", config.N_YN_VOCAB_ITEMS, config.YN_CHANCE,
                            config.YN_ITEMS_PER_SESSION, config.DET_RANDOM_SEED)
    check_restricted_range("ViC", config.N_VIC_ITEMS, config.VIC_CHANCE,
                            config.VIC_ITEMS_PER_SESSION, config.DET_RANDOM_SEED + 100)

    print("\nWhat to look for:")
    print("- Lower coverage % and/or lower median per-item theta std under 'adaptive' than")
    print("  'random' is the restricted-range/coverage mechanism: item calibration from a naive")
    print("  batch fit needs each item seen across a WIDE ability range, which adaptive selection")
    print("  actively works against by design. This is a real, literature-grounded reason")
    print("  calibration quality could get worse under adaptive selection, distinct from (and in")
    print("  addition to) the paper's separate warning about item-grade correlation being")
    print("  inflated by non-random administration -- that inflation effect would move Pearson up")
    print("  while test_loss_nll stays flat; a genuine restricted-range/coverage problem moves")
    print("  BOTH metrics in the same (worse) direction, which is what the actual DET run showed.")
    print("- As random_injection_rate increases from 0%, coverage and median theta std should")
    print("  climb back toward random's numbers -- that's the tradeoff to pick a value from:")
    print("  higher injection recovers more coverage/range (closer to random's calibration")
    print("  behavior) but dilutes how strongly administration actually tracks ability (closer")
    print("  to random's item-selection behavior too, defeating the point of doing this at all")
    print("  if pushed too high). There is no injection rate that gets both for free.")


if __name__ == "__main__":
    main()