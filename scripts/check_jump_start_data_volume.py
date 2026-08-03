"""
check_jump_start_data_volume.py
================================

Fast check for Jump 20/40/80 data starvation -- NO model fitting happens
here, just item/session simulation and the same split_jump_start()
logic used in run_det_experiment.py. This answers "does the split logic
actually produce different training sets for R=20/40/80 at this
N_DET_SESSIONS" without waiting hours for AutoML or ensemble fits, which
don't affect this question at all.

Run with:
    python scripts/check_jump_start_data_volume.py

Takes a few seconds. If it reports "OK" for a given N_DET_SESSIONS,
THEN it's worth spending the hours on run_det_experiment.py at that
session count -- not before.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.simulate_det import simulate_det_items, simulate_det_sessions, simulate_det_responses, \
    COLD_JUMP_SPLIT_DAY
from run_det_experiment import choose_pilot_and_operational_items, split_jump_start


def check(item_type_name, n_items, chance_value, items_per_session, n_sessions, random_seed):
    print(f"\n=== {item_type_name}: n_items={n_items}, n_sessions={n_sessions} ===")
    items = simulate_det_items(n_items=n_items, chance_value=chance_value, random_seed=random_seed)
    sessions = simulate_det_sessions(n_sessions=n_sessions, random_seed=random_seed + 1)
    theta = sessions["yn_theta"] if item_type_name == "Y/N Vocab" else sessions["vic_theta"]
    all_responses = simulate_det_responses(
        items, theta, sessions["session_day"], items_per_session, random_seed=random_seed + 2,
    )
    pilot_item_ids, operational_item_ids = choose_pilot_and_operational_items(
        n_items, random_seed=random_seed + 3,
    )

    for r in config.JUMP_START_R_VALUES:
        train, _ = split_jump_start(
            all_responses, operational_item_ids, pilot_item_ids, COLD_JUMP_SPLIT_DAY, r,
        )
        print(f"  Jump {r:>3}: train n={len(train['grade'])}")


def main():
    for n_sessions in [1500, 6000, 25000]:
        check("Y/N Vocab", config.N_YN_VOCAB_ITEMS, config.YN_CHANCE,
              config.YN_ITEMS_PER_SESSION, n_sessions, config.DET_RANDOM_SEED)
        check("ViC", config.N_VIC_ITEMS, config.VIC_CHANCE,
              config.VIC_ITEMS_PER_SESSION, n_sessions, config.DET_RANDOM_SEED + 100)
    print("\nLook for Jump 20 / 40 / 80 train n to actually differ from each other.")
    print("If Jump 40 and Jump 80 still match at a given n_sessions, that session")
    print("count is still too low for that item bank size -- see the diagnostic")
    print("warnings above (printed by split_jump_start) for exact numbers.")


if __name__ == "__main__":
    main()