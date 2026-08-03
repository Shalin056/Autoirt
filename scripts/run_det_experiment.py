"""
Phase 2: replicates the "Offline calibration analysis for DET" section
of the AutoIRT paper, as opposed to run_experiment.py which replicates
the more abstract simulation study from the supplement.

The paper models two item types together, each with its own ability
parameter, under three conditions:

  1. Cold start: split by date, hold out 50% of items as "pilot" items.
     Train on pre-split responses to operational items only. Test on
     ALL post-split responses (pilot and operational both).
  2. Jump start (R = 20, 40, 80): same item/date split as cold-start,
     but training also gets the first R post-split responses per pilot
     item, in chronological order. Test is whatever's left over.
  3. Warm start: no item holdout at all, just split responses by date
     (two different dates, 2024-05-08 and 2024-05-15 in the paper).

Real response timestamps aren't available, so each simulated session
gets a random day over a 79-day window standing in for the real
2024-04-01 to 2024-06-18 window, with split days picked to match the
paper's actual dates (see simulate_det.py for the exact numbers).

Run with:
    python scripts/run_det_experiment.py

Settings are in config.py. This is heavier than run_experiment.py:
6 split conditions x 2 item types = 12 full calibration runs, each
iterating to convergence instead of a fixed step count, so start small
before scaling up the item/session counts.

Item bank sizes (config.N_YN_VOCAB_ITEMS / config.N_VIC_ITEMS) and the
AutoML backend (config.BACKEND) are set in config.py -- larger item
banks and the AutoGluon backend both add substantially to runtime
compared to a small ensemble-backend run.
"""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from src.simulate_det import simulate_det_items, simulate_det_sessions, simulate_det_responses, \
    COLD_JUMP_SPLIT_DAY, WARM_SPLIT_DAY_1, WARM_SPLIT_DAY_2
from src.autoirt_model import run_autoirt_calibration
from src.evaluate import evaluate_calibration


def select_subset(responses: dict, mask: np.ndarray) -> dict:
    """Rows of `responses` where mask is True."""
    return {key: values[mask] for key, values in responses.items()}


def choose_pilot_and_operational_items(n_items: int, random_seed: int):
    """Splits the item bank 50/50 into pilot (held out for testing
    cold/jump-start) and operational (always has training data) items,
    matching the paper's 50-50 pilot/operational split."""
    rng = np.random.default_rng(random_seed)
    n_pilot = n_items // 2
    pilot_item_ids = rng.choice(n_items, size=n_pilot, replace=False)
    is_pilot = np.zeros(n_items, dtype=bool)
    is_pilot[pilot_item_ids] = True
    operational_item_ids = np.where(~is_pilot)[0]
    return pilot_item_ids, operational_item_ids


def split_cold_start(responses: dict, operational_item_ids: np.ndarray, split_day: int):
    """Train on pre-split operational-item responses. Test on everything
    post-split, pilot and operational both."""
    is_operational = np.isin(responses["item_id"], operational_item_ids)
    is_pre_split = responses["day"] < split_day

    train = select_subset(responses, is_operational & is_pre_split)
    test = select_subset(responses, ~is_pre_split)
    return train, test


def split_jump_start(responses: dict, operational_item_ids: np.ndarray,
                      pilot_item_ids: np.ndarray, split_day: int, n_responses_per_pilot_item: int):
    """Same as cold-start, but training also gets the first R responses
    per pilot item after the split date (ordered by day, then response_seq
    to break same-day ties). Test is whatever's left over post-split.

    Also checks whether pilot items actually had R responses available
    post-split. If most items are data-starved (fewer than R responses
    exist at all), the "Jump R" condition silently collapses into "use
    whatever's available" rather than testing R specifically, which can
    make different R values produce identical training sets and
    therefore identical metrics. Printing this diagnostic surfaces that
    immediately instead of it only showing up later as suspiciously
    identical metrics.
    """
    is_operational = np.isin(responses["item_id"], operational_item_ids)
    is_pre_split = responses["day"] < split_day
    is_post_split = ~is_pre_split
    is_pilot = np.isin(responses["item_id"], pilot_item_ids)

    operational_pre_mask = is_operational & is_pre_split

    pilot_post_idx = np.where(is_pilot & is_post_split)[0]
    chronological_order = np.lexsort((
        responses["response_seq"][pilot_post_idx],
        responses["day"][pilot_post_idx],
    ))
    pilot_post_idx_sorted = pilot_post_idx[chronological_order]

    seen_count_by_item = {}
    keep_pilot_idx = []
    for idx in pilot_post_idx_sorted:
        item_id = responses["item_id"][idx]
        count_so_far = seen_count_by_item.get(item_id, 0)
        if count_so_far < n_responses_per_pilot_item:
            keep_pilot_idx.append(idx)
            seen_count_by_item[item_id] = count_so_far + 1
    keep_pilot_idx = np.array(keep_pilot_idx, dtype=int)

    # Diagnostic: what fraction of pilot items actually reached R?
    n_pilot_items = len(pilot_item_ids)
    n_items_reaching_r = sum(1 for count in seen_count_by_item.values()
                              if count >= n_responses_per_pilot_item)
    n_items_with_any_data = len(seen_count_by_item)
    fraction_reaching_r = n_items_reaching_r / n_pilot_items if n_pilot_items else 0.0
    if fraction_reaching_r < 0.8:
        print(f"    [DIAGNOSTIC] Jump {n_responses_per_pilot_item}: only "
              f"{n_items_reaching_r}/{n_pilot_items} pilot items "
              f"({fraction_reaching_r:.0%}) actually had {n_responses_per_pilot_item} "
              f"post-split responses available ({n_items_with_any_data}/{n_pilot_items} "
              f"had any data at all). This condition is likely data-starved -- "
              f"increase N_DET_SESSIONS or treat this result as not meaningfully "
              f"testing R={n_responses_per_pilot_item} before reporting it.")

    train_mask = operational_pre_mask.copy()
    train_mask[keep_pilot_idx] = True

    test_mask = is_post_split.copy()
    test_mask[keep_pilot_idx] = False  # these went into training, not testing

    return select_subset(responses, train_mask), select_subset(responses, test_mask)


def split_warm_start(responses: dict, split_day: int):
    """No item holdout, just a date split."""
    is_pre_split = responses["day"] < split_day
    return select_subset(responses, is_pre_split), select_subset(responses, ~is_pre_split)


RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "det_offline_analysis.csv")
RESULT_FIELDNAMES = ["item_type", "split", "n_steps_run", "stopped_reason", "elapsed_seconds",
                      "test_loss_nll", "item_grade_pearson_r", "item_grade_spearman_r",
                      "ability_recovery_pearson_r"]


def append_result_row(row: dict):
    """Writes one condition's result immediately, instead of waiting for
    everything to finish. This is the difference between losing 3 hours
    of work if the run is killed/crashes at condition 11 of 12, and
    already having 11 of 12 rows safely on disk. Also means you can
    `tail -f` or just re-open the CSV mid-run to see live progress."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_one_item_type(item_type_name: str, n_items: int, chance_value: float,
                       items_per_session: int, n_sessions: int, random_seed: int) -> list:
    """Runs all 6 split conditions for one item type, saving each result to
    CSV as soon as it's ready, and returns the list of result rows."""
    print(f"\n{'#' * 70}")
    print(f"# ITEM TYPE: {item_type_name}  (n_items={n_items}, chance={chance_value})")
    print(f"{'#' * 70}")

    items = simulate_det_items(n_items=n_items, chance_value=chance_value, random_seed=random_seed)
    sessions = simulate_det_sessions(n_sessions=n_sessions, random_seed=random_seed + 1)
    theta_key = "yn_theta" if item_type_name == "Y/N Vocab" else "vic_theta"
    theta = sessions[theta_key]
    all_responses = simulate_det_responses(
        items, theta, sessions["session_day"], items_per_session, random_seed=random_seed + 2,
    )

    pilot_item_ids, operational_item_ids = choose_pilot_and_operational_items(
        n_items, random_seed=random_seed + 3,
    )

    conditions = []
    train, test = split_cold_start(all_responses, operational_item_ids, COLD_JUMP_SPLIT_DAY)
    conditions.append(("Cold", train, test))
    for r in config.JUMP_START_R_VALUES:
        train, test = split_jump_start(all_responses, operational_item_ids, pilot_item_ids,
                                        COLD_JUMP_SPLIT_DAY, r)
        conditions.append((f"Jump {r}", train, test))
    train, test = split_warm_start(all_responses, WARM_SPLIT_DAY_1)
    conditions.append(("Warm 05-08", train, test))
    train, test = split_warm_start(all_responses, WARM_SPLIT_DAY_2)
    conditions.append(("Warm 05-15", train, test))

    results = []
    max_em_steps = getattr(config, "DET_MAX_EM_STEPS", 20)
    for condition_index, (split_name, train_responses, test_responses) in enumerate(conditions, start=1):
        print(f"\n--- [{condition_index}/{len(conditions)}] {item_type_name} / {split_name} "
              f"(train n={len(train_responses['grade'])}, test n={len(test_responses['grade'])}) ---")
        start_time = time.time()
        calibration_result = run_autoirt_calibration(
            train_responses, items, n_items=n_items, random_seed=random_seed,
            backend=(config.DET_BACKEND_OVERRIDE or config.BACKEND),
            max_em_steps=max_em_steps,
        )
        elapsed_seconds = time.time() - start_time
        metrics = evaluate_calibration(
            test_responses, theta,
            calibration_result["discrimination"], calibration_result["difficulty"],
            n_items=n_items,
        )
        print(f"{item_type_name} / {split_name} metrics ({elapsed_seconds:.0f}s):", metrics)
        row = {
            "item_type": item_type_name,
            "split": split_name,
            "n_steps_run": calibration_result["n_steps_run"],
            "stopped_reason": calibration_result["stopped_reason"],
            "elapsed_seconds": round(elapsed_seconds, 1),
            **metrics,
        }
        append_result_row(row)
        results.append(row)

    return results


def main():
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)  # start this run's CSV fresh; each row gets appended as it finishes

    run_start_time = time.time()
    all_results = []
    all_results += run_one_item_type(
        "Y/N Vocab", config.N_YN_VOCAB_ITEMS, config.YN_CHANCE,
        config.YN_ITEMS_PER_SESSION, config.N_DET_SESSIONS, config.DET_RANDOM_SEED,
    )
    all_results += run_one_item_type(
        "ViC", config.N_VIC_ITEMS, config.VIC_CHANCE,
        config.VIC_ITEMS_PER_SESSION, config.N_DET_SESSIONS, config.DET_RANDOM_SEED + 100,
    )
    total_elapsed_minutes = (time.time() - run_start_time) / 60

    print("\n" + "=" * 70)
    print("SUMMARY (compare to the paper's Table 1 / Table 2 pattern)")
    print("=" * 70)
    header = f"{'Type':<10}{'Split':<12}{'Loss':>8}{'Pearson':>10}{'Spearman':>10}{'Steps':>8}{'Secs':>8}"
    print(header)
    for row in all_results:
        print(f"{row['item_type']:<10}{row['split']:<12}"
              f"{row['test_loss_nll']:>8.3f}{row['item_grade_pearson_r']:>10.3f}"
              f"{row['item_grade_spearman_r']:>10.3f}{row['n_steps_run']:>8}{row['elapsed_seconds']:>8.0f}")

    print(f"\nTotal run time: {total_elapsed_minutes:.1f} minutes")
    print(f"Results were saved incrementally to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()