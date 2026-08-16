"""
test_det_split_pipeline.py
===========================

Stage-by-stage correctness tests for the DET split logic in
run_det_experiment.py: item partitioning, date splitting, R-capping for
jump-start, and response reorganization. Uses small, hand-constructed
synthetic response sets with a known ground truth for each stage, so
every assertion below is checked against a value computed by hand, not
just "the code ran without crashing."

This tests the actual functions imported from run_det_experiment.py,
not a reimplementation, so a bug in the real split logic will fail a
test here.

Run with:
    python scripts/test_det_split_pipeline.py

Exits non-zero and prints which stage failed if any assertion fails;
prints "ALL TESTS PASSED" if every stage checks out.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from run_det_experiment import (
    choose_pilot_and_operational_items, split_cold_start,
    split_jump_start, split_warm_start, select_subset,
)


def make_responses(rows):
    """rows: list of dicts with keys item_id, day, response_seq, grade,
    session_id. Returns the same dict-of-arrays format used everywhere
    else in the pipeline."""
    keys = ["item_id", "day", "response_seq", "grade", "session_id"]
    return {k: np.array([r[k] for r in rows]) for k in keys}


def assert_no_overlap(mask_a, mask_b, label):
    overlap = np.sum(mask_a & mask_b)
    assert overlap == 0, f"{label}: {overlap} rows appear in both sets (should be disjoint)"


def assert_row_ids_equal(responses, mask, expected_row_indices, label):
    actual = set(np.where(mask)[0].tolist())
    expected = set(expected_row_indices)
    assert actual == expected, (
        f"{label}: row indices differ.\n  expected: {sorted(expected)}\n  actual:   {sorted(actual)}"
    )


def test_item_partitioning():
    """Stage 1: pilot/operational split. Every item assigned to exactly
    one group; group sizes match n_items // 2; results differ by seed
    (not hardcoded to always pick the same items)."""
    for n_items, seed in [(10, 0), (11, 0), (3290, 42), (585, 142)]:
        pilot_ids, operational_ids = choose_pilot_and_operational_items(n_items, random_seed=seed)

        assert len(pilot_ids) == n_items // 2, (
            f"n_items={n_items}: expected {n_items // 2} pilot items, got {len(pilot_ids)}"
        )
        assert len(operational_ids) == n_items - n_items // 2, (
            f"n_items={n_items}: expected {n_items - n_items // 2} operational items, "
            f"got {len(operational_ids)}"
        )

        pilot_set = set(pilot_ids.tolist())
        operational_set = set(operational_ids.tolist())
        assert pilot_set.isdisjoint(operational_set), (
            f"n_items={n_items}: pilot and operational sets overlap: "
            f"{pilot_set & operational_set}"
        )
        assert pilot_set | operational_set == set(range(n_items)), (
            f"n_items={n_items}: pilot + operational doesn't cover every item id"
        )

    # Different seeds should (almost certainly) produce different partitions --
    # if they don't, choose_pilot_and_operational_items isn't actually using
    # the seed, which would make every "replication" identical.
    pilot_a, _ = choose_pilot_and_operational_items(100, random_seed=1)
    pilot_b, _ = choose_pilot_and_operational_items(100, random_seed=2)
    assert set(pilot_a.tolist()) != set(pilot_b.tolist()), (
        "Different random seeds produced the identical pilot set -- "
        "check_pilot_and_operational_items may not be using random_seed correctly"
    )
    print("[PASS] Stage 1: item partitioning (pilot/operational split)")


def test_cold_start_split():
    """Stage 2: date splitting for cold-start. Train = operational items,
    day < split_day. Test = ALL items (pilot + operational), day >= split_day.
    Every row lands in exactly train or test, never both, never neither."""
    split_day = 5
    operational_ids = np.array([0, 1, 2])
    pilot_ids = np.array([3, 4, 5])

    rows = []
    row_index = 0
    ground_truth_train = set()
    ground_truth_test = set()
    for item_id in [0, 1, 2, 3, 4, 5]:
        for day in [2, 4, 5, 7]:  # two pre-split days, two post-split days
            rows.append({"item_id": item_id, "day": day, "response_seq": row_index,
                         "grade": 1, "session_id": row_index})
            is_operational = item_id in operational_ids.tolist()
            is_pre_split = day < split_day
            if is_operational and is_pre_split:
                ground_truth_train.add(row_index)
            if not is_pre_split:
                ground_truth_test.add(row_index)
            row_index += 1

    responses = make_responses(rows)
    train, test = split_cold_start(responses, operational_ids, split_day)

    train_mask = np.isin(np.arange(len(rows)), [])  # placeholder, rebuild via row_seq matching
    # Recover which original rows ended up in train/test by response_seq (unique per row here).
    train_row_ids = set(train["response_seq"].tolist())
    test_row_ids = set(test["response_seq"].tolist())

    assert train_row_ids == ground_truth_train, (
        f"cold-start train mismatch.\n  expected: {sorted(ground_truth_train)}\n"
        f"  actual:   {sorted(train_row_ids)}"
    )
    assert test_row_ids == ground_truth_test, (
        f"cold-start test mismatch.\n  expected: {sorted(ground_truth_test)}\n"
        f"  actual:   {sorted(test_row_ids)}"
    )
    assert train_row_ids.isdisjoint(test_row_ids), "cold-start train/test overlap"
    print("[PASS] Stage 2: cold-start date split")


def test_jump_start_r_capping():
    """Stage 3: jump-start R-capping. This is the stage Dr. Zheng flagged
    as having the most moving pieces -- date split, R value, and
    chronological reconstruction all interact here.

    Three pilot items with deliberately different post-split response
    counts, so R=2 exercises all three cases in one test:
      pilot item A: 5 post-split responses (more than R -- only first 2 kept)
      pilot item B: exactly 2 post-split responses (exactly R -- both kept)
      pilot item C: 1 post-split response (fewer than R -- the 1 available
                    is kept, this item never reaches R, matching the
                    data-starvation case the [DIAGNOSTIC] print flags)
    Chronological order is day first, then response_seq as the tie-break
    for same-day responses -- item A's 5 responses are placed across two
    days specifically to test both parts of that ordering.
    """
    split_day = 5
    operational_ids = np.array([0])
    pilot_ids = np.array([10, 11, 12])  # A, B, C
    R = 2

    rows = []
    row_index = 0

    def add_row(item_id, day):
        nonlocal row_index
        rows.append({"item_id": item_id, "day": day, "response_seq": row_index,
                     "grade": 1, "session_id": row_index})
        row_index += 1

    # Operational item: 2 pre-split (used for training), 2 post-split
    # (NOT used for jump-start training -- only pilot post-split responses
    # are added on top of operational pre-split data).
    add_row(0, day=1)  # op pre-split -> row 0, expect in train
    add_row(0, day=2)  # op pre-split -> row 1, expect in train
    add_row(0, day=6)  # op post-split -> row 2, expect in test (not train)
    add_row(0, day=7)  # op post-split -> row 3, expect in test (not train)

    # Pilot item A (id 10): 5 post-split responses across two days.
    # Chronological order (day, then response_seq) should be:
    #   row 4 (day 6), row 5 (day 6), row 6 (day 7), row 7 (day 8), row 8 (day 8)
    # R=2 keeps the first 2 of that order: rows 4 and 5.
    add_row(10, day=6)  # row 4 -- expect KEPT (1st in order)
    add_row(10, day=6)  # row 5 -- expect KEPT (2nd in order)
    add_row(10, day=7)  # row 6 -- expect NOT kept (3rd)
    add_row(10, day=8)  # row 7 -- expect NOT kept (4th)
    add_row(10, day=8)  # row 8 -- expect NOT kept (5th)

    # Pilot item B (id 11): exactly R=2 post-split responses -- both kept.
    add_row(11, day=6)  # row 9  -- expect KEPT
    add_row(11, day=7)  # row 10 -- expect KEPT

    # Pilot item C (id 12): only 1 post-split response, R=2 requested --
    # the 1 available is kept (this is the data-starved case).
    add_row(12, day=6)  # row 11 -- expect KEPT (only one available)

    responses = make_responses(rows)
    train, test = split_jump_start(responses, operational_ids, pilot_ids, split_day, R)

    train_row_ids = set(train["response_seq"].tolist())
    test_row_ids = set(test["response_seq"].tolist())

    expected_train = {0, 1,      # operational pre-split
                       4, 5,      # pilot A: first 2 chronologically
                       9, 10,     # pilot B: both (== R)
                       11}        # pilot C: only 1 available
    expected_test = {2, 3,        # operational post-split (not touched by R-capping)
                      6, 7, 8}    # pilot A: remaining 3 responses go to test

    assert train_row_ids == expected_train, (
        f"jump-start train mismatch.\n  expected: {sorted(expected_train)}\n"
        f"  actual:   {sorted(train_row_ids)}"
    )
    assert test_row_ids == expected_test, (
        f"jump-start test mismatch.\n  expected: {sorted(expected_test)}\n"
        f"  actual:   {sorted(test_row_ids)}"
    )
    assert train_row_ids.isdisjoint(test_row_ids), "jump-start train/test overlap"

    # Every post-split row must land in exactly train or test -- none silently dropped.
    all_post_split = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    accounted_for = (train_row_ids | test_row_ids) & all_post_split
    assert accounted_for == all_post_split, (
        f"some post-split rows vanished entirely (neither train nor test): "
        f"{all_post_split - accounted_for}"
    )
    print("[PASS] Stage 3: jump-start R-capping and chronological ordering")


def test_jump_start_r_values_actually_differ():
    """Regression test for the exact bug found in the previous report:
    with enough post-split data, R=20/40/80 must produce train sets of
    different sizes for the same pilot item pool. If they don't, the
    R-capping is silently collapsing into 'use whatever's available'."""
    split_day = 5
    operational_ids = np.array([0])
    pilot_ids = np.array([10])

    rows = []
    row_index = 0
    for day_offset in range(100):  # 100 post-split responses for the one pilot item
        rows.append({"item_id": 10, "day": split_day + 1, "response_seq": row_index,
                     "grade": 1, "session_id": row_index})
        row_index += 1
    responses = make_responses(rows)

    train_sizes = {}
    for r in [20, 40, 80]:
        train, _ = split_jump_start(responses, operational_ids, pilot_ids, split_day, r)
        train_sizes[r] = len(train["grade"])

    assert train_sizes[20] < train_sizes[40] < train_sizes[80], (
        f"Jump 20/40/80 train sizes should be strictly increasing when enough "
        f"data exists, got {train_sizes} -- this is the exact failure mode "
        f"from the earlier data-starvation bug."
    )
    assert train_sizes[80] == 80, f"expected exactly 80 rows kept for R=80 with 100 available, got {train_sizes[80]}"
    print("[PASS] Stage 3b: Jump 20/40/80 produce distinct train sizes when data isn't starved")


def test_warm_start_split():
    """Stage 4: warm-start. Pure date split, no item filtering at all --
    every item (pilot or operational) is eligible in both train and test,
    the only criterion is which side of split_day a response falls on."""
    split_day = 5
    rows = []
    row_index = 0
    for item_id in range(6):  # mix of what would be pilot/operational ids elsewhere
        for day in [3, 4, 6, 8]:
            rows.append({"item_id": item_id, "day": day, "response_seq": row_index,
                         "grade": 1, "session_id": row_index})
            row_index += 1
    responses = make_responses(rows)

    train, test = split_warm_start(responses, split_day)
    train_row_ids = set(train["response_seq"].tolist())
    test_row_ids = set(test["response_seq"].tolist())

    expected_train = {i for i, r in enumerate(rows) if r["day"] < split_day}
    expected_test = {i for i, r in enumerate(rows) if r["day"] >= split_day}

    assert train_row_ids == expected_train, "warm-start train mismatch"
    assert test_row_ids == expected_test, "warm-start test mismatch"
    assert train_row_ids | test_row_ids == set(range(len(rows))), (
        "warm-start: some rows missing from both train and test"
    )
    assert train_row_ids.isdisjoint(test_row_ids), "warm-start train/test overlap"
    print("[PASS] Stage 4: warm-start date split")


def test_select_subset():
    """Stage 5: the low-level row-selection helper every split function
    is built on. Boolean mask in, matching rows out, all fields kept in
    sync (a bug here would silently corrupt every split condition)."""
    responses = make_responses([
        {"item_id": 0, "day": 1, "response_seq": 0, "grade": 1, "session_id": 100},
        {"item_id": 1, "day": 2, "response_seq": 1, "grade": 0, "session_id": 101},
        {"item_id": 2, "day": 3, "response_seq": 2, "grade": 1, "session_id": 102},
    ])
    mask = np.array([True, False, True])
    subset = select_subset(responses, mask)

    assert list(subset["item_id"]) == [0, 2], "select_subset: item_id not filtered correctly"
    assert list(subset["session_id"]) == [100, 102], (
        "select_subset: session_id out of sync with item_id after filtering -- "
        "fields are no longer aligned per-row"
    )
    assert len(subset["grade"]) == 2, "select_subset: wrong row count"
    print("[PASS] Stage 5: select_subset row/field alignment")


def main():
    tests = [
        test_item_partitioning,
        test_cold_start_split,
        test_jump_start_r_capping,
        test_jump_start_r_values_actually_differ,
        test_warm_start_split,
        test_select_subset,
    ]
    failures = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            failures.append((test.__name__, str(e)))
            print(f"[FAIL] {test.__name__}\n       {e}")

    print()
    if failures:
        print(f"{len(failures)}/{len(tests)} STAGE(S) FAILED -- see [FAIL] lines above")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} STAGES PASSED")


if __name__ == "__main__":
    main()