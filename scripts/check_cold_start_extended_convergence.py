"""
check_cold_start_extended_convergence.py
==========================================

run_det_backend_comparison.py showed Y/N Vocab Cold-start hitting the
max_em_steps=20 cap without meeting the convergence check, in BOTH
backends. Looking at the actual per-step training loss numbers from that
run: the last 10 steps' mean only moved by 0.001-0.004 (ensemble/
autogluon respectively), well within the step-to-step noise band of
0.004-0.006 seen over the same window. That is consistent with the loss
having already flattened out, just bouncing around due to Monte Carlo
EM's single stochastic ability draw per session -- not with the model
still meaningfully learning. But "the training loss looks flat" and "the
held-out item parameters have stopped improving" are two different
claims, and only the second one actually matters for whether raising
DET_MAX_EM_STEPS is worth doing.

ViC Cold is a different case: ensemble converged right at the step-20
boundary and autogluon converged cleanly at step 13, so ViC Cold is not
failing to converge -- it converges, just at a value far below the
paper's number (0.50 ensemble / 0.43 autogluon vs. paper's 0.742). That
gap needs a different explanation (most likely the item-bank-size effect
the paper itself documents for cold-start), and this script's held-out
trajectory for ViC is included mainly as a contrast case: if ViC's
held-out Pearson is also flat well before step 13-20, that supports
"already exhausted, not a step-count problem" for ViC specifically.

This reruns the EXACT same Cold-start condition, data, and seeds as
run_det_backend_comparison.py's ensemble/Y-N-Vocab/Cold and
ensemble/ViC/Cold passes (see COMPARISON_SEED and the seed offsets
there), just with max_em_steps raised from 20 to 40 and a step_callback
(added to run_autoirt_calibration in autoirt_model.py) that evaluates
held-out Pearson/Spearman/loss at EVERY step, not just the final one.
Steps 1-20 here should reproduce the SAME training losses already seen
in det_backend_comparison.csv -- if they don't match, something about
the data/seeding has drifted and that needs fixing before trusting
steps 21-40.

Ensemble backend only (not AutoGluon): this needs 40 evaluate_calibration
calls per item type on top of the model fits themselves, and AutoGluon
was already ~2.4x slower per step in the backend comparison with no
meaningful accuracy edge for Cold-start specifically (+0.024 Y/N Vocab,
-0.066 ViC) -- not worth doubling the runtime of this check for that.

Run with:
    python scripts/check_cold_start_extended_convergence.py

Takes noticeably longer than a normal 20-step Cold-start run (roughly
2x, plus the extra evaluate_calibration calls, which are cheap). Results
saved incrementally per step, so an interrupted run still leaves usable
partial data.
"""

import sys
import os
import csv
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.simulate_det import simulate_det_items, simulate_det_sessions, simulate_det_responses, \
    COLD_JUMP_SPLIT_DAY
from src.autoirt_model import run_autoirt_calibration
from src.evaluate import evaluate_calibration
from run_det_experiment import choose_pilot_and_operational_items, split_cold_start

# Must match run_det_backend_comparison.py's COMPARISON_* constants and
# seed scheme exactly, or this is silently testing different data instead
# of extending the same run.
COMPARISON_N_YN_VOCAB_ITEMS = 150
COMPARISON_N_VIC_ITEMS = 80
COMPARISON_N_DET_SESSIONS = 6000
COMPARISON_SEED = 42

EXTENDED_MAX_EM_STEPS = 40

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                             "cold_start_extended_convergence.csv")
FIELDNAMES = ["item_type", "step", "loss_nonparametric", "loss_parametric",
              "held_out_test_loss_nll", "held_out_pearson", "held_out_spearman"]


def append_row(row: dict):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(item_type_name: str, n_items: int, chance_value: float,
            items_per_session: int, seed_offset: int):
    items = simulate_det_items(n_items=n_items, chance_value=chance_value,
                                random_seed=COMPARISON_SEED + seed_offset)
    sessions = simulate_det_sessions(n_sessions=COMPARISON_N_DET_SESSIONS,
                                      random_seed=COMPARISON_SEED + seed_offset + 1)
    theta_key = "yn_theta" if item_type_name == "Y/N Vocab" else "vic_theta"
    theta = sessions[theta_key]
    all_responses = simulate_det_responses(
        items, theta, sessions["session_day"], items_per_session,
        random_seed=COMPARISON_SEED + seed_offset + 2,
    )
    pilot_item_ids, operational_item_ids = choose_pilot_and_operational_items(
        n_items, random_seed=COMPARISON_SEED + seed_offset + 3,
    )
    train, test = split_cold_start(all_responses, operational_item_ids, COLD_JUMP_SPLIT_DAY)

    print(f"\n{'#' * 70}")
    print(f"# {item_type_name} Cold-start, extended to {EXTENDED_MAX_EM_STEPS} steps")
    print(f"{'#' * 70}")
    print(f"train n={len(train['grade'])}, test n={len(test['grade'])}  "
          f"(should match run_det_backend_comparison.py's ensemble/{item_type_name}/Cold row)")

    def step_callback(step, discrimination, difficulty, loss_nonparametric, loss_parametric):
        metrics = evaluate_calibration(test, theta, discrimination, difficulty, n_items=n_items)
        row = {
            "item_type": item_type_name,
            "step": step,
            "loss_nonparametric": round(loss_nonparametric, 4),
            "loss_parametric": round(loss_parametric, 4),
            "held_out_test_loss_nll": round(metrics["test_loss_nll"], 4),
            "held_out_pearson": round(metrics["item_grade_pearson_r"], 4),
            "held_out_spearman": round(metrics["item_grade_spearman_r"], 4),
        }
        append_row(row)
        flag = "  <-- step 20 (this is where the original run stopped)" if step == 20 else ""
        print(f"  step {step:>2}: held-out Pearson={row['held_out_pearson']:.4f}  "
              f"held-out loss={row['held_out_test_loss_nll']:.4f}{flag}")

    start_time = time.time()
    result = run_autoirt_calibration(
        train, items, n_items=n_items, random_seed=COMPARISON_SEED,
        backend="ensemble", max_em_steps=EXTENDED_MAX_EM_STEPS,
        step_callback=step_callback,
    )
    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.0f}s. stopped_reason={result['stopped_reason']}, "
          f"n_steps_run={result['n_steps_run']}")


def main():
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    run_one("Y/N Vocab", COMPARISON_N_YN_VOCAB_ITEMS, config.YN_CHANCE,
             config.YN_ITEMS_PER_SESSION, seed_offset=0)
    run_one("ViC", COMPARISON_N_VIC_ITEMS, config.VIC_CHANCE,
             config.VIC_ITEMS_PER_SESSION, seed_offset=100)

    print(f"\nSaved step-by-step held-out metrics to: {RESULTS_PATH}")
    print("\nWhat to look for:")
    print("- Compare held_out_pearson at step 20 here against the Pearson already reported")
    print("  for ensemble/Y-N-Vocab/Cold and ensemble/ViC/Cold in det_backend_comparison.csv --")
    print("  they should match closely; if they don't, the seeding has drifted somewhere.")
    print("- If held_out_pearson is still climbing noticeably from step 20 to 40, raising")
    print("  DET_MAX_EM_STEPS is worth doing for Cold-start specifically.")
    print("- If it's flat (bouncing within +/-0.01 or so) from well before step 20 onward,")
    print("  the training-loss plateau reflects a real plateau in calibration quality too,")
    print("  and the remaining gap needs a different lever than more EM steps.")


if __name__ == "__main__":
    main()