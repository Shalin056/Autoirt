"""
check_feature_richness_full_sweep.py
======================================

Closes two open items from the feature-richness investigation:

1. Y/N Vocab only, so far. This runs both item types.

2. Every check so far showed "rich" recovering difficulty well
   (0.88-0.98 Pearson across seeds) but discrimination much more weakly
   (0.53-0.56). Re-checking the true item-generation formulas in
   simulate_det.py explains why:

       true difficulty     = 4 * sin(z) / |z|
       true discrimination = 0.5 * cos(0.1 * z^2) + 1.0
       (z = feature_1 - feature_2)

   "rich" gives the model sin(z) directly -- exactly what difficulty is
   built from. It does NOT give cos(0.1*z^2) -- it gives z_squared and
   cos(z) as separate, uncombined columns, which still leaves the model
   to reconstruct that specific nonlinear composition itself, the same
   class of hard problem "rich" was built to remove for difficulty. This
   adds a third variant, "richer" (= rich + cos(0.1*z^2) explicitly), to
   test that explanation directly: if discrimination recovery jumps the
   way difficulty's did, the mechanism is confirmed.

Single seed per item type (RANDOM_SEED, matching the original
feature-richness check's first pass) -- not replicated yet. If the
"richer" effect on discrimination looks as large and consistent as the
original "rich" effect on difficulty did, that's the point to replicate
it across seeds the same way check_feature_richness_sensitivity_replicated.py
did for the original finding.

Run with:
    python scripts/check_feature_richness_full_sweep.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from check_feature_richness_sensitivity import run_one_seed, print_summary, RANDOM_SEED

ITEM_TYPES = [
    # (label, n_items, chance_value, items_per_session)
    ("Y/N Vocab", 150, 0.25, 18),
    ("ViC", 80, 0.0, 9),
]
VARIANTS = ("baseline", "rich", "richer")


def main():
    all_results = {}
    for label, n_items, chance_value, items_per_session in ITEM_TYPES:
        print(f"\n{'=' * 70}")
        print(f"ITEM TYPE: {label}  (n_items={n_items}, chance={chance_value})")
        print(f"{'=' * 70}")
        results = run_one_seed(
            RANDOM_SEED, n_items=n_items, n_sessions=6000,
            items_per_session=items_per_session, chance_value=chance_value,
            variants=VARIANTS, verbose=True,
        )
        print_summary(results)
        all_results[label] = results

    print("\n" + "=" * 90)
    print("COMBINED SUMMARY")
    print("=" * 90)
    header = f"{'Item type':<12}{'Variant':<12}{'Difficulty r':>14}{'Discrimination r':>18}"
    print(header)
    for label, results in all_results.items():
        for r in results:
            print(f"{label:<12}{r['variant']:<12}{r['difficulty_pearson']:>14.4f}"
                  f"{r['discrimination_pearson']:>18.4f}")

    print("\nKey comparison for the discrimination hypothesis (richer vs. rich, same item type):")
    for label, results in all_results.items():
        by_variant = {r["variant"]: r for r in results}
        if "rich" in by_variant and "richer" in by_variant:
            delta = (by_variant["richer"]["discrimination_pearson"]
                     - by_variant["rich"]["discrimination_pearson"])
            print(f"  {label}: discrimination Pearson rich={by_variant['rich']['discrimination_pearson']:.4f} "
                  f"-> richer={by_variant['richer']['discrimination_pearson']:.4f}  (delta {delta:+.4f})")

    print("\nA large positive delta here (comparable in size to rich's difficulty jump over")
    print("baseline) confirms the missing-composite-term explanation for why discrimination")
    print("recovery lagged difficulty recovery. A small delta means the explanation is wrong")
    print("or incomplete, and the discrimination gap needs a different diagnosis.")


if __name__ == "__main__":
    main()
