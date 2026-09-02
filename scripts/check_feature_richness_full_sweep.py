"""
check_feature_richness_full_sweep.py
======================================

Extends check_feature_richness_sensitivity.py two ways: runs both item
types (Y/N Vocab and ViC), and adds a third variant "richer" (=rich +
cos(0.1*z^2)) to test why "rich" recovers difficulty well but
discrimination poorly. True item-generation formulas:

    true difficulty     = 4 * sin(z) / |z|
    true discrimination = 0.5 * cos(0.1 * z^2) + 1.0
    (z = feature_1 - feature_2)

"rich" gives sin(z) directly (matches difficulty's formula) but not
cos(0.1*z^2) as one combined term (only z_squared and cos(z)
separately). "richer" adds that exact term to test if discrimination
recovery jumps the way difficulty's did with "rich".

Single seed (not replicated) -- replicate if the effect looks real.

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
        # Look up each variant's result by name for this item type.
        by_variant = {}
        for r in results:
            by_variant[r["variant"]] = r

        if "rich" in by_variant and "richer" in by_variant:
            rich_discrimination = by_variant["rich"]["discrimination_pearson"]
            richer_discrimination = by_variant["richer"]["discrimination_pearson"]
            delta = richer_discrimination - rich_discrimination
            print(f"  {label}: discrimination Pearson rich={rich_discrimination:.4f} "
                  f"-> richer={richer_discrimination:.4f}  (delta {delta:+.4f})")

    print("\nLarge positive delta -> confirms the missing-composite-term explanation.")
    print("Small delta -> explanation is wrong/incomplete, discrimination gap needs")
    print("a different diagnosis.")


if __name__ == "__main__":
    main()