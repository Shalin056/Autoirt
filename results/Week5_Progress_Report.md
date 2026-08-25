# **Week 5 Progress Report: Closing Out the AutoIRT Gap Investigation** 

Prepared by: Shalin Bhavsar Date: August 19 - August 26, 2026 Supervisor: Dr. Yi Zheng, Arizona State University 

## **Summary** 

Following your request to spend one more week attempting to close the gap between this implementation and the paper's numbers, I tested every mechanically tunable explanation I could identify: AutoML backend choice, EM convergence behavior for both Cold-start and Jump-start, and two specific feature-representation questions (engineered transforms of the existing synthetic features, and an item-ID-as-feature detail from the papers' Method sections that I had missed on an earlier reading). Three hypotheses were ruled out. Two were confirmed as real, and one of those also explains a pattern seen throughout the week: difficulty recovery consistently outperforming discrimination recovery. The overall picture is that the remaining gap looks feature-fidelity-limited rather than a bug or a tuning gap, which lines up with your plan to move to the NLP feature phase regardless of the outcome here. Full detail follows. 

## **1. Backend (ensemble vs. AutoGluon): ruled out** 

Ran all 12 DET conditions (2 item types x 6 splits) through both backends on identical data and splits. Per-condition Pearson differences were small (0.01 to 0.02 on average) and inconsistent in direction: AutoGluon was slightly better for Y/N Vocab across every condition, but slightly worse for ViC in 5 of 6 conditions. This comparison ran at the same small-scale settings (150/80 items, 6,000 sessions) as the small-scale replication, where the gap to the paper ranges from 0.022 to 0.17 depending on condition; the backend deltas are an order of magnitude smaller than that. Backend choice was never the explanation for the gap. 

## **2. EM convergence, Cold-start and Jump-start: ruled out** 

Extended max_em_steps from 20 to 40 and tracked held-out Pearson at every single step, not just the final one, for the two conditions most worth checking. 

Cold-start (both item types): held-out Pearson was flat or declining from around step 7 onward. ViC's held-out Pearson actually peaked at step 7 (0.561) and was lower by step 20 (0.501); running longer cost accuracy rather than gaining it. 

Jump-start (the worst full-scale gap conditions: Y/N Vocab Jump 40, ViC Jump 20): converged cleanly on their own at step 12 to 13, nowhere near the extended 40-step ceiling. This is corroborated independently by the full-scale replicated data, where jump-start conditions already converge at a mean of 10.67 to 17.0 steps across 3 replications, well under the 20-step cap used throughout. 

Neither condition is limited by iteration count. 

## **3. Feature representation: confirmed as a real lever** 

Isolated the question by holding the true item parameters completely fixed and asking whether an easier-to-use representation of the same 2 raw synthetic features (explicit z = feature_1 - feature_2, z-squared, sin(z), cos(z), |z| columns) improves cold-start item parameter recovery, without adding any new information. It does, substantially: difficulty recovery correlation moved from -0.23 (baseline) to 0.88 (richer representation) on the 

initial run. Replicated across 5 seeds: mean paired gain +0.675 (difficulty), +0.473 (discrimination), +0.581 (an approximate item-grade correlation), all three positive in every single replication, none close to crossing zero. 

## **4. Item ID as a feature: ruled out, but for an informative reason** 

Re-reading both papers' Method sections closely surfaced a detail I had missed: item ID is passed to the AutoML predictor as an explicit feature, described as functioning like a random effect. I checked our implementation's FEATURE_COLUMNS and confirmed we do not do this. I built a targeted test: does adding item ID (one-hot encoded across the operational item set) improve recalibration of items the model already has substantial response history for? This specifically tests the Warm-start use case, not Cold-start, since a held-out pilot item's ID was never seen in training and there is nothing to look up for it. 

Result: no meaningful effect (-0.005 difficulty, -0.007 discrimination), but for an informative reason rather than a null one. With about 1,440 responses per operational item and true theta (no MCEM noise), the baseline model already recovers difficulty at 0.976 Pearson, essentially at ceiling. There is no room left for item ID to help. This is useful negative evidence: it suggests the real pipeline's Warm-start gap against the paper under uniform random item selection, the configuration we are back on, is unlikely to be a features problem. That gap does exist (roughly 0.02 to 0.04 at small scale, 0.07 to 0.14 at full scale) but is more likely MCEM ability-estimation noise, since that is the one thing this clean test deliberately removes and the real pipeline cannot. 

## **5. Why discrimination recovery lags difficulty recovery** 

Every check this week showed the same pattern: richer features recover difficulty well (0.75 to 0.98 across item types and seeds) but discrimination more weakly (0.31 to 0.56). Tracing this against the true item-generation formulas explains part of it directly: difficulty is 4*sin(z)/|z|, and the richer feature set gives the model sin(z) directly. Discrimination is 0.5*cos(0.1*z-squared)+1.0, cosine of z-squared, but the richer set only gave z-squared and cos(z) as separate, uncombined columns, leaving the model to reconstruct that specific composition itself. 

Tested directly: added a third variant with the exact missing term, cos(0.1*z-squared). Discrimination recovery jumped in both item types: Y/N Vocab moved from 0.455 to 0.705 (+0.250); ViC moved from 0.311 to 0.454 (+0.143). This confirms the missing-composite-term explanation is real. 

It only partially closes the gap, though. Even with the exact right feature, discrimination recovery (0.70 / 0.45) still trails difficulty recovery (0.91 / 0.75) in the same runs. My reading is that this remainder is structural rather than further feature-engineerable: difficulty has additive noise on a signal spanning roughly -4 to 4, while discrimination has multiplicative, log-space noise compressed into a narrower range of roughly 0.5 to 1.5, so the same noise level eats a proportionally larger share of discrimination's usable signal. That is a property of how the simulation generates the ground truth, not something any feature representation can fix. 

Caveat: this finding is from a single seed per item type, not yet replicated across multiple seeds the way the Section 3 finding was. I am reporting it as well supported but not yet held to the same evidentiary bar as the headline result. 

## **6. Overall assessment** 

Every mechanically tunable hypothesis I could identify has now been tested. 

**Hypothesis Verdict** 

|Backend (ensemble vs. AutoGluon)|Ruled out|
|---|---|
|EM convergence, Cold-start|Ruled out|
|EM convergence, Jump-start|Ruled out|
|Feature representation|Confirmed real lever (replicated)|
|Item ID as feature|Ruled out (points at MCEM noise for Warm-start)|
|Discrimination-specific feature gap|Partially explained; residual looks structural|



The consistent picture across all of this is that the remaining gap to the paper's numbers is very likely a feature-fidelity limitation (our synthetic 2-feature items versus the paper's real BERT embeddings and corpus statistics), not something further tuning of the algorithm, backend, or convergence settings can close. This matches your plan to move to the NLP feature phase regardless of the outcome here. 

## **Next steps** 

- Ready to move on to simulating real DET item features using NLP packages, per your plan. 

- Two smaller optional threads if useful, otherwise I will leave them here: the Warm-start / MCEM-noise hypothesis from Section 4 could be tested directly the same way Cold-start's convergence was; and the discrimination-recovery finding in Section 5 could be replicated across seeds if you would like more confidence before treating it as settled. 

- Otherwise, ready to start on the NLP feature work whenever you would like. 

