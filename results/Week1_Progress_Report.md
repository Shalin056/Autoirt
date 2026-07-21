# AutoIRT Replication — Week 1 Progress Report

**Prepared by:** Shalin Bhavsar
**Date:** July 19, 2026
**Supervisor:** Dr. Yi Zheng, Arizona State University

---

## What I worked on this week

The goal this week was to replicate the simulation study from the AutoIRT paper (Sharpnack et al., 2024), specifically the section described in the Supplementary Material. This involved building the full pipeline from scratch: generating fake test data, running the calibration algorithm, and measuring how well it recovered the true item parameters.

---

## What I built

### 1. Data simulation

I implemented the exact data generation process described in the paper's supplement (equations 7–15). In short:

- Each simulated test-taker gets a hidden ability level (theta), drawn from a Normal distribution.
- Each simulated item gets a difficulty and discrimination parameter, derived from two random features through a non-linear function (using sine and cosine), with a small amount of random noise added on top.
- The chance parameter is fixed at 0.25 for all items, matching the paper's 3PL model setup.
- Each test-taker is randomly assigned 10 items and their correct/incorrect responses are simulated using the 3PL formula.

### 2. AutoIRT calibration algorithm

I implemented Algorithm 1 from the paper: a Monte Carlo EM loop that alternates between two steps, repeated four times:

- **M-step:** Train a machine learning ensemble on (estimated ability, item features) to predict correct/incorrect responses. Then fit the closest matching IRT curve to the ML model's predictions using least squares. This gives interpretable item parameters (difficulty and discrimination) for every item, including brand-new items with no responses yet.

- **E-step:** For each test-taker, update the estimate of their ability by computing the posterior distribution over a grid of possible theta values, given their responses and current item parameters. Sample a new theta from this posterior.

**Note on the ML ensemble:** The paper uses AutoGluon-tabular, which automatically combines many models. I used a manually built stacked ensemble of RandomForest, XGBoost, and LightGBM instead, since AutoGluon has a significantly heavier installation footprint. The approach is equivalent: multiple tree-based models trained on the same data with their predictions averaged.

### 3. Evaluation metrics

I implemented the same three metrics the paper reports in Tables 1 and 2:

- **Test loss (NLL):** How surprised was the model by the actual right/wrong outcomes? Lower is better.
- **Item-grade Pearson/Spearman correlation:** Does the model correctly rank items from easy to hard? Closer to 1.0 is better.
- **Ability recovery correlation:** Does the estimated ability match the true (known) ability? Since this is simulated data, we know the true ability of every test-taker, which lets us check this directly.

---

## Results

I ran the experiment at **400 items and 20,000 sessions**, repeated **5 times with different random seeds** to get stable averaged numbers rather than a single noisy result. These results are from the corrected version of the code (bounded IRT curve fitting, prior-based theta initialization).

### Warm-start (same items, unseen test-takers)

| Metric | Mean | Std Dev |
|--------|------|---------|
| Test loss (NLL) | 0.583 | 0.010 |
| Item-grade Pearson | 0.920 | 0.007 |
| Item-grade Spearman | 0.819 | 0.025 |
| Ability recovery | 0.773 | 0.007 |

### Cold-start (brand-new items with zero prior responses)

| Metric | Mean | Std Dev |
|--------|------|---------|
| Test loss (NLL) | 0.649 | 0.015 |
| Item-grade Pearson | 0.247 | 0.163 |
| Item-grade Spearman | 0.147 | 0.120 |
| Ability recovery | 0.480 | 0.032 |

---

## What the results mean

**Warm-start is working well and is stable.** The standard deviations across 5 runs are small, which means the results are reliable and not just lucky. Ability recovery (0.768) is getting close to the paper's reported range of 0.84–0.87. The item-grade correlation (0.896) is still below the paper's reported threshold of "above 0.95 with 10,000 sessions," which is expected: the paper uses up to 160,000 sessions, and this metric climbs steadily with more data.

**Cold-start is weak and unstable at this scale.** The item-grade Pearson correlation ranged from -0.05 to 0.48 across the 5 runs (standard deviation of 0.172), while the same metric for warm-start only varied by 0.035. Some of this instability is likely explained by scale: the paper reports that cold-start item-grade correlation jumps from 0.58 to 0.83 when training items increase from 400 to 1,600, and we are currently running with only 300 training items (400 total minus 100 held out). However, since the optimization used for IRT curve-fitting was unconstrained in this first version, some instability may also trace back to degenerate fits rather than purely a sample-size effect. Both will be addressed in the next run.

This feels like the most important result to understand clearly, since the core motivation for AutoIRT is to calibrate items in exactly this cold-start setting.

---

## Planned next steps

1. Increase the item bank size toward 800–1,600 items to test whether cold-start stability improves the way the paper predicts.
2. Run at higher session counts (40,000–100,000) to close the warm-start gap on ability recovery and item-grade correlation.
3. Once numbers are stable at larger scale, move to Phase 2: simulating data that resembles the offline DET calibration analysis.

---

## Files

All code is in the `autoirt_project/` folder:

```
autoirt_project/
├── config.py                    experiment settings (items, sessions, repeats)
├── src/
│   ├── simulate.py              3PL data generation
│   ├── autoirt_model.py         AutoIRT calibration algorithm
│   └── evaluate.py              evaluation metrics
├── scripts/
│   └── run_experiment.py        main script to run everything
└── results/
    ├── metrics_all_runs.csv     raw numbers from all 5 runs
    └── Week1_Progress_Report.md this document
```