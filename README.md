# AutoIRT Simulation Study Replication

This project replicates the simulation study from:

> Sharpnack, Mulcaire, Bicknell, LaFlair, and Yancey (2024), **"AutoIRT:
> Calibrating Item Response Theory Models with Automated Machine
> Learning"** (arXiv:2409.08823)

It is written to be readable by someone with basic Python knowledge but
no prior background in Item Response Theory (IRT) or machine learning
calibration methods. Every function has a plain-English explanation of
*why* it exists, not just *what* it does.

## What is this project actually doing?

Adaptive tests (like language proficiency tests) need to know, for every
question ("item"), how hard it is and how well it separates strong
students from weak ones. Figuring this out normally requires hundreds of
real student responses per item. **AutoIRT** is a method that speeds
this up using machine learning, so that even brand-new items (with zero
responses so far) can get a reasonable difficulty estimate, just from
the item's raw content features.

This project:
1. Generates **fake but realistic** test data where we secretly know the
   true difficulty of every item and the true ability of every student
   (so we can check whether our method recovers the truth).
2. Runs the **AutoIRT algorithm** on that fake data.
3. **Measures** how well it worked, using the same metrics as the paper.

## Folder structure

```
autoirt_project/
├── README.md                 <- you are here
├── requirements.txt          <- Python packages needed to run this
├── config.py                 <- all adjustable experiment settings
├── src/                       <- the actual method, as an importable package
│   ├── __init__.py
│   ├── simulate.py           <- generates fake test data
│   ├── autoirt_model.py      <- the AutoIRT calibration algorithm
│   └── evaluate.py           <- scoring / metrics
├── scripts/
│   └── run_experiment.py     <- the script you actually run
└── results/
    └── metrics.csv           <- output metrics get saved here after running
```

## How to run it

1. Install the required packages (only needs to be done once):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the experiment:

   ```bash
   python scripts/run_experiment.py
   ```

3. Read the printed summary table, and check `results/metrics.csv` for
   the saved numbers afterward.

## How to change the experiment size

Open `config.py` and change any of these numbers, then re-run the script:

| Setting | What it controls |
|---|---|
| `N_ITEMS` | Total number of test questions in the simulated item bank |
| `N_SESSIONS` | Total number of simulated test-takers |
| `N_COLD_START_HOLDOUT_ITEMS` | How many items are held out with zero training data, to test "cold-start" calibration |
| `N_EM_STEPS` | How many rounds of the calibration algorithm to run |
| `N_REPEATS` | How many independent times to run the whole experiment. Set this above 1 (e.g. 5) to see the average result plus how much it bounces around by chance, instead of a single noisy run |

Note: increasing `N_ITEMS` and `N_SESSIONS` will make the script take
noticeably longer to run, since a fresh machine learning ensemble is
re-trained at every EM step. Increasing `N_REPEATS` multiplies the total
run time by that many times, since it reruns the whole experiment from
scratch each time.

## About repeated runs

Machine learning results always have some randomness in them (different
random data, different random model training). Running the experiment
only once can make a result look better or worse than it "really" is,
just by chance.

Setting `N_REPEATS` to something like 5 in `config.py` runs the entire
experiment 5 times with different random seeds, then reports:
- The **mean** (average) of each metric across all 5 runs
- The **standard deviation** (how spread out the 5 results were)

A small standard deviation means the result is stable and trustworthy.
A large one means you are looking at noise, and would need either more
repeats or a bigger dataset to get a reliable answer.

When `N_REPEATS > 1`, results are saved to `results/metrics_all_runs.csv`
(one row per individual run) instead of `results/metrics.csv`.

## What the three metrics mean

| Metric | What it tells you | Good value looks like |
|---|---|---|
| **Test loss (NLL)** | How well predicted probabilities matched actual right/wrong outcomes | Lower is better |
| **Item-grade correlation (Pearson/Spearman)** | Whether items predicted to be easy/hard actually were easy/hard | Closer to 1.0 is better |
| **Ability recovery correlation** | Whether the recovered student ability matches the true (simulated) ability | Closer to 1.0 is better |

## Two experiment types

- **Warm-start**: the item bank is unchanged between training and
  testing; only the test-takers are new. This is the "easy" case.
- **Cold-start**: entirely new items are introduced with zero prior
  responses, and the model must estimate their difficulty using only
  their raw content features. This is the harder, more practically
  useful case, and the main contribution of the AutoIRT paper.

## A note on the machine learning model used

The original paper uses a tool called **AutoGluon-tabular**, which
automatically trains and combines many different models. To keep this
project lightweight and fast to install, we instead hand-build a similar
"stacked ensemble" using three well-known tree-based models
(`RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`) and average
their predictions. This captures the same core idea (several different
tree models voting together) without requiring the heavier AutoGluon
dependency stack. If a byte-for-byte match to the paper's exact tool is
needed later, `GradeEnsembleModel` in `src/autoirt_model.py` is the only
place that would need to change.
