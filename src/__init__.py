"""
AutoIRT Replication Package
============================

This package replicates the simulation study from:

    Sharpnack et al. (2024), "AutoIRT: Calibrating Item Response Theory
    Models with Automated Machine Learning"

It contains three modules:

    simulate.py       -- generates fake (synthetic) test-taker data that
                          follows a known 3PL Item Response Theory model.
    autoirt_model.py  -- the AutoIRT calibration algorithm itself
                          (fits an ML model, then converts it into
                          interpretable IRT parameters).
    evaluate.py       -- measures how good the calibration is, using the
                          same metrics as the paper.

See the top-level README.md for how to run everything.
"""
