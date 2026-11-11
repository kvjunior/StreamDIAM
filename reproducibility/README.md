# Reproducibility

This directory is the submission's artifact: everything needed to reproduce every
number and figure in the paper. The design rule throughout the codebase is that a
number reaches a figure only by being written to `results/*.jsonl` by a committed
experiment driver, then read by a figure that generates nothing itself
(enforced by `tests/test_no_synthetic_in_results.py`).

## Quickstart (smoke run, minutes)

```bash
# 1. environment
conda env create -f reproducibility/environment.yml   # or: pip install -e .
# 2. point at the Elliptic CSVs
export STREAMDIAM_ELLIPTIC_ROOT=/path/to/elliptic_bitcoin_dataset
# 3. a fast, timestep-spanning subsample run of each research question
python -m streamdiam.experiments.rq1_maintenance_speed --root "$STREAMDIAM_ELLIPTIC_ROOT" --per-step 60
python -m streamdiam.experiments.rq2_accuracy_over_time --root "$STREAMDIAM_ELLIPTIC_ROOT" --per-step 60
python -m streamdiam.experiments.rq3_drift_trigger      --root "$STREAMDIAM_ELLIPTIC_ROOT" --per-step 60
python -m streamdiam.experiments.rq7_scalability        --root "$STREAMDIAM_ELLIPTIC_ROOT"
# 4. figures (read only from results/)
python -m streamdiam.figures.fig_accuracy_over_time results/rq2_accuracy_over_time/elliptic/drift.jsonl --out fig_rq2.png
python -m streamdiam.figures.fig_latency_scaling    results/rq7_scalability/elliptic/scaling.jsonl       --out fig_rq7.png
```

`--per-step N` keeps the first N edges of each timestep: a fast development
subsample that still spans the whole stream. Omit it for the full-graph headline
runs (see the note on scale below).

## Full tests

```bash
export STREAMDIAM_ELLIPTIC_ROOT=/path/to/elliptic_bitcoin_dataset
pytest -q           # correctness, equivalence, and the anti-fabrication guard
```

Tests that need the dataset skip cleanly if `STREAMDIAM_ELLIPTIC_ROOT` is unset.

## Claim-to-script map

See `claims_to_scripts.md`: every claim in the paper is linked to the exact
command that reproduces it.

## A note on scale (stated honestly)

The maintenance maintainers ship as a **pure-Python reference implementation**,
chosen for correctness and portability. It is enough to demonstrate the scaling
*trend* — the incremental maintainer's advantage grows with graph size (RQ7),
and from-scratch recompute exhausts memory on large graphs while the incremental
maintainer continues. Reproducing the trend needs only the commands above.

Pushing the absolute numbers to the full 203k-account Elliptic graph (and the
DIAM-native Ethereum/Bitcoin graphs) needs the optional native store/maintainer
hot path (`store/native/`, future work in the blueprint); the Python reference is
not intended for that scale and will be slow or memory-bound there. We report
what the reference measures and are explicit about where the native path takes
over — we do not extrapolate numbers we did not run.
