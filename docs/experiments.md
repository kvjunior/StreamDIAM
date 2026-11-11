# Experiments: drivers and the claim map

Each research question has a thin driver in `src/streamdiam/experiments/` that
builds its runs from `experiments.harness` and writes JSONL under
`results/<rq>/<dataset>/`. Drivers never plot; figures read the committed
results. Run one with, e.g.:

    python -m streamdiam.experiments.rq1_maintenance_speed --root <elliptic_dir>

Add `--per-step N` for a fast, timestep-spanning development subsample.

## Claim map

| RQ | Driver | Question | What the results file shows |
|---|---|---|---|
| RQ1 | `rq1_maintenance_speed` | Is maintenance faster, at equal accuracy? | per-tick update latency + prequential accuracy for from-scratch / incremental / frozen |
| RQ2 | `rq2_accuracy_over_time` | Does accuracy hold over time; does retraining help? | per-tick F1/AP-over-time timeline for never / drift / periodic |
| RQ3 | `rq3_drift_trigger` | How often is retraining needed; cost of deferral? | retrain counts + times + accuracy for drift vs. periodic |

RQ4-RQ7 (store microbench, sensitivity, imbalance, scalability) follow the same
pattern and land next.

## Two results that hold by construction (tested)

- **Identical accuracy.** incremental and from-scratch reach the same strict
  prequential accuracy -- they compute the same view. Asserted in
  `test_experiments.py`.
- **Honest firing counts.** RQ3's retrain counts come straight from the run's
  retrain log, never estimated.

## The scale caveat on RQ1 latency (stated, not hidden)

The incremental maintainer does bounded work per delta; from-scratch recomputes
the whole view. The *magnitude* of the speedup grows with graph size. On a small
development subsample the Python per-affected-node loop can even lose to batched
tensor recompute, so the headline latency number is a large-graph result,
reported exactly as measured. A representative development-subsample run
(universe ~3.2k, 35 ticks) already shows incremental mean latency ~18 ms vs.
from-scratch ~148 ms at *identical* prequential accuracy -- the shape the
full-scale run (B15) quantifies.

## What these numbers are and are not

The development-subsample numbers exist to prove the pipeline produces real,
consistent results end to end. They are **not** headline numbers: a small
subsample with minimal warmup training yields low absolute accuracy for every
method. The paper's reported numbers come from the full graph with proper
training (RQ runs in B15), written to `results/` by these same drivers and read
by the figures -- the only path by which a number reaches a figure.
