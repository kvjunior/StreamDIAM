# Streaming evaluation: design (C5)

The harness is where the old paper failed hardest (rigor, reproducibility,
contradictory tables), so its design is deliberate and conservative. Code:
`src/streamdiam/eval/{metrics,protocol,runner}.py`,
`src/streamdiam/utils/{timing,memory}.py`. Tests: `tests/test_eval.py`.

## 1. One harness, one set of numbers (resolves R2 O7)

The old paper reported an incremental-vs-full-retrain F1 contradiction because
the two came from different code. Here there is one `PrequentialProtocol`. Every
strategy -- from-scratch, incremental, each baseline -- is a `StreamingMaintainer`
plugged into the *same* loop, so their numbers are directly comparable by
construction. A maintainer exposes exactly three operations (`scores`, `ingest`,
`rebuild`); nothing else about a strategy can leak into the measurement.

## 2. Test-then-train ordering, and no look-ahead

Per tick (one timestamp's delta) the loop does, in this order:

1. **Score** the active accounts with the current view.
2. **Update** the store and the maintained view with the delta (timed).
3. **Monitor** drift on the refreshed scores + any known labels.
4. **Retrain** iff the trigger fires: fine-tune, hot-swap, rebuild, reset.

Because scoring precedes both the view update and any retrain, the model that
scores tick `t` was last trained strictly before `t`. The invariant is
machine-checked (`test_no_future_leakage_invariant`): a tick's model version only
advances *after* a tick that retrained, never from the tick's own data.

## 3. Two metric streams

- **Timeline** -- per-tick window metrics over all labelled active accounts: the
  F1/AP-over-time curve (RQ2 headline). Shows how accuracy evolves and how it
  responds to drift and retraining.
- **Strict prequential** -- each account is scored exactly once, at its first
  appearance, *before* it can enter any training window; these decisions
  aggregate to one leakage-free summary. Warmup accounts are pre-marked seen, so
  training data never enters this metric.

Both come from the one run; they measure different, complementary things, and
neither can contradict the other the way two harnesses did.

## 4. Imbalance-aware metrics

Positives are 2-10% of labels, so the metrics are F1 for the illicit class at a
threshold plus threshold-free ROC-AUC and **average precision (AP), the
headline**. Degenerate windows (no positives, or one class) are the norm early in
a stream and are handled explicitly -- AUC is `nan` without both classes, AP is
`nan` without a positive, and `nan`s are carried through, never crashing a run.

## 5. Latency, throughput, memory (RQ1)

`utils/timing` measures per-tick view-update latency with a monotonic clock and
reports the **tail** (p50/p95/p99/max), not just the mean -- a real-time system is
judged by its tail. Throughput is events/second. `utils/memory` probes resident
set size from `/proc` (no psutil). These ride along every run.

## 6. Results only; figures read committed records

A run emits a `RunResult` and, via `write_run_jsonl`, a JSONL file: one summary
record then one per tick. That is the *only* output. Figures (a later batch) read
these committed records and nothing else -- they cannot synthesise data they were
not given, which is the structural guarantee against the old paper's fabricated
figures.

## 7. Honest scope

The harness measures what it is given. It does not itself claim the incremental
maintainer is faster or as accurate -- those are experiments (RQ1/RQ2) run through
it once the incremental and baseline maintainers are wired in (next batches). The
from-scratch maintainer here is both the correctness anchor and the from-scratch
baseline.
