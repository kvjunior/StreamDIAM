# Architecture

StreamDIAM is a pipeline from an append-only transaction stream to fresh risk
scores, with retraining decoupled off the critical path. The critical path is
inference-only; retraining happens to the side and is installed atomically.

## System diagram

```
            append-only stream of transactions (deltas, one per timestamp)
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │  CRITICAL PATH (per delta, bounded work)                 │
        │                                                          │
        │   TemporalMultigraphStore (C2)                           │
        │     append log + per-account in/out sequence index       │
        │                    │                                     │
        │                    ▼                                     │
        │   Incremental maintainer (C3)                            │
        │     Edge2Seq state extension  ──►  affected-set MGD       │
        │     (O(c) per edge)                (exact L-hop closure)  │
        │                    │                                     │
        │                    ▼                                     │
        │              risk scores  ──────────────►  OUTPUT         │
        └────────────────────┬─────────────────────────────────────┘
                             │ scores + (delayed) labels
                             ▼
        ┌─────────────────────────────────────────────────────────┐
        │  OFF THE CRITICAL PATH                                   │
        │   Drift monitor (C4)                                     │
        │     unsupervised: score-distribution (ADWIN)             │
        │     supervised:   prequential error (DDM)                │
        │                    │ status                              │
        │                    ▼                                     │
        │   Retrain trigger (drift / periodic)                    │
        │                    │ fires                               │
        │                    ▼                                     │
        │   Background trainer ──► checkpoint ──► Hot-swap registry │
        │     (fine-tune a COPY)                   (atomic install) │
        └────────────────────┬─────────────────────────────────────┘
                             │ new model version
                             ▼
                 maintainer.rebuild()  (picks up new weights)
```

## Data flow, step by step

1. **Ingest.** A delta (one timestamp's transactions) is appended to the store,
   which keeps a columnar log and a per-account, time-sorted in/out sequence
   index (binary-search range queries).

2. **Maintain (C3).** The incremental maintainer extends only the touched
   accounts' Edge2Seq GRU states (O(c) per edge, running max over time) and then
   refreshes exactly the affected L-hop MGD closure of the delta's endpoints. The
   result equals a full recompute within floating-point summation order
   (`docs/proofs.md`). Risk scores for the touched region are now current.

3. **Score (output).** The maintained view exposes per-account risk scores
   continuously, with no retraining on the path.

4. **Monitor (C4).** Scores feed an unsupervised drift branch (ADWIN on the score
   distribution) immediately; labels, when they arrive (often late), feed a
   supervised branch (prequential error → DDM). The two run on independent
   clocks — the honest model of this domain's label latency.

5. **Retrain, decoupled (C4).** When the trigger fires, the background trainer
   fine-tunes a *copy* of the model on a recency-weighted, class-balanced,
   labelled window; the live model keeps serving throughout. The checkpoint is
   installed by the hot-swap registry in a single atomic pointer rebind, and the
   maintainer rebuilds its view under the new weights.

6. **Evaluate (C5).** The prequential harness wraps all of the above: each tick is
   scored *before* its delta and any retrain that uses it (no look-ahead),
   producing an F1/AP-over-time timeline and a strict first-appearance summary,
   plus latency/throughput/memory — all written to `results/*.jsonl`.

## Correctness anchors

- The from-scratch maintainer is the oracle; the incremental maintainer is tested
  equal to it per delta and across whole real streams.
- The temporal split and the protocol ordering guarantee no future leakage
  (tested).
- Figures read only committed results; fabrication is caught by a guard test.

## Where the native path fits

The store and maintainer ship as a pure-Python reference (correctness, port
ability). The `store/native/` C++/pybind11 hot path (future work) is a drop-in
replacement behind the same `TemporalMultigraphStore` interface for full-scale
latency; nothing above it changes.
