# Baselines and comparison strategies

The old submission was rejected partly for **mocked baselines**. The rule here is
absolute: a baseline is either real and runnable through the one harness, or it
is not listed as done. This document states exactly what is real now and what is
pending real upstream code -- nothing is stubbed to look complete.

## Two axes of comparison

Streaming detection has two independent knobs, and the harness separates them:

**View-maintenance strategy** (how fresh scores are kept) -- a `StreamingMaintainer`:

| name | what it does | role |
|---|---|---|
| `from_scratch` | recompute the whole view each tick | oracle + baseline |
| `incremental` | C3: incremental Edge2Seq + exact affected-set MGD | **proposed method** |
| `frozen` | compute once, never update on new edges | ablation (is maintenance worth it?) |

`incremental` is proven to equal `from_scratch` at **every tick** of a real
Elliptic stream (`test_baselines.py::test_incremental_equals_from_scratch_every_tick`),
so RQ1's "faster *and* identical" claim rests on a test, not a hope.

**Retrain policy** (when weights update) -- a run config:

| policy | when it retrains | the baseline it represents |
|---|---|---|
| `drift` | only when the monitor fires | **proposed method (C4)** |
| `periodic` | every *k* ticks, drift-blind | DIAM full-retrain-per-window |
| `never` | not after warmup | DIAM-frozen weights |

Any strategy × any policy runs through the identical `PrequentialProtocol`, so
every cell of the comparison is measured the same way (resolves R2 O7).

## External-model baselines (pending upstream code, not stubbed)

These are different *detectors*, not maintenance strategies, and faithful
versions require their authors' code. They will be wired through the same
`StreamingMaintainer` interface, with the upstream repo pinned under
`third_party/` at an exact commit, each in its own step:

- **XGB-Graph** -- gradient-boosted trees on node features + neighbour
  aggregation (GADBench found tree ensembles strong on these graphs). A real,
  self-contained baseline (no heavy repo); next to land.
- **TGN / APAN** -- temporal-GNN baselines; wrap the official implementations.
- **Spade** -- dense-subgraph fraud baseline (a DB-venue comparison point).

Until a baseline's real implementation runs through the harness it is listed here
as pending and appears in **no** results table. A pending baseline is never
represented by a placeholder number.
