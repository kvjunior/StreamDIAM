# StreamDIAM

**Incremental inference maintenance for real-time illicit-account detection on
temporal multigraphs.**

StreamDIAM reframes streaming fraud detection as a data-engineering problem:
instead of re-running a graph model on every update or retraining per window, it
maintains the model's *inference view* — per-account Edge2Seq states, multigraph
discrepancy messages, and risk scores — as an **incrementally updated view over an
append-only temporal multigraph**, with bounded per-delta work and exact
correctness against a from-scratch oracle. The model is retrained only when drift
is detected, decoupled and off the critical path. Everything is measured by one
honest streaming harness with real baselines.

## Contributions

- **C1 — Formalization.** Real-time detection as incremental inference-view
  maintenance over an append-only temporal multigraph, with an ε-correctness
  criterion (`docs/problem_formalization.md`).
- **C2 — Temporal multigraph store.** Append-only log + per-account in/out
  sequence index with binary-search range queries; a pure-Python reference that
  doubles as a correctness oracle (`src/streamdiam/store/`).
- **C3 — Bounded incremental maintenance.** O(c)-per-edge Edge2Seq state
  extension + exact affected-set MGD maintenance, proven equal to from-scratch
  (`docs/proofs.md`, `src/streamdiam/maintain/`, `baselines/maintainers.py`).
- **C4 — Decoupled drift-triggered retraining.** Faithful DDM / ADWIN /
  Page-Hinkley detectors, a label-latency-aware monitor, and a background trainer
  that fine-tunes a copy and hot-swaps it in (`src/streamdiam/drift/`,
  `src/streamdiam/retrain/`, `docs/drift_design.md`, `docs/retrain_design.md`).
- **C5 — Rigorous streaming evaluation.** One prequential test-then-train harness,
  imbalance-aware metrics, latency/throughput/memory, real baselines, committed
  results, and figures that can only plot committed numbers
  (`src/streamdiam/eval/`, `docs/eval_design.md`).

## Headline results (committed, real Elliptic)

- **Incremental == from-scratch** at every tick of the stream (ε = float
  summation order) — `test_baselines.py`.
- **Faster and feasible at scale (RQ7).** Update-latency speedup grows with the
  graph: ~6.7× at 3.2k accounts, ~14× at 6.4k, ~20× at 10.6k; from-scratch runs
  out of memory at ~32k while the incremental maintainer continues
  (`results/rq7_scalability/`, `figures/fig_latency_scaling`).
- **Retrain rarely (RQ3).** Drift-triggering fires far less often than a fixed
  periodic schedule at comparable accuracy — the empirical core of the reframing.

## Quickstart

```bash
pip install -e .[dev]
export STREAMDIAM_ELLIPTIC_ROOT=/path/to/elliptic_bitcoin_dataset
make check                 # lint + type + full test suite
make experiments figures   # reproduce results/ and figures (subsample; see repro note)
```

See `reproducibility/README.md` for the full artifact and
`reproducibility/claims_to_scripts.md` for the claim→command map.

## Repository layout

```
src/streamdiam/
  data/        schema, Elliptic loader, temporal split, stream replay
  store/       temporal multigraph store (C2) + reference oracle
  model/       DIAM detector (Edge2Seq + MGD + head), reused faithfully
  maintain/    from-scratch oracle + incremental Edge2Seq/MGD (C3)
  drift/       detectors, monitor, triggers (C4)
  retrain/     recency weighting, background trainer, hot-swap (C4)
  eval/        prequential protocol, metrics, runner (C5)
  baselines/   maintenance strategies + XGB-Graph baseline
  experiments/ RQ drivers that emit results/*.jsonl
  figures/     pure consumers of results/ (no data generation)
  utils/       seeding, timing, memory
docs/          formalization, proofs, per-component design, threats to validity
reproducibility/  artifact: claims→scripts, env, Dockerfile, hardware
results/       committed numbers behind every figure
tests/         correctness, equivalence, no-leakage, anti-fabrication guard
```

## Scope and honesty

The maintainers ship as a **pure-Python reference** — correct and portable, and
enough to demonstrate the scaling trend. Full 203k-account headline numbers need
the optional native (C++/pybind11) store/maintainer hot path, noted as future
work; we report what the reference measures and never extrapolate unrun numbers.
Baselines are real and runnable through the one harness — never mocked; external
models still being wired (TGN/APAN/Spade) appear in no results table until their
real implementations run. See `docs/threat_to_validity.md`.

## License

MIT — see `LICENSE`.
