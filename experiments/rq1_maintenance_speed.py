"""RQ1 -- is maintenance faster than from-scratch, at equal accuracy?

Runs the three maintenance strategies (from-scratch, incremental, frozen) on the
same stream with fixed weights (no retrain), so differences are purely the cost
and fidelity of keeping the view fresh. The results file records per-tick update
latency and the strict prequential accuracy for each.

Two honest claims come out of this:

* **Identical accuracy.** incremental must match from-scratch's prequential score
  (they compute the same view). The test asserts this.
* **Latency.** incremental does bounded per-delta work; from-scratch recomputes
  the whole view. The *size* of the win is scale-dependent -- on a small graph,
  batched tensor recompute can beat a Python per-affected-node loop, so the
  headline latency number is a large-graph result, reported as measured, not
  assumed. The driver records the latency either way.
"""

from __future__ import annotations

from pathlib import Path

from streamdiam.experiments.harness import Dataset, run_strategy, save_result

__all__ = ["run"]

_STRATEGIES = ["from_scratch", "incremental", "frozen"]


def run(dataset: Dataset, out: str | Path, *, epochs: int = 15) -> dict[str, dict]:
    """Run RQ1 and return a compact summary keyed by maintainer."""
    summary: dict[str, dict] = {}
    for strat in _STRATEGIES:
        result = run_strategy(
            dataset, maintainer=strat, retrain_policy="never", epochs=epochs
        )
        save_result(result, out, "rq1_maintenance_speed", dataset.name, strat)
        summary[strat] = {
            "n_ticks": len(result.ticks),
            "latency_mean_ms": result.latency.mean_ms,
            "latency_p99_ms": result.latency.p99_ms,
            "throughput_eps": result.throughput_eps,
            "prequential_ap": result.prequential.ap,
            "prequential_f1": result.prequential.f1,
        }
    return summary


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    from streamdiam.experiments.harness import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Elliptic CSV directory")
    ap.add_argument("--out", default="results")
    ap.add_argument("--per-step", type=int, default=None)
    args = ap.parse_args()
    ds = load_dataset("elliptic", args.root, per_step=args.per_step)
    print(json.dumps(run(ds, args.out), indent=2))
