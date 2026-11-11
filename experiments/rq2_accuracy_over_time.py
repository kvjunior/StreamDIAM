"""RQ2 -- does maintenance keep accuracy over time, and does retraining help?

Holds the maintenance strategy fixed (incremental) and varies the retrain policy:
``never`` (frozen weights after warmup), ``drift`` (our decoupled trigger), and
``periodic`` (full-retrain on a schedule). The results file records the per-tick
F1/AP-over-time timeline for each -- the paper's headline RQ2 curve -- all from the
one harness, which is what kills the old paper's F1 contradiction (R2 O7).
"""

from __future__ import annotations

from pathlib import Path

from streamdiam.experiments.harness import Dataset, run_strategy, save_result

__all__ = ["run"]

_POLICIES = ["never", "drift", "periodic"]


def run(dataset: Dataset, out: str | Path, *, epochs: int = 15) -> dict[str, dict]:
    """Run RQ2 and return a compact summary keyed by retrain policy."""
    summary: dict[str, dict] = {}
    for policy in _POLICIES:
        result = run_strategy(
            dataset, maintainer="incremental", retrain_policy=policy, epochs=epochs
        )
        save_result(result, out, "rq2_accuracy_over_time", dataset.name, policy)
        defined_ap = [t.window.ap for t in result.ticks if t.window.ap == t.window.ap]
        summary[policy] = {
            "n_ticks": len(result.ticks),
            "n_retrains": result.n_retrains,
            "retrain_times": result.retrain_times,
            "prequential_ap": result.prequential.ap,
            "mean_window_ap": (sum(defined_ap) / len(defined_ap)) if defined_ap else float("nan"),
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
