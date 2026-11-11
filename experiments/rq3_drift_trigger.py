"""RQ3 -- how often is retraining actually needed, and what does deferral cost?

This is the empirical heart of the reframing (answers R1 O1 / R2 O8). It contrasts
drift-triggered retraining against a fixed periodic schedule on the same stream:
how many times each fires, when, and the resulting accuracy. If drift-triggering
fires far less often than a sensible periodic schedule while holding comparable
accuracy, the paper's thesis -- "mostly you don't need to retrain" -- is
demonstrated, not asserted.

The firing counts and times come straight from the run's retrain log; nothing is
estimated.
"""

from __future__ import annotations

from pathlib import Path

from streamdiam.experiments.harness import Dataset, run_strategy, save_result

__all__ = ["run"]


def run(
    dataset: Dataset, out: str | Path, *, epochs: int = 15, periodic_period: int = 5
) -> dict[str, dict]:
    """Run RQ3 (drift vs. periodic) and return a compact summary."""
    summary: dict[str, dict] = {}
    runs = {
        "drift": {"retrain_policy": "drift"},
        "periodic": {"retrain_policy": "periodic", "config_overrides": {"retrain_period": periodic_period}},
    }
    for label, kwargs in runs.items():
        result = run_strategy(
            dataset, maintainer="incremental", epochs=epochs, **kwargs  # type: ignore[arg-type]
        )
        save_result(result, out, "rq3_drift_trigger", dataset.name, label)
        summary[label] = {
            "n_retrains": result.n_retrains,
            "retrain_times": result.retrain_times,
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
