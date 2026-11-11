"""RQ7 -- does the maintainer scale, and where does from-scratch stop being viable?

Sweeps graph size and measures per-tick update latency for the from-scratch
oracle versus the incremental maintainer (C3), with fixed random weights and no
warmup -- pure maintenance cost. The point of the paper's efficiency story is not
a single speedup number but a **trend**: the incremental maintainer does work
bounded by the delta's affected set, so its advantage grows with graph size, and
it stays feasible at scales where recomputing the whole view exhausts memory.

From-scratch is skipped above ``from_scratch_max_n`` (it materialises a dense
padded tensor over all accounts and runs out of memory on large graphs); that
skip is itself a result -- recorded, not hidden -- and the incremental maintainer
continues past it.
"""

from __future__ import annotations

import json
from pathlib import Path

from streamdiam.experiments.harness import load_dataset, run_strategy

__all__ = ["run"]


def run(
    root: str,
    out: str | Path,
    *,
    sizes: list[int] | None = None,
    from_scratch_max_n: int = 12_000,
    dataset: str = "elliptic",
) -> list[dict]:
    """Sweep ``per_step`` sizes; return and save one scaling point per (size, maintainer)."""
    sizes = sizes or [60, 120, 200]
    points: list[dict] = []
    for per_step in sizes:
        ds = load_dataset(dataset, root, per_step=per_step)
        n = ds.universe_size
        for strat in ("from_scratch", "incremental"):
            if strat == "from_scratch" and n > from_scratch_max_n:
                points.append({
                    "record": "scaling_point", "maintainer": strat,
                    "n": n, "edges": len(ds.edges), "status": "infeasible_memory",
                    "latency_mean_ms": None, "latency_p99_ms": None,
                })
                continue
            result = run_strategy(
                ds, maintainer=strat, retrain_policy="never",
                warmup_fraction=0.0, epochs=0,
            )
            points.append({
                "record": "scaling_point", "maintainer": strat,
                "n": n, "edges": len(ds.edges), "status": "ok",
                "latency_mean_ms": result.latency.mean_ms,
                "latency_p99_ms": result.latency.p99_ms,
            })

    path = Path(out) / "rq7_scalability" / dataset / "scaling.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for pt in points:
            fh.write(json.dumps(pt) + "\n")
    return points


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Elliptic CSV directory")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--from-scratch-max-n", type=int, default=12_000)
    args = ap.parse_args()
    pts = run(args.root, args.out, sizes=args.sizes, from_scratch_max_n=args.from_scratch_max_n)
    print(json.dumps(pts, indent=2))
