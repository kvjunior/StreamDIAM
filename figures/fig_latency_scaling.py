"""RQ7 figure: update latency vs. graph size, from a committed scaling file only.

Plots per-tick update latency against the number of accounts for the from-scratch
oracle and the incremental maintainer, on log-log axes, so the diverging slopes
show the growing speedup. Points where from-scratch was infeasible (out of
memory) are marked on the axis rather than invented. Generates no data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from streamdiam.figures._io import load_scaling  # noqa: E402

__all__ = ["make"]

_COLORS = {"from_scratch": "#d62728", "incremental": "#1f77b4"}


def make(scaling_path: str | Path, out_path: str | Path) -> Path:
    """Plot latency-vs-size from ``scaling_path`` to ``out_path`` (PNG)."""
    points = load_scaling(scaling_path)
    fig, ax = plt.subplots(figsize=(7, 4))

    for strat in ("from_scratch", "incremental"):
        pts = [p for p in points if p["maintainer"] == strat]
        xs = [p["n"] for p in pts if p.get("status") == "ok"]
        ys = [p["latency_mean_ms"] for p in pts if p.get("status") == "ok"]
        if xs:
            ax.plot(xs, ys, "o-", color=_COLORS[strat], label=strat)
        # mark infeasible from-scratch sizes
        for p in pts:
            if p.get("status") != "ok":
                ax.axvline(p["n"], color=_COLORS[strat], linestyle=":", linewidth=0.8)
                ax.text(p["n"], ax.get_ylim()[0], " from-scratch OOM",
                        rotation=90, va="bottom", ha="right", fontsize=7,
                        color=_COLORS[strat])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of accounts (N)")
    ax.set_ylabel("mean update latency (ms/tick)")
    ax.set_title("Maintenance latency vs. graph size")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("scaling", help="path to rq7 scaling.jsonl")
    ap.add_argument("--out", default="figures_out/latency_scaling.png")
    args = ap.parse_args()
    print(make(args.scaling, args.out))
