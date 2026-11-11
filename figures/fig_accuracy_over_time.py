"""RQ2 figure: F1/AP over time, from a committed results file only.

Reads one run's JSONL via ``figures._io`` and plots the per-tick window AP and F1
against stream time, marking the ticks where a retrain fired. It generates no
data: every point is a committed number. Produces a PNG and returns its path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt  # noqa: E402

from streamdiam.figures._io import load_run, tick_series  # noqa: E402

__all__ = ["make"]


def make(results_path: str | Path, out_path: str | Path) -> Path:
    """Plot accuracy-over-time from ``results_path`` to ``out_path`` (PNG)."""
    run = load_run(results_path)
    t_ap, ap = tick_series(run.ticks, "ap")
    t_f1, f1 = tick_series(run.ticks, "f1")
    retrain_times = run.summary.get("retrain_times", [])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_f1, f1, label="F1", color="#1f77b4", linewidth=1.5)
    ax.plot(t_ap, ap, label="AP", color="#d62728", linewidth=1.5)
    for i, rt in enumerate(retrain_times):
        ax.axvline(
            rt, color="#888888", linestyle="--", linewidth=0.8,
            label="retrain" if i == 0 else None,
        )
    ax.set_xlabel("stream time")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Accuracy over time ({run.summary.get('maintainer', '?')})")
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
    ap.add_argument("results", help="path to a run .jsonl file")
    ap.add_argument("--out", default="figures_out/accuracy_over_time.png")
    args = ap.parse_args()
    print(make(args.results, args.out))
