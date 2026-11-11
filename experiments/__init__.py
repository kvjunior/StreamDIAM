"""Experiment drivers (one per research question).

Each driver builds its runs from ``experiments.harness`` and writes JSONL results
under ``results/<rq>/<dataset>/``. Drivers never plot; figures read the committed
results. Run e.g. ``python -m streamdiam.experiments.rq1_maintenance_speed
--root <elliptic_dir>``.
"""

from streamdiam.experiments.harness import (
    Dataset,
    load_dataset,
    run_strategy,
    save_result,
)

__all__ = ["Dataset", "load_dataset", "run_strategy", "save_result"]
