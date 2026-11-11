"""Anti-fabrication guard (the single most important fix for the old rejection).

The original submission plotted fabricated/synthetic data. This test makes that
structurally impossible to reintroduce:

1. no figure module may call a data-generation primitive (RNG, sklearn
   ``make_*``), so a figure cannot invent numbers;
2. no figure module may import the run/model machinery, so a figure cannot run a
   model to produce numbers; and
3. a figure, given a real run's committed results file, plots back exactly those
   committed numbers -- verified by comparing the figure's own data loader to the
   raw file.

If any of these breaks, this test fails and the fabrication path is caught.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FIG_DIR = Path(__file__).resolve().parents[1] / "src" / "streamdiam" / "figures"

ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")

# Data-generation primitives a figure must never use.
_FORBIDDEN_CALLS = [
    "np.random",
    "numpy.random",
    "default_rng",
    "torch.rand",
    "torch.randn",
    "make_classification",
    "make_blobs",
    "sklearn.datasets",
]
# A figure may only import from the figures package (plus plotting/std libs).
_FORBIDDEN_IMPORTS = re.compile(
    r"import\s+streamdiam\.(?!figures)"
    r"|from\s+streamdiam\.(?!figures)"
)


def _figure_sources() -> list[Path]:
    return [p for p in FIG_DIR.glob("*.py")]


def test_figures_generate_no_data() -> None:
    for path in _figure_sources():
        text = path.read_text()
        for token in _FORBIDDEN_CALLS:
            assert token not in text, f"{path.name} uses data-generation call {token!r}"


def test_figures_do_not_import_run_machinery() -> None:
    for path in _figure_sources():
        text = path.read_text()
        bad = _FORBIDDEN_IMPORTS.search(text)
        assert bad is None, f"{path.name} imports non-figures streamdiam module: {bad.group() if bad else ''}"


@needs_elliptic
def test_figure_plots_only_committed_values(tmp_path) -> None:
    from streamdiam.experiments import load_dataset, run_strategy, save_result
    from streamdiam.figures import load_run, tick_series
    from streamdiam.figures.fig_accuracy_over_time import make

    ds = load_dataset("elliptic", ELLIPTIC_ROOT, per_step=30)
    result = run_strategy(ds, maintainer="incremental", retrain_policy="drift", epochs=4)
    path = save_result(result, tmp_path, "rq2_accuracy_over_time", ds.name, "drift")

    # The figure's loader returns exactly what is in the committed file.
    raw = [json.loads(ln) for ln in path.read_text().strip().splitlines()]
    raw_ticks = [r for r in raw if r["record"] == "tick"]
    run = load_run(path)
    assert len(run.ticks) == len(raw_ticks)
    _, ap = tick_series(run.ticks, "ap")
    raw_ap = [r["window_ap"] for r in raw_ticks]
    both = np.array(ap)
    assert np.array_equal(np.isnan(both), np.isnan(np.array(raw_ap)))
    defined = ~np.isnan(both)
    assert np.allclose(both[defined], np.array(raw_ap)[defined])

    # And the figure renders to a real PNG from that file alone.
    out = make(path, tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
