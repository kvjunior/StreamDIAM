"""Figures -- pure consumers of committed ``results/*.jsonl``.

Every figure reads numbers through ``figures._io`` and generates nothing itself:
no model, no RNG, no run. This is the structural fix for the original rejection's
fabricated figures, enforced by ``tests/test_no_synthetic_in_results.py``. A
number reaches a figure only by having been written to a results file by a real
experiment run.
"""

from streamdiam.figures._io import RunRecords, load_run, tick_series

__all__ = ["RunRecords", "load_run", "tick_series"]
