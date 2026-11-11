"""Read committed run results for plotting.

Figures are **pure consumers** of ``results/*.jsonl``. This module is the only way
they get data: there is no model, no RNG, no run here -- just parsing numbers that
a committed experiment already produced. The anti-fabrication guard
(``tests/test_no_synthetic_in_results.py``) enforces that every figure module goes
through here and generates nothing of its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["RunRecords", "load_run", "tick_series", "load_scaling"]


@dataclass(frozen=True)
class RunRecords:
    """Parsed contents of one run's JSONL file."""

    summary: dict
    ticks: list[dict]


def load_run(path: str | Path) -> RunRecords:
    """Parse a ``<run>.jsonl`` file (one summary record then one per tick)."""
    lines = [json.loads(ln) for ln in Path(path).read_text().strip().splitlines()]
    if not lines or lines[0].get("record") != "summary":
        raise ValueError(f"{path}: first record must be the run summary")
    ticks = [r for r in lines[1:] if r.get("record") == "tick"]
    return RunRecords(summary=lines[0], ticks=ticks)


def tick_series(ticks: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Extract ``(t, window_<key>)`` from tick records, verbatim.

    No smoothing, imputation, or synthesis -- the returned values are exactly the
    committed numbers, which is the property the guard test checks.
    """
    ts: list[int] = []
    vals: list[float] = []
    field = f"window_{key}"
    for row in ticks:
        ts.append(int(row["t"]))
        vals.append(float(row[field]))
    return ts, vals


def load_scaling(path: str | Path) -> list[dict]:
    """Parse a RQ7 scaling file (one ``scaling_point`` record per line)."""
    out: list[dict] = []
    for ln in Path(path).read_text().strip().splitlines():
        rec = json.loads(ln)
        if rec.get("record") == "scaling_point":
            out.append(rec)
    return out
