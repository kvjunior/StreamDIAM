"""Tests for the experiment drivers (B13).

These run the RQ drivers on a small real Elliptic subsample and check the claims
that must hold by construction: incremental and from-scratch reach *identical*
prequential accuracy (same view), results are written as valid JSONL, and the
drift/periodic policies log their retrains. The headline latency comparison is
scale-dependent and is measured at scale, not asserted here.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamdiam.experiments import load_dataset  # noqa: E402
from streamdiam.experiments import (  # noqa: E402
    rq1_maintenance_speed,
    rq2_accuracy_over_time,
    rq3_drift_trigger,
    rq7_scalability,
)

ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")


@functools.lru_cache(maxsize=1)
def _dataset():
    # Small, timestep-spanning subsample for a fast but real run.
    return load_dataset("elliptic", ELLIPTIC_ROOT, per_step=30)


def _valid_jsonl(path: Path) -> list[dict]:
    recs = [json.loads(ln) for ln in path.read_text().strip().splitlines()]
    assert recs[0]["record"] == "summary"
    assert all(r["record"] == "tick" for r in recs[1:])
    return recs


@needs_elliptic
def test_rq1_incremental_matches_from_scratch_accuracy(tmp_path) -> None:
    summary = rq1_maintenance_speed.run(_dataset(), tmp_path, epochs=4)
    assert set(summary) == {"from_scratch", "incremental", "frozen"}
    # Identical view => identical strict prequential accuracy.
    assert abs(summary["incremental"]["prequential_ap"] - summary["from_scratch"]["prequential_ap"]) < 1e-6
    assert abs(summary["incremental"]["prequential_f1"] - summary["from_scratch"]["prequential_f1"]) < 1e-9
    # latency recorded for every strategy
    for strat in summary:
        assert summary[strat]["latency_mean_ms"] >= 0.0
    # results files exist and parse
    for strat in ("from_scratch", "incremental", "frozen"):
        p = tmp_path / "rq1_maintenance_speed" / "elliptic" / f"{strat}.jsonl"
        recs = _valid_jsonl(p)
        assert len(recs) - 1 == summary[strat]["n_ticks"]


@needs_elliptic
def test_rq2_policies_run_and_log(tmp_path) -> None:
    summary = rq2_accuracy_over_time.run(_dataset(), tmp_path, epochs=4)
    assert set(summary) == {"never", "drift", "periodic"}
    assert summary["never"]["n_retrains"] == 0
    for policy in ("never", "drift", "periodic"):
        p = tmp_path / "rq2_accuracy_over_time" / "elliptic" / f"{policy}.jsonl"
        _valid_jsonl(p)


@needs_elliptic
def test_rq3_drift_vs_periodic(tmp_path) -> None:
    summary = rq3_drift_trigger.run(_dataset(), tmp_path, epochs=4, periodic_period=5)
    assert set(summary) == {"drift", "periodic"}
    # periodic fires on a fixed schedule; both retrain counts are logged, not estimated
    assert summary["periodic"]["n_retrains"] >= 1
    assert isinstance(summary["drift"]["n_retrains"], int)
    for label in ("drift", "periodic"):
        p = tmp_path / "rq3_drift_trigger" / "elliptic" / f"{label}.jsonl"
        _valid_jsonl(p)


@needs_elliptic
def test_rq7_scalability_and_figure(tmp_path) -> None:
    from streamdiam.figures.fig_latency_scaling import make

    pts = rq7_scalability.run(ELLIPTIC_ROOT, tmp_path, sizes=[40, 80])
    assert all(p["record"] == "scaling_point" for p in pts)
    assert any(p["maintainer"] == "incremental" and p["status"] == "ok" for p in pts)
    path = tmp_path / "rq7_scalability" / "elliptic" / "scaling.jsonl"
    assert path.exists()
    out = make(path, tmp_path / "scaling.png")
    assert out.exists() and out.stat().st_size > 0


@needs_elliptic
def test_rq7_marks_from_scratch_infeasible(tmp_path) -> None:
    # Force the infeasibility path: cap from-scratch below any real N.
    pts = rq7_scalability.run(ELLIPTIC_ROOT, tmp_path, sizes=[40], from_scratch_max_n=0)
    fs = [p for p in pts if p["maintainer"] == "from_scratch"]
    assert fs and fs[0]["status"] == "infeasible_memory"
    assert fs[0]["latency_mean_ms"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
