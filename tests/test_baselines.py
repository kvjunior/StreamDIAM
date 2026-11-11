"""Tests for the maintenance strategies and retrain policies (B12).

The headline is streaming equivalence: the incremental maintainer (C3) produces
the same scores as the from-scratch oracle at *every* tick of a real Elliptic
stream, not merely per isolated delta -- so RQ1 can claim "incremental is faster
*and* identical". Frozen-view staleness, the periodic trigger, and the
full-retrain run mode are also checked.
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamdiam.baselines import (  # noqa: E402
    FrozenViewMaintainer,
    IncrementalStreamingMaintainer,
)
from streamdiam.data import TransactionBatch, load_elliptic, replay_by_time  # noqa: E402
from streamdiam.drift import PeriodicTrigger, RetrainSignal  # noqa: E402
from streamdiam.eval import FromScratchStreamingMaintainer, RunConfig, run_streaming  # noqa: E402
from streamdiam.model import DIAM  # noqa: E402
from streamdiam.retrain import TrainConfig  # noqa: E402
from streamdiam.store import ReferencePyStore  # noqa: E402

ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")


@functools.lru_cache(maxsize=1)
def _load_graph():
    return load_elliptic(ELLIPTIC_ROOT)


def _subgraph(per_step: int = 60):
    """Sample up to ``per_step`` edges from *every* timestep, so the stream spans
    all ~49 Elliptic ticks (needed to exercise multi-tick streaming behaviour),
    then remap to a compact labelled universe."""
    graph = _load_graph()
    e = graph.edges
    ts = np.asarray(e.timestamp)
    _, starts, counts = np.unique(ts, return_index=True, return_counts=True)
    sel = np.concatenate(
        [np.arange(s, s + min(per_step, c)) for s, c in zip(starts, counts)]
    )
    src = np.asarray(e.src)[sel]
    dst = np.asarray(e.dst)[sel]
    t = ts[sel]
    attr = np.asarray(e.attr)[sel]
    inv = np.unique(np.concatenate([src, dst]))
    remap = -np.ones(graph.num_accounts, dtype=np.int64)
    remap[inv] = np.arange(inv.size)
    batch = TransactionBatch.from_arrays(
        np.arange(sel.size), remap[src], remap[dst], t, attr
    )
    labels = graph.account_label[inv].copy()
    return batch, labels, int(inv.size), graph.num_features


# --------------------------------------------------------------------------- #
# Streaming equivalence: incremental == from-scratch at every tick
# --------------------------------------------------------------------------- #
@needs_elliptic
def test_incremental_equals_from_scratch_every_tick() -> None:
    edges, labels, universe, nfeat = _subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=2).eval()

    store_fs = ReferencePyStore(nfeat)
    store_inc = ReferencePyStore(nfeat)
    fs = FromScratchStreamingMaintainer(lambda: model, store_fs, universe)
    inc = IncrementalStreamingMaintainer(lambda: model, store_inc, universe)
    fs.rebuild()
    inc.rebuild()

    max_gap = 0.0
    for delta in replay_by_time(edges):
        store_fs.insert(delta)
        fs.ingest(delta)
        store_inc.insert(delta)
        inc.ingest(delta)
        gap = float(np.max(np.abs(fs.scores() - inc.scores())))
        max_gap = max(max_gap, gap)
    assert max_gap < 1e-4, max_gap  # identical up to float summation order


@needs_elliptic
def test_incremental_harness_matches_from_scratch_timeline() -> None:
    edges, labels, universe, nfeat = _subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=2)
    cfg_fs = RunConfig(maintainer="from_scratch", warmup_fraction=0.2,
                       train_config=TrainConfig(epochs=5))
    cfg_inc = RunConfig(maintainer="incremental", warmup_fraction=0.2,
                        train_config=TrainConfig(epochs=5))
    # Fixed weights (no retrain) so the two strategies see identical models.
    import copy
    r_fs = run_streaming(edges, labels, universe, copy.deepcopy(model), cfg_fs, enable_retrain=False)
    r_inc = run_streaming(edges, labels, universe, copy.deepcopy(model), cfg_inc, enable_retrain=False)
    assert r_fs.maintainer == "from_scratch" and r_inc.maintainer == "incremental"
    assert len(r_fs.ticks) == len(r_inc.ticks)
    for a, b in zip(r_fs.ticks, r_inc.ticks):
        # window AP is nan on tickless-positive windows; compare where defined
        if not (np.isnan(a.window.ap) or np.isnan(b.window.ap)):
            assert abs(a.window.ap - b.window.ap) < 1e-4
        assert abs(a.window.f1 - b.window.f1) < 1e-4


# --------------------------------------------------------------------------- #
# Frozen-view ablation
# --------------------------------------------------------------------------- #
@needs_elliptic
def test_frozen_view_is_stale() -> None:
    edges, labels, universe, nfeat = _subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1).eval()
    store = ReferencePyStore(nfeat)
    frozen = FrozenViewMaintainer(lambda: model, store, universe)
    # Seed with a prefix, then grow the graph without rebuilding.
    deltas = list(replay_by_time(edges))
    store.insert(deltas[0])
    frozen.rebuild()
    before = frozen.scores().copy()
    for d in deltas[1:]:
        store.insert(d)
        frozen.ingest(d)  # no-op by design
    assert np.array_equal(before, frozen.scores())  # scores never moved: stale


# --------------------------------------------------------------------------- #
# Periodic trigger
# --------------------------------------------------------------------------- #
def test_periodic_trigger_fires_on_schedule() -> None:
    trig = PeriodicTrigger(period=4)
    fired_at = [i for i in range(12) if isinstance(trig.step(stream_time=i), RetrainSignal)]
    assert fired_at == [4, 8]  # ticks 4 and 8, never tick 0
    assert trig.firing_count == 2
    assert all(ev.reason == "periodic" for ev in trig.events)


def test_periodic_trigger_validation() -> None:
    with pytest.raises(ValueError):
        PeriodicTrigger(period=0)


@needs_elliptic
def test_periodic_retrain_run_mode() -> None:
    edges, labels, universe, nfeat = _subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1)
    cfg = RunConfig(
        maintainer="from_scratch", warmup_fraction=0.2, retrain_policy="periodic",
        retrain_period=5, train_config=TrainConfig(epochs=5),
    )
    result = run_streaming(edges, labels, universe, model, cfg)
    assert result.n_retrains >= 1  # the fixed schedule fired at least once


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
