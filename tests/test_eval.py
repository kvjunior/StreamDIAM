"""Tests for the eval harness (C5).

Metrics and timing are checked on exact synthetic cases. The protocol and runner
are checked on a real Elliptic subgraph: a timeline is produced, the strict
no-future-leakage invariant holds (a tick is scored by a model trained strictly
before it), frozen vs. retraining behave as configured, and results round-trip
through JSONL.
"""

from __future__ import annotations

import functools
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamdiam.data import TransactionBatch, load_elliptic  # noqa: E402
from streamdiam.eval import (  # noqa: E402
    PrequentialAccumulator,
    RunConfig,
    compute_classification_metrics,
    run_streaming,
    write_run_jsonl,
)
from streamdiam.model import DIAM  # noqa: E402
from streamdiam.retrain import TrainConfig  # noqa: E402
from streamdiam.utils.memory import current_rss_bytes, peak_rss_bytes  # noqa: E402
from streamdiam.utils.timing import LatencyRecorder, Stopwatch  # noqa: E402

ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_metrics_perfect_separation() -> None:
    yt = np.array([0, 0, 1, 1])
    ys = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_classification_metrics(yt, ys)
    assert m.f1 == 1.0 and m.precision == 1.0 and m.recall == 1.0
    assert m.roc_auc == 1.0 and m.ap == 1.0
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 0, 0, 2)


def test_metrics_known_confusion() -> None:
    # threshold 0.5: preds [1,1,0,0], truth [1,0,1,0] -> tp1 fp1 fn1 tn1
    m = compute_classification_metrics(
        np.array([1, 0, 1, 0]), np.array([0.9, 0.6, 0.4, 0.1])
    )
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    assert m.precision == 0.5 and m.recall == 0.5 and m.f1 == 0.5


def test_metrics_degenerate_cases() -> None:
    empty = compute_classification_metrics(np.array([], dtype=int), np.array([]))
    assert empty.n == 0 and np.isnan(empty.ap)
    allneg = compute_classification_metrics(np.array([0, 0]), np.array([0.2, 0.8]))
    assert np.isnan(allneg.roc_auc) and np.isnan(allneg.ap) and allneg.f1 == 0.0


def test_prequential_accumulator() -> None:
    acc = PrequentialAccumulator()
    for lab, sc in [(1, 0.9), (0, 0.1), (1, 0.8), (0, 0.2)]:
        acc.add(lab, sc)
    assert len(acc) == 4 and acc.metrics().ap == 1.0


# --------------------------------------------------------------------------- #
# timing / memory
# --------------------------------------------------------------------------- #
def test_latency_summary_and_throughput() -> None:
    rec = LatencyRecorder()
    for s in [0.001, 0.002, 0.003, 0.004]:
        rec.record(s)
    summ = rec.summary()
    assert summ.count == 4
    assert abs(summ.mean_ms - 2.5) < 1e-9
    assert summ.max_ms == 4.0
    assert rec.throughput_eps == pytest.approx(4 / 0.010)


def test_stopwatch_measures() -> None:
    with Stopwatch() as sw:
        sum(range(10000))
    assert sw.elapsed >= 0.0


def test_memory_probes() -> None:
    assert current_rss_bytes() > 0
    assert peak_rss_bytes() > 0


# --------------------------------------------------------------------------- #
# real Elliptic streaming run
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _load_graph():
    return load_elliptic(ELLIPTIC_ROOT)


def _subgraph(n_edges: int = 2500):
    graph = _load_graph()
    e = graph.edges
    src = np.asarray(e.src)[:n_edges]
    dst = np.asarray(e.dst)[:n_edges]
    t = np.asarray(e.timestamp)[:n_edges]
    attr = np.asarray(e.attr)[:n_edges]
    inv = np.unique(np.concatenate([src, dst]))
    remap = -np.ones(graph.num_accounts, dtype=np.int64)
    remap[inv] = np.arange(inv.size)
    batch = TransactionBatch.from_arrays(np.arange(n_edges), remap[src], remap[dst], t, attr)
    labels = graph.account_label[inv].copy()
    return batch, labels, int(inv.size), graph.num_features


@needs_elliptic
def test_run_streaming_produces_timeline() -> None:
    edges, labels, universe, nfeat = _subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1)
    cfg = RunConfig(warmup_fraction=0.3, train_config=TrainConfig(epochs=10), cooldown=3)
    result = run_streaming(edges, labels, universe, model, cfg)
    assert len(result.ticks) > 0
    assert result.latency.count == len(result.ticks)  # one update per tick
    assert result.prequential.n > 0  # strict first-appearance decisions recorded
    assert all(tr.n_active > 0 for tr in result.ticks)


@needs_elliptic
def test_no_future_leakage_invariant() -> None:
    """A tick's score comes from a model whose version only advances *after* a
    retrain tick -- i.e. the model that scored tick t was trained before t."""
    edges, labels, universe, nfeat = _subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1)
    cfg = RunConfig(warmup_fraction=0.2, train_config=TrainConfig(epochs=8), cooldown=2)
    result = run_streaming(edges, labels, universe, model, cfg)
    ticks = result.ticks
    for i in range(1, len(ticks)):
        if ticks[i].model_version != ticks[i - 1].model_version:
            # the version used to score tick i rose only because the PREVIOUS
            # tick retrained and swapped afterwards -- never from tick i's data.
            assert ticks[i - 1].retrained
            assert ticks[i].model_version > ticks[i - 1].model_version


@needs_elliptic
def test_frozen_run_never_retrains() -> None:
    edges, labels, universe, nfeat = _subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1)
    cfg = RunConfig(warmup_fraction=0.3, train_config=TrainConfig(epochs=8))
    result = run_streaming(edges, labels, universe, model, cfg, enable_retrain=False)
    assert result.n_retrains == 0
    assert all(not tr.retrained for tr in result.ticks)


@needs_elliptic
def test_results_round_trip_through_jsonl(tmp_path) -> None:
    edges, labels, universe, nfeat = _subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1)
    cfg = RunConfig(warmup_fraction=0.3, train_config=TrainConfig(epochs=5), retrain_policy="never")
    result = run_streaming(edges, labels, universe, model, cfg, enable_retrain=False)
    path = write_run_jsonl(result, tmp_path / "rq" / "run.jsonl")
    lines = path.read_text().strip().splitlines()
    recs = [json.loads(ln) for ln in lines]
    assert recs[0]["record"] == "summary"
    assert all(r["record"] == "tick" for r in recs[1:])
    assert len(recs) == 1 + len(result.ticks)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
