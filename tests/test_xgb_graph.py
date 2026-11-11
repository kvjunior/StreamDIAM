"""Tests for the XGB-Graph baseline (B14).

A real gradient-boosted-tree detector run through the same prequential harness --
no mock. Checks that it runs end to end on real Elliptic, actually fits (version
advances), and separates the classes above chance on the training labels.
"""

from __future__ import annotations

import functools
import os
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamdiam.baselines import XGBGraphMaintainer, run_xgb_graph  # noqa: E402
from streamdiam.data import TransactionBatch, load_elliptic  # noqa: E402
from streamdiam.store import ReferencePyStore  # noqa: E402

ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")


@functools.lru_cache(maxsize=1)
def _load_graph():
    return load_elliptic(ELLIPTIC_ROOT)


def _subgraph(per_step: int = 60):
    graph = _load_graph()
    e = graph.edges
    ts = np.asarray(e.timestamp)
    _, starts, counts = np.unique(ts, return_index=True, return_counts=True)
    sel = np.concatenate([np.arange(s, s + min(per_step, c)) for s, c in zip(starts, counts)])
    src = np.asarray(e.src)[sel]
    dst = np.asarray(e.dst)[sel]
    inv = np.unique(np.concatenate([src, dst]))
    remap = -np.ones(graph.num_accounts, dtype=np.int64)
    remap[inv] = np.arange(inv.size)
    batch = TransactionBatch.from_arrays(
        np.arange(sel.size), remap[src], remap[dst], ts[sel], np.asarray(e.attr)[sel]
    )
    labels = graph.account_label[inv].copy()
    return batch, labels, int(inv.size), graph.num_features


@needs_elliptic
def test_xgb_graph_runs_through_harness() -> None:
    edges, labels, universe, nfeat = _subgraph()
    result = run_xgb_graph(edges, labels, universe, nfeat, retrain_policy="drift")
    assert result.maintainer == "xgb_graph"
    assert len(result.ticks) > 0
    assert result.prequential.n > 0
    assert result.latency.count == len(result.ticks)


@needs_elliptic
def test_xgb_graph_periodic_retrains() -> None:
    edges, labels, universe, nfeat = _subgraph()
    result = run_xgb_graph(
        edges, labels, universe, nfeat, retrain_policy="periodic", retrain_period=5
    )
    assert result.n_retrains >= 1


@needs_elliptic
def test_xgb_graph_fits_and_separates_classes() -> None:
    """Trained on the labelled accounts, the tree beats chance on train AP."""
    edges, labels, universe, nfeat = _subgraph()
    store = ReferencePyStore(nfeat)
    store.insert(edges)
    m = XGBGraphMaintainer(store, universe, labels, seed=0)
    m.rebuild()
    seen = np.zeros(universe, dtype=bool)
    seen[np.unique(np.concatenate([edges.src, edges.dst]))] = True
    version = m.retrain(seen, ref_time=float(edges.timestamp[-1]))
    assert version == 1  # fit happened
    scores = m.scores()
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    mask = labels >= 0
    ap = average_precision_score((labels[mask] == 1).astype(int), scores[mask])
    base_rate = float((labels[mask] == 1).mean())
    assert ap > base_rate  # better than predicting the prior everywhere


@needs_elliptic
def test_xgb_graph_no_fit_with_one_class() -> None:
    edges, labels, universe, nfeat = _subgraph()
    store = ReferencePyStore(nfeat)
    store.insert(edges)
    m = XGBGraphMaintainer(store, universe, labels, seed=0)
    m.rebuild()
    empty = np.zeros(universe, dtype=bool)  # nothing seen -> no classes
    assert m.retrain(empty, ref_time=1.0) is None
    assert np.array_equal(m.scores(), np.zeros(universe))  # no classifier yet


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
