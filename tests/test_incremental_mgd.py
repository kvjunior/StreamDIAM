"""Equivalence tests: full incremental stack (Edge2Seq + MGD) == oracle (Theorem 2).

A graph is streamed, one timestamp-delta at a time, through the incremental
Edge2Seq and incremental MGD maintainers starting from the empty graph. After
the stream the maintained representations and scores must equal
``FromScratchMaintainer`` on the full graph, within floating-point summation
tolerance. Also checks the affected-set locality directly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from streamdiam.data import TransactionBatch, load_elliptic, replay_by_time  # noqa: E402
from streamdiam.maintain import (  # noqa: E402
    FromScratchMaintainer,
    IncrementalEdge2Seq,
    IncrementalMGD,
)
from streamdiam.maintain.affected_set import neighbor_closure  # noqa: E402
from streamdiam.model import DIAM  # noqa: E402
from streamdiam.store import ReferencePyStore  # noqa: E402

F = 4
BIG_TMAX = 100_000
ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")


def random_graph(seed, n_edges, n_accounts, n_times=10, n_features=F):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_accounts, n_edges)
    dst = rng.integers(0, n_accounts, n_edges)
    t = np.sort(rng.integers(0, n_times, n_edges))
    attr = rng.standard_normal((n_edges, n_features)).astype(np.float32)
    return TransactionBatch.from_arrays(
        np.arange(n_edges), src, dst, t, attr).sorted_by_time()


def stream_incrementally(model, batch, n_accounts, n_features):
    """Run the full incremental stack from the empty graph over the stream."""
    store = ReferencePyStore(n_features)
    e2s = IncrementalEdge2Seq(model.edge2seq)
    mgd = IncrementalMGD(model, n_accounts)
    mgd.rebuild(e2s.codes(n_accounts), store)  # empty-graph initial view
    for delta in replay_by_time(batch):
        store.insert(delta)                     # 1) adjacency now includes delta
        e2s.on_batch(delta)                     # 2) refresh Edge2Seq codes
        changed = np.unique(np.concatenate([delta.src, delta.dst]))
        rows = torch.stack([e2s.code(int(v)) for v in changed])
        mgd.update(changed, rows, store)        # 3) refresh MGD on the closure
    return mgd


def oracle(model, batch, n_accounts, n_features):
    store = ReferencePyStore(n_features)
    store.insert(batch)
    return FromScratchMaintainer(model, t_max=BIG_TMAX, chunk_size=10_000).compute_view(
        store, n_accounts
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_incremental_stack_matches_oracle(seed, num_layers):
    torch.manual_seed(seed)
    n_acc = 18
    model = DIAM(in_dim=F, hidden_dim=16, num_layers=num_layers)
    batch = random_graph(seed, n_edges=120, n_accounts=n_acc)
    mgd = stream_incrementally(model, batch, n_acc, F)
    view = oracle(model, batch, n_acc, F)
    assert torch.allclose(mgd.hidden(), view.hidden, atol=1e-4), \
        float((mgd.hidden() - view.hidden).abs().max())
    assert torch.allclose(mgd.scores(), view.scores, atol=1e-4)


def test_rebuild_equals_oracle_on_static_graph():
    """rebuild() alone (no streaming) must equal the oracle MGD pass."""
    torch.manual_seed(3)
    n_acc, nfeat = 20, F
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=2)
    batch = random_graph(4, 150, n_acc)
    store = ReferencePyStore(nfeat)
    store.insert(batch)
    e2s = IncrementalEdge2Seq(model.edge2seq)
    e2s.on_batch(batch)
    mgd = IncrementalMGD(model, n_acc)
    mgd.rebuild(e2s.codes(n_acc), store)
    view = oracle(model, batch, n_acc, nfeat)
    assert torch.allclose(mgd.hidden(), view.hidden, atol=1e-4)


def test_neighbor_closure_is_both_directions():
    store = ReferencePyStore(F)
    # edges: 1->2, 3->1  => N({1}) = {2 (out), 3 (in)}
    store.insert(TransactionBatch.from_arrays(
        [0, 1], [1, 3], [2, 1], [0, 1], [[0, 0, 0, 0], [0, 0, 0, 0]]))
    clo = neighbor_closure(store, np.array([1]))
    assert set(clo.tolist()) == {1, 2, 3}


def test_untouched_nodes_keep_values():
    """A delta far from node w must not change w's representation."""
    torch.manual_seed(9)
    n_acc = 30
    model = DIAM(in_dim=F, hidden_dim=12, num_layers=1)
    # seed graph only among nodes [0..9]; node 25 is isolated
    seed_batch = random_graph(1, 60, 10)
    store = ReferencePyStore(F)
    e2s = IncrementalEdge2Seq(model.edge2seq)
    mgd = IncrementalMGD(model, n_acc)
    mgd.rebuild(e2s.codes(n_acc), store)
    for delta in replay_by_time(seed_batch):
        store.insert(delta)
        e2s.on_batch(delta)
        changed = np.unique(np.concatenate([delta.src, delta.dst]))
        rows = torch.stack([e2s.code(int(v)) for v in changed])
        mgd.update(changed, rows, store)
    h25_before = mgd.hidden()[25].clone()
    # new delta among nodes [0..9], 1-layer model => node 25 (2+ hops away) unchanged
    extra = TransactionBatch.from_arrays([100], [0], [1], [99], [[1, 1, 1, 1]])
    store.insert(extra)
    e2s.on_batch(extra)
    mgd.update(np.array([0, 1]), torch.stack([e2s.code(0), e2s.code(1)]), store)
    assert torch.equal(mgd.hidden()[25], h25_before)


@needs_elliptic
def test_real_elliptic_subgraph_stack():
    graph = load_elliptic(ELLIPTIC_ROOT)
    e = graph.edges
    K = 400
    idx = np.flatnonzero((e.src < K) & (e.dst < K))
    sub = TransactionBatch.from_arrays(
        np.arange(idx.size), e.src[idx], e.dst[idx], e.timestamp[idx], e.attr[idx])
    model = DIAM(in_dim=graph.num_features, hidden_dim=24, num_layers=2)
    mgd = stream_incrementally(model, sub, K, graph.num_features)
    view = oracle(model, sub, K, graph.num_features)
    assert torch.allclose(mgd.scores(), view.scores, atol=1e-4), \
        float((mgd.scores() - view.scores).abs().max())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
