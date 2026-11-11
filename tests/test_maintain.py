"""Tests for the from-scratch oracle, the store->model adapter, and all_edges.

The oracle is the ground truth the incremental maintainer (next batch) is graded
against, so its two load-bearing properties are checked explicitly: chunked
Edge2Seq equals a single-chunk pass (chunking is transparent), and the whole
computation is deterministic in eval mode. One bounded real-Elliptic run shows
it composes with the real loader end to end.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from streamdiam.data import Direction, TransactionBatch, load_elliptic  # noqa: E402
from streamdiam.maintain import FromScratchMaintainer  # noqa: E402
from streamdiam.maintain.inputs import (  # noqa: E402
    build_sequence_inputs,
    edge_index_from_store,
)
from streamdiam.model import DIAM  # noqa: E402
from streamdiam.store import ReferencePyStore  # noqa: E402

F = 4
ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")


def small_store(seed=0, n_edges=120, n_accounts=20):
    rng = np.random.default_rng(seed)
    store = ReferencePyStore(F)
    eid = rng.permutation(n_edges)
    src = rng.integers(0, n_accounts, n_edges)
    dst = rng.integers(0, n_accounts, n_edges)
    t = np.sort(rng.integers(0, 15, n_edges))
    attr = rng.standard_normal((n_edges, F)).astype(np.float32)
    store.insert(TransactionBatch.from_arrays(eid, src, dst, t, attr))
    return store, n_accounts


# --------------------------------------------------------------------------- #
# all_edges
# --------------------------------------------------------------------------- #

def test_all_edges_canonical_and_complete():
    store, _ = small_store()
    allb = store.all_edges()
    assert len(allb) == store.num_edges
    assert allb.is_time_sorted()
    # same multiset of (src,dst,t) as what we can read back per-account (OUT side)
    from_out = [store.scan(a, Direction.OUT) for a in store.accounts().tolist()]
    total_out = sum(len(b) for b in from_out)
    assert total_out == store.num_edges  # every edge is some account's out-edge


def test_all_edges_empty_store():
    store = ReferencePyStore(F)
    allb = store.all_edges()
    assert len(allb) == 0 and allb.num_features == F


# --------------------------------------------------------------------------- #
# sequence input adapter
# --------------------------------------------------------------------------- #

def test_self_loop_for_accounts_without_edges():
    store = ReferencePyStore(F)
    store.insert(TransactionBatch.from_arrays([0], [1], [2], [5], [[1, 1, 1, 1]]))
    # account 9 has no edges -> length-1 self-loop of zeros
    si = build_sequence_inputs(store, np.array([9]), t_max=8)
    assert int(si.in_len[0]) == 1 and int(si.out_len[0]) == 1
    assert torch.count_nonzero(si.in_seq[0]) == 0
    assert torch.count_nonzero(si.out_seq[0]) == 0


def test_sequence_capping_and_recency():
    store = ReferencePyStore(F)
    n = 10
    # account 3 sends n edges at increasing times; attr encodes the time
    eid = np.arange(n)
    attr = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, F))
    store.insert(TransactionBatch.from_arrays(eid, np.full(n, 3), np.arange(n), np.arange(n), attr))
    si = build_sequence_inputs(store, np.array([3]), t_max=4)
    assert int(si.out_len[0]) == 4                       # capped at t_max
    # the 4 kept rows are the most recent (times 6,7,8,9), in order
    assert torch.allclose(si.out_seq[0, :4, 0], torch.tensor([6.0, 7.0, 8.0, 9.0]))


def test_edge_index_matches_store():
    store, _ = small_store()
    ei = edge_index_from_store(store)
    allb = store.all_edges()
    assert ei.shape == (2, store.num_edges)
    assert np.array_equal(ei[0].numpy(), allb.src)
    assert np.array_equal(ei[1].numpy(), allb.dst)


# --------------------------------------------------------------------------- #
# from-scratch oracle
# --------------------------------------------------------------------------- #

def test_view_shapes_and_score_range():
    store, n = small_store()
    model = DIAM(in_dim=F, hidden_dim=16, num_layers=2)
    view = FromScratchMaintainer(model, t_max=8).compute_view(store, n)
    assert view.codes.shape == (n, 32)
    assert view.hidden.shape == (n, 16)
    assert view.logits.shape == (n,) and view.scores.shape == (n,)
    assert len(view.layer_hidden) == 2
    assert float(view.scores.min()) >= 0.0 and float(view.scores.max()) <= 1.0
    assert torch.isfinite(view.logits).all()


def test_chunking_is_transparent():
    store, n = small_store(seed=1)
    torch.manual_seed(0)
    model = DIAM(in_dim=F, hidden_dim=16, num_layers=2)
    v_small = FromScratchMaintainer(model, t_max=8, chunk_size=3).compute_view(store, n)
    v_big = FromScratchMaintainer(model, t_max=8, chunk_size=10_000).compute_view(store, n)
    assert torch.allclose(v_small.codes, v_big.codes, atol=1e-6)
    assert torch.allclose(v_small.scores, v_big.scores, atol=1e-6)


def test_deterministic():
    store, n = small_store(seed=2)
    torch.manual_seed(0)
    model = DIAM(in_dim=F, hidden_dim=16, num_layers=2)
    m = FromScratchMaintainer(model, t_max=8)
    a = m.compute_view(store, n)
    b = m.compute_view(store, n)
    assert torch.equal(a.scores, b.scores)


def test_universe_too_small_rejected():
    store, n = small_store()
    model = DIAM(in_dim=F, hidden_dim=8, num_layers=1)
    with pytest.raises(ValueError):
        FromScratchMaintainer(model, t_max=8).compute_view(store, n - 1)


def test_feature_dim_mismatch_rejected():
    store, n = small_store()
    model = DIAM(in_dim=F + 1, hidden_dim=8, num_layers=1)
    with pytest.raises(ValueError):
        FromScratchMaintainer(model, t_max=8).compute_view(store, n)


@needs_elliptic
def test_real_elliptic_subgraph_smoke():
    """Build a store from a bounded Elliptic subgraph and materialize the view."""
    graph = load_elliptic(ELLIPTIC_ROOT)
    e = graph.edges
    K = 1500  # compact node universe
    keep = (e.src < K) & (e.dst < K)
    idx = np.flatnonzero(keep)
    sub = TransactionBatch.from_arrays(
        np.arange(idx.size), e.src[idx], e.dst[idx], e.timestamp[idx], e.attr[idx])
    store = ReferencePyStore(graph.num_features)
    store.insert(sub)
    model = DIAM(in_dim=graph.num_features, hidden_dim=32, num_layers=2)
    view = FromScratchMaintainer(model, t_max=16, chunk_size=512).compute_view(store, K)
    assert view.scores.shape == (K,)
    assert torch.isfinite(view.logits).all()
    assert float(view.scores.min()) >= 0.0 and float(view.scores.max()) <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
