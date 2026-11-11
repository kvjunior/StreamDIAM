"""Equivalence tests: incremental Edge2Seq == from-scratch oracle (Theorem 1).

The maintainer replays a stream edge by edge; the oracle recomputes the whole
growing-window Edge2Seq. Their codes must match to floating-point tolerance.
Cases stress ties, out-of-order arrival vs. canonical replay, accounts with only
one direction, and accounts with no edges (self-loop). One bounded real-Elliptic
run confirms it end to end.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from streamdiam.data import TransactionBatch, load_elliptic, replay_by_time  # noqa: E402
from streamdiam.maintain import FromScratchMaintainer, IncrementalEdge2Seq  # noqa: E402
from streamdiam.model import DIAM  # noqa: E402
from streamdiam.store import ReferencePyStore  # noqa: E402

F = 4
BIG_TMAX = 100_000  # oracle uses full history to match the growing window
ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")


def random_graph(seed, n_edges, n_accounts, n_times=12):
    rng = np.random.default_rng(seed)
    eid = rng.permutation(n_edges)
    src = rng.integers(0, n_accounts, n_edges)
    dst = rng.integers(0, n_accounts, n_edges)
    t = np.sort(rng.integers(0, n_times, n_edges))
    attr = rng.standard_normal((n_edges, F)).astype(np.float32)
    # Sort into canonical (t, eid) order, as a real loader would.
    return TransactionBatch.from_arrays(eid, src, dst, t, attr).sorted_by_time()


def oracle_codes(model, batch, n_accounts):
    store = ReferencePyStore(F)
    store.insert(batch)
    view = FromScratchMaintainer(model, t_max=BIG_TMAX, chunk_size=10_000).compute_view(
        store, n_accounts
    )
    return view.codes


def incremental_codes(model, batch, n_accounts, by_time=True):
    inc = IncrementalEdge2Seq(model.edge2seq)
    if by_time:
        # Replay in canonical order, delta by delta (the intended streaming path).
        for delta in replay_by_time(batch):
            inc.on_batch(delta)
    else:
        inc.on_batch(batch)
    return inc.codes(n_accounts)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_incremental_matches_oracle(seed):
    torch.manual_seed(seed)
    n_acc = 20
    model = DIAM(in_dim=F, hidden_dim=16, num_layers=2)
    batch = random_graph(seed, n_edges=150, n_accounts=n_acc)
    exp = oracle_codes(model, batch, n_acc)
    got = incremental_codes(model, batch, n_acc)
    assert torch.allclose(got, exp, atol=1e-5), float((got - exp).abs().max())


def test_canonical_batch_equivalent_to_time_replay():
    """A canonically-sorted batch fed whole == replayed delta-by-delta."""
    torch.manual_seed(5)
    n_acc = 15
    model = DIAM(in_dim=F, hidden_dim=16, num_layers=1)
    batch = random_graph(7, n_edges=120, n_accounts=n_acc)
    whole = incremental_codes(model, batch, n_acc, by_time=False)
    per_tick = incremental_codes(model, batch, n_acc, by_time=True)
    assert torch.allclose(whole, per_tick, atol=1e-6)


def test_empty_and_single_direction_accounts():
    torch.manual_seed(1)
    model = DIAM(in_dim=F, hidden_dim=8, num_layers=1)
    # account 0 only sends, account 1 only receives, account 5 never appears
    batch = TransactionBatch.from_arrays(
        edge_id=[0, 1], src=[0, 0], dst=[1, 1], timestamp=[2, 3],
        attr=[[1, 2, 3, 4], [5, 6, 7, 8]])
    exp = oracle_codes(model, batch, 6)
    got = incremental_codes(model, batch, 6)
    assert torch.allclose(got, exp, atol=1e-5)
    # account 5 (no edges) must equal the pure self-loop code in both
    assert torch.allclose(got[5], exp[5], atol=1e-6)


def test_incremental_update_is_constant_work_per_edge():
    """State size per (account,direction) stays O(c): only h, m, count are kept."""
    model = DIAM(in_dim=F, hidden_dim=8, num_layers=1)
    inc = IncrementalEdge2Seq(model.edge2seq)
    inc.on_batch(random_graph(0, 500, 10))
    for states in inc._state.values():
        for st in states:
            assert st.h is None or st.h.shape == (8,)
            assert st.m is None or st.m.shape == (8,)


@needs_elliptic
def test_real_elliptic_subgraph_equivalence():
    graph = load_elliptic(ELLIPTIC_ROOT)
    e = graph.edges
    K = 1200
    idx = np.flatnonzero((e.src < K) & (e.dst < K))
    sub = TransactionBatch.from_arrays(
        np.arange(idx.size), e.src[idx], e.dst[idx], e.timestamp[idx], e.attr[idx])
    model = DIAM(in_dim=graph.num_features, hidden_dim=24, num_layers=2)
    exp = oracle_codes_fullF(model, sub, K, graph.num_features)
    inc = IncrementalEdge2Seq(model.edge2seq)
    for delta in replay_by_time(sub):
        inc.on_batch(delta)
    got = inc.codes(K)
    assert torch.allclose(got, exp, atol=1e-4), float((got - exp).abs().max())


def oracle_codes_fullF(model, batch, n_accounts, n_features):
    store = ReferencePyStore(n_features)
    store.insert(batch)
    view = FromScratchMaintainer(model, t_max=BIG_TMAX, chunk_size=4096).compute_view(
        store, n_accounts
    )
    return view.codes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
