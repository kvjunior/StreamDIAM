"""Correctness tests for the reference temporal multigraph store.

The store is the oracle for the rest of the system, so it is itself checked
against an *independent* brute-force oracle: filter the raw inserted edges and
sort them canonically with numpy, with no shared code path. We stress the
cases that break naive implementations -- tie timestamps (parallel edges in one
block), out-of-order batch arrival, self-loops, repeated endpoints, unbounded
and empty windows -- and verify snapshot isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from streamdiam.data.schema import (
    ATTR_DTYPE,
    LABEL_DTYPE,
    Direction,
    SchemaError,
    TransactionBatch,
)
from streamdiam.store import ReferencePyStore

F = 4
RNG = np.random.default_rng(20270211)


# --------------------------------------------------------------------------- #
# Helpers: a random stream and an independent brute-force oracle.
# --------------------------------------------------------------------------- #

def make_stream(n_edges: int, n_accounts: int, n_times: int):
    """Return (edge_id, src, dst, t, attr, label) for ``n_edges`` transactions.

    edge_id is a shuffled 0..n-1 (unique, but arrival order != id order, so the
    (t, eid) tie-break is actually exercised). Timestamps are drawn from a small
    set so ties and parallel edges are common. Self-loops are allowed.
    """
    eid = RNG.permutation(n_edges).astype(np.int64)
    src = RNG.integers(0, n_accounts, n_edges, dtype=np.int64)
    dst = RNG.integers(0, n_accounts, n_edges, dtype=np.int64)
    t = np.sort(RNG.integers(0, n_times, n_edges)).astype(np.int64)  # mostly ordered
    # Perturb a few timestamps to force genuinely out-of-order inserts.
    for _ in range(n_edges // 10):
        i = int(RNG.integers(0, n_edges))
        t[i] = int(RNG.integers(0, n_times))
    attr = RNG.standard_normal((n_edges, F)).astype(ATTR_DTYPE)
    label = RNG.integers(-1, 2, n_edges).astype(LABEL_DTYPE)
    return eid, src, dst, t, attr, label


def brute_force(eid, src, dst, t, attr, label, account, direction, t_start, t_end):
    """Independent oracle: the canonical ordered rows of S_dir(account) in window."""
    key = dst if direction is Direction.IN else src
    mask = key == account
    if t_start is not None:
        mask &= t >= t_start
    if t_end is not None:
        mask &= t <= t_end
    sel = np.flatnonzero(mask)
    order = sel[np.lexsort((eid[sel], t[sel]))]  # primary t, tie-break eid
    return order


def insert_in_batches(store, eid, src, dst, t, attr, label, sizes):
    """Insert the stream split into out-of-order chunks (shuffled chunk order)."""
    bounds = np.cumsum([0] + sizes)
    chunks = list(range(len(sizes)))
    RNG.shuffle(chunks)  # arrival order of batches is itself scrambled
    for c in chunks:
        a, b = int(bounds[c]), int(bounds[c + 1])
        store.insert(TransactionBatch.from_arrays(
            eid[a:b], src[a:b], dst[a:b], t[a:b], attr[a:b], label[a:b]))


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_range_query_matches_brute_force():
    eid, src, dst, t, attr, label = make_stream(2000, n_accounts=40, n_times=50)
    store = ReferencePyStore(F)
    insert_in_batches(store, eid, src, dst, t, attr, label, [700, 500, 800])
    assert store.num_edges == 2000

    windows = [(None, None), (10, 40), (0, 0), (25, 25), (None, 30), (20, None), (49, 3)]
    for account in range(40):
        for direction in (Direction.IN, Direction.OUT):
            for (ts, te) in windows:
                exp = brute_force(eid, src, dst, t, attr, label, account, direction, ts, te)
                got = store.range_query(account, direction, ts, te)
                # edge ids in exact canonical order
                assert np.array_equal(got.edge_id, eid[exp]), (account, direction, ts, te)
                # payload columns line up row-for-row
                assert np.array_equal(got.src, src[exp])
                assert np.array_equal(got.dst, dst[exp])
                assert np.array_equal(got.timestamp, t[exp])
                assert np.array_equal(got.label, label[exp])
                assert np.allclose(got.attr, attr[exp])
                # the returned sequence is itself canonically sorted
                assert got.is_time_sorted()


def test_scan_equals_unbounded_range():
    eid, src, dst, t, attr, label = make_stream(500, 15, 30)
    store = ReferencePyStore(F)
    insert_in_batches(store, eid, src, dst, t, attr, label, [250, 250])
    for a in range(15):
        for d in (Direction.IN, Direction.OUT):
            full = store.scan(a, d)
            rng = store.range_query(a, d, None, None)
            assert np.array_equal(full.edge_id, rng.edge_id)
            assert store.sequence_length(a, d) == len(full)


def test_tail_is_suffix_of_scan():
    eid, src, dst, t, attr, label = make_stream(600, 12, 25)
    store = ReferencePyStore(F)
    insert_in_batches(store, eid, src, dst, t, attr, label, [600])
    for a in range(12):
        for d in (Direction.IN, Direction.OUT):
            full = store.scan(a, d)
            for k in (0, 1, 3, len(full), len(full) + 5):
                tail = store.tail(a, d, k)
                exp = full[max(0, len(full) - k):] if k > 0 else full[len(full):]
                assert np.array_equal(tail.edge_id, exp.edge_id), (a, d, k)
    # non-positive k is empty
    assert len(store.tail(0, Direction.IN, 0)) == 0
    assert len(store.tail(0, Direction.IN, -3)) == 0


def test_neighbors_and_accounts():
    eid, src, dst, t, attr, label = make_stream(1500, 25, 40)
    store = ReferencePyStore(F)
    insert_in_batches(store, eid, src, dst, t, attr, label, [500, 500, 500])
    assert np.array_equal(store.accounts(), np.array(sorted(set(src) | set(dst))))
    for a in range(25):
        exp_in = np.array(sorted(set(src[dst == a])), dtype=np.int64)
        exp_out = np.array(sorted(set(dst[src == a])), dtype=np.int64)
        assert np.array_equal(store.neighbors(a, Direction.IN), exp_in)
        assert np.array_equal(store.neighbors(a, Direction.OUT), exp_out)


def test_unknown_account_returns_empty_typed_batch():
    store = ReferencePyStore(F)
    store.insert(TransactionBatch.from_arrays([0], [1], [2], [5], [[1, 2, 3, 4]]))
    empty = store.scan(999, Direction.IN)
    assert len(empty) == 0 and empty.num_features == F
    assert store.sequence_length(999, Direction.OUT) == 0
    assert len(store.neighbors(999, Direction.IN)) == 0


def test_self_loop_appears_in_both_sequences():
    store = ReferencePyStore(F)
    store.insert(TransactionBatch.from_arrays([7], [3], [3], [10], [[0, 0, 0, 0]]))
    assert store.sequence_length(3, Direction.IN) == 1
    assert store.sequence_length(3, Direction.OUT) == 1
    assert int(store.scan(3, Direction.IN).edge_id[0]) == 7
    assert np.array_equal(store.neighbors(3, Direction.IN), np.array([3]))


def test_empty_insert_is_noop():
    store = ReferencePyStore(F)
    store.insert(TransactionBatch.from_records([], num_features=F))
    assert store.num_edges == 0 and store.num_accounts == 0


def test_feature_dim_mismatch_rejected():
    store = ReferencePyStore(F)
    bad = TransactionBatch.from_arrays([0], [0], [0], [0], [[1, 2]])  # F=2
    with pytest.raises(SchemaError):
        store.insert(bad)


def test_snapshot_isolation():
    eid, src, dst, t, attr, label = make_stream(400, 10, 20)
    store = ReferencePyStore(F)
    insert_in_batches(store, eid, src, dst, t, attr, label, [200, 200])
    snap = store.snapshot()
    before = {a: store.scan(a, Direction.OUT).edge_id.copy() for a in range(10)}

    # Mutate the parent after snapshotting.
    store.insert(TransactionBatch.from_arrays(
        [9999], [0], [1], [0], [[9, 9, 9, 9]]))  # a very-early timestamp, forces a shift

    # Snapshot is frozen at its creation state ...
    for a in range(10):
        assert np.array_equal(snap.scan(a, Direction.OUT).edge_id, before[a]), a
    assert snap.num_edges == 400
    assert store.num_edges == 401
    # ... and is read-only.
    with pytest.raises(RuntimeError):
        snap.insert(TransactionBatch.from_arrays([1], [0], [0], [0], [[0, 0, 0, 0]]))


def test_results_are_frozen():
    store = ReferencePyStore(F)
    store.insert(TransactionBatch.from_arrays([0, 1], [0, 0], [1, 1], [0, 1],
                                              [[1, 1, 1, 1], [2, 2, 2, 2]]))
    out = store.scan(0, Direction.OUT)
    assert not out.edge_id.flags.writeable
    with pytest.raises(ValueError):
        out.attr[0, 0] = 5.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
