"""Ingestion tests: real Elliptic invariants + synthetic split/replay checks.

The Elliptic tests run only when ``STREAMDIAM_ELLIPTIC_ROOT`` points at the
directory of the three Elliptic CSVs; otherwise they skip, so the suite stays
runnable without the ~150 MB download. The split and replay tests use
constructed inputs and always run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from streamdiam.data import (
    SchemaError,
    TransactionBatch,
    load_elliptic,
    remap_accounts,
    replay_by_count,
    replay_by_time,
    temporal_split,
)
from streamdiam.data.schema import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN

ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(
    ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT to run Elliptic tests"
)

# Published Elliptic ground-truth counts (sanity anchors).
N_NODES, N_EDGES, N_FEAT = 203_769, 234_355, 165
N_ILLICIT, N_LICIT = 4_545, 42_019
N_UNKNOWN = N_NODES - N_ILLICIT - N_LICIT  # 157,205


@pytest.fixture(scope="module")
def elliptic():
    return load_elliptic(ELLIPTIC_ROOT)


# --------------------------------------------------------------------------- #
# remap_accounts (always runs)
# --------------------------------------------------------------------------- #

def test_remap_is_a_bijection():
    orig = np.array([50, 7, 7, 900, 50, 3], dtype=np.int64)
    codes, uniques = remap_accounts(orig)
    assert np.array_equal(uniques[codes], orig)             # recovers originals
    assert set(codes.tolist()) == set(range(len(uniques)))  # dense [0, N)
    assert np.array_equal(uniques, np.sort(uniques))        # uniques sorted


# --------------------------------------------------------------------------- #
# Elliptic on real data
# --------------------------------------------------------------------------- #

@needs_elliptic
def test_elliptic_counts_match_published(elliptic):
    assert elliptic.num_accounts == N_NODES
    assert elliptic.num_features == N_FEAT
    assert elliptic.num_edges == N_EDGES
    assert elliptic.extra["dropped_edges"] == 0
    counts = elliptic.label_counts()
    assert counts[LABEL_ILLICIT] == N_ILLICIT
    assert counts[LABEL_LICIT] == N_LICIT
    assert counts[LABEL_UNKNOWN] == N_UNKNOWN


@needs_elliptic
def test_elliptic_structural_invariants(elliptic):
    e = elliptic.edges
    assert e.is_time_sorted()
    assert np.array_equal(e.edge_id, np.arange(N_EDGES))     # dense eids
    assert int(e.src.min()) >= 0 and int(e.src.max()) < N_NODES
    assert int(e.dst.min()) >= 0 and int(e.dst.max()) < N_NODES
    assert np.isfinite(e.attr).all()
    assert e.num_features == N_FEAT
    # id map is a bijection onto [0, N)
    uniques = elliptic.extra["original_ids"]
    assert uniques.shape[0] == N_NODES
    assert np.unique(uniques).shape[0] == N_NODES


@needs_elliptic
def test_elliptic_edge_timestamp_follows_source_rule(elliptic):
    """Every edge's timestamp equals its source account's Elliptic time step."""
    e = elliptic.edges
    node_time = elliptic.extra["node_time"]
    assert np.array_equal(e.timestamp, node_time[e.src])
    # time steps lie in 1..49
    assert int(e.timestamp.min()) >= 1 and int(e.timestamp.max()) <= 49


@needs_elliptic
def test_elliptic_temporal_split_no_leakage(elliptic):
    e = elliptic.edges
    split = temporal_split(e.timestamp, (0.6, 0.2, 0.2))
    # exact partition
    assert np.array_equal(split.train_mask | split.val_mask | split.test_mask,
                          np.ones(N_EDGES, dtype=bool))
    assert not np.any(split.train_mask & split.val_mask)
    assert not np.any(split.val_mask & split.test_mask)
    assert not np.any(split.train_mask & split.test_mask)
    # strict ordering: max(train) <= t_train < min(test)
    t = e.timestamp
    assert t[split.train_mask].max() <= split.t_train
    assert t[split.test_mask].min() > split.t_val
    assert split.t_train <= split.t_val
    tr, va, te = split.sizes
    assert tr + va + te == N_EDGES and tr > 0 and va > 0 and te > 0


@needs_elliptic
def test_elliptic_replay_by_time_is_lossless(elliptic):
    e = elliptic.edges
    parts = list(replay_by_time(e))
    # one delta per distinct timestamp, each internally single-timestamp
    assert len(parts) == len(np.unique(e.timestamp))
    for p in parts:
        assert len(p) > 0 and int(p.timestamp.min()) == int(p.timestamp.max())
    rebuilt = TransactionBatch.concat(parts)
    assert np.array_equal(rebuilt.edge_id, e.edge_id)


# --------------------------------------------------------------------------- #
# split / replay on constructed inputs (always run)
# --------------------------------------------------------------------------- #

def _toy_batch(times):
    n = len(times)
    return TransactionBatch.from_arrays(
        edge_id=np.arange(n), src=np.zeros(n, int), dst=np.ones(n, int),
        timestamp=times, attr=np.zeros((n, 2), np.float32))


def test_split_keeps_equal_timestamps_together():
    # all-equal timestamps -> everything lands in train (t <= t_train)
    split = temporal_split(np.array([5, 5, 5, 5]), (0.5, 0.25, 0.25))
    assert split.sizes == (4, 0, 0)


def test_split_validates_fractions():
    with pytest.raises(SchemaError):
        temporal_split(np.array([1, 2, 3]), (0.6, 0.2, 0.1))  # sum != 1
    with pytest.raises(SchemaError):
        temporal_split(np.array([1, 2, 3]), (0.6, 0.4, 0.0))  # non-positive
    with pytest.raises(SchemaError):
        temporal_split(np.array([], dtype=int))               # empty


def test_replay_by_time_requires_sorted():
    unsorted = TransactionBatch.from_arrays([0, 1], [0, 0], [1, 1], [9, 2],
                                            [[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(SchemaError):
        list(replay_by_time(unsorted))


def test_replay_by_count_is_lossless_and_sized():
    b = _toy_batch([1, 1, 2, 3, 3, 3, 4])
    parts = list(replay_by_count(b, 3))
    assert [len(p) for p in parts] == [3, 3, 1]
    assert np.array_equal(TransactionBatch.concat(parts).edge_id, b.edge_id)
    with pytest.raises(SchemaError):
        list(replay_by_count(b, 0))


def test_replay_by_time_toy():
    b = _toy_batch([1, 1, 2, 5, 5])
    parts = list(replay_by_time(b))
    assert [len(p) for p in parts] == [2, 1, 2]
    assert [int(p.timestamp[0]) for p in parts] == [1, 2, 5]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
