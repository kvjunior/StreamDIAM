"""Replay a time-sorted edge table as a stream of deltas ``dG``.

The streaming system consumes the graph as a sequence of
:class:`TransactionBatch` deltas in canonical ``(t, eid)`` order. Two groupings
are provided:

* :func:`replay_by_time` — one delta per distinct timestamp, which is the
  natural tick for the maintainer and keeps every event that is simultaneous in
  the data simultaneous in the stream;
* :func:`replay_by_count` — fixed-size deltas, for throughput benchmarking and
  for models trained on uniform batches.

Both preserve order and are lossless: concatenating the yielded deltas
reproduces the input edge table exactly.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from streamdiam.data.schema import SchemaError, TransactionBatch

__all__ = ["replay_by_time", "replay_by_count"]


def replay_by_time(edges: TransactionBatch) -> Iterator[TransactionBatch]:
    """Yield one delta per distinct timestamp, in increasing time order.

    Requires ``edges`` to be in canonical order (so equal timestamps are
    contiguous). Each yielded batch is non-empty and shares a single timestamp.
    """
    if not edges.is_time_sorted():
        raise SchemaError("replay_by_time requires canonically ordered edges")
    n = len(edges)
    if n == 0:
        return
    t = edges.timestamp
    # Boundaries where the timestamp changes: start of each run.
    change = np.flatnonzero(t[1:] != t[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    for lo, hi in zip(starts.tolist(), ends.tolist()):
        chunk = edges[lo:hi]
        assert isinstance(chunk, TransactionBatch)
        yield chunk


def replay_by_count(edges: TransactionBatch, size: int) -> Iterator[TransactionBatch]:
    """Yield fixed-size deltas of ``size`` edges (the last may be smaller).

    Order is preserved; ``size`` must be positive. Does not require the input to
    be time-sorted (useful for raw throughput tests), but note that a delta may
    then straddle a timestamp boundary.
    """
    if size <= 0:
        raise SchemaError(f"size must be positive, got {size}")
    n = len(edges)
    for lo in range(0, n, size):
        chunk = edges[lo : min(lo + size, n)]
        assert isinstance(chunk, TransactionBatch)
        yield chunk
