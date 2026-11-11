"""Pure-Python reference implementation of the temporal multigraph store.

This backend is written for auditability, not speed: it is the **correctness
oracle** against which the native store and the incremental maintainer are
checked. The design mirrors the intended physical layout so the oracle and the
optimized store share semantics:

* an append-only **columnar log** — parallel growable arrays holding every
  transaction's ``edge_id``, ``src``, ``dst``, ``timestamp``, feature row, and
  ``label``, indexed by a global insertion position ``pos``;
* a per-account **sequence index** — for each account and direction, a list of
  ``(timestamp, edge_id, pos)`` tuples kept in canonical order, so a windowed
  read is a pair of binary searches plus a slice.

Honest complexity note (this is the claim the old paper got wrong): inserting
into a sorted sequence is ``O(log s)`` to locate the position but ``O(s)`` in
the worst case to shift elements. On blockchain streams timestamps are *almost*
non-decreasing, so inserts land at or near the end and are ``O(1)`` amortized in
the common case; a genuinely out-of-order insert costs a shift. The reference
store makes no complexity claim — it exists to be correct. Where the paper
needs a complexity guarantee, it is stated and proved for the native store in
``docs/proofs.md``, for the operation the native store actually performs.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from typing import Final

import numpy as np
import numpy.typing as npt

from streamdiam.data.schema import (
    ACCOUNT_DTYPE,
    ATTR_DTYPE,
    EDGE_ID_DTYPE,
    LABEL_DTYPE,
    LABEL_UNKNOWN,
    TIME_DTYPE,
    Direction,
    SchemaError,
    TransactionBatch,
)
from streamdiam.store.interface import TemporalMultigraphStore

__all__ = ["ReferencePyStore"]

# A sentinel strictly below any real (edge_id, pos), which are non-negative.
# Used to build half-open search keys on the (timestamp, edge_id, pos) tuples.
_BELOW_ANY: Final[int] = -1


class _GrowableBuffer:
    """A numpy-backed append-only buffer that doubles its capacity on demand.

    Supports 1-D columns (``ncols=None``) and 2-D columns (the feature matrix).
    :attr:`data` is a view of exactly the filled rows; appends preserve the
    prefix, so a view taken earlier stays valid for positions already written
    (the property snapshot isolation relies on).
    """

    __slots__ = ("_buf", "_size", "_ncols")

    def __init__(self, dtype: np.dtype, ncols: int | None = None, capacity: int = 1024):
        self._ncols = ncols
        shape: tuple[int, ...] = (capacity,) if ncols is None else (capacity, ncols)
        self._buf = np.empty(shape, dtype=dtype)
        self._size = 0

    def append(self, block: np.ndarray) -> None:
        m = int(block.shape[0])
        if m == 0:
            return
        need = self._size + m
        cap = self._buf.shape[0]
        if need > cap:
            new_cap = max(need, cap * 2)
            new_shape: tuple[int, ...] = (
                (new_cap,) if self._ncols is None else (new_cap, self._ncols)
            )
            new_buf = np.empty(new_shape, dtype=self._buf.dtype)
            new_buf[: self._size] = self._buf[: self._size]
            self._buf = new_buf
        self._buf[self._size : need] = block
        self._size = need

    @property
    def data(self) -> np.ndarray:
        return self._buf[: self._size]


class ReferencePyStore(TemporalMultigraphStore):
    """Auditable, correctness-first temporal multigraph store."""

    __slots__ = (
        "_F",
        "_eid",
        "_src",
        "_dst",
        "_t",
        "_attr",
        "_label",
        "_n",
        "_seq",
        "_nbr",
        "_read_only",
    )

    def __init__(self, num_features: int):
        if num_features < 1:
            raise SchemaError(f"num_features must be >= 1, got {num_features}")
        self._F = int(num_features)
        # Columnar log (one logical row per transaction, keyed by position).
        self._eid = _GrowableBuffer(EDGE_ID_DTYPE)
        self._src = _GrowableBuffer(ACCOUNT_DTYPE)
        self._dst = _GrowableBuffer(ACCOUNT_DTYPE)
        self._t = _GrowableBuffer(TIME_DTYPE)
        self._attr = _GrowableBuffer(ATTR_DTYPE, ncols=self._F)
        self._label = _GrowableBuffer(LABEL_DTYPE)
        self._n = 0
        # account -> [IN sequence, OUT sequence]; each a sorted list of
        # (timestamp, edge_id, pos). Index with int(Direction) (IN=0, OUT=1).
        self._seq: dict[int, list[list[tuple[int, int, int]]]] = {}
        # account -> [IN counterparties, OUT counterparties] as sets.
        self._nbr: dict[int, list[set[int]]] = {}
        self._read_only = False

    # -- size / shape ------------------------------------------------------- #

    @property
    def num_features(self) -> int:
        return self._F

    @property
    def num_edges(self) -> int:
        return self._n

    @property
    def num_accounts(self) -> int:
        return len(self._seq)

    # -- write -------------------------------------------------------------- #

    def _ensure_account(self, account: int) -> None:
        if account not in self._seq:
            self._seq[account] = [[], []]
            self._nbr[account] = [set(), set()]

    def insert(self, batch: TransactionBatch) -> None:
        if self._read_only:
            raise RuntimeError("cannot insert into a read-only snapshot")
        if batch.num_features != self._F:
            raise SchemaError(
                f"batch has F={batch.num_features}, store expects F={self._F}"
            )
        n = len(batch)
        if n == 0:
            return

        base = self._n
        # Append to the columnar log. Labels default to unknown when absent, so
        # an unlabeled batch is never silently recorded as licit.
        self._eid.append(batch.edge_id)
        self._src.append(batch.src)
        self._dst.append(batch.dst)
        self._t.append(batch.timestamp)
        self._attr.append(batch.attr)
        if batch.has_labels:
            assert batch.label is not None
            self._label.append(batch.label)
        else:
            self._label.append(np.full(n, LABEL_UNKNOWN, dtype=LABEL_DTYPE))

        # Index each row into its source's OUT sequence and dest's IN sequence.
        ts, es, us, vs = batch.timestamp, batch.edge_id, batch.src, batch.dst
        in_idx, out_idx = int(Direction.IN), int(Direction.OUT)
        for i in range(n):
            pos = base + i
            t, eid, u, v = int(ts[i]), int(es[i]), int(us[i]), int(vs[i])
            self._ensure_account(u)
            self._ensure_account(v)
            insort(self._seq[u][out_idx], (t, eid, pos))
            insort(self._seq[v][in_idx], (t, eid, pos))
            self._nbr[u][out_idx].add(v)
            self._nbr[v][in_idx].add(u)

        self._n += n

    # -- read --------------------------------------------------------------- #

    def _gather(self, positions: list[int]) -> TransactionBatch:
        """Build a frozen, canonically-ordered batch from global positions.

        ``positions`` must already be in canonical order (they come from the
        sorted sequence index). Fancy-indexing the columnar log yields fresh,
        contiguous arrays detached from the backing buffers.
        """
        if not positions:
            return TransactionBatch.from_records([], num_features=self._F)
        idx = np.asarray(positions, dtype=np.intp)
        return TransactionBatch(
            edge_id=self._eid.data[idx],
            src=self._src.data[idx],
            dst=self._dst.data[idx],
            timestamp=self._t.data[idx],
            attr=self._attr.data[idx],
            label=self._label.data[idx],
            validate=True,
        ).freeze()

    def range_query(
        self,
        account: int,
        direction: Direction,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> TransactionBatch:
        entries = self._seq.get(account)
        if entries is None:
            return TransactionBatch.from_records([], num_features=self._F)
        seq = entries[int(direction)]
        # Half-open search keys. (t, _BELOW_ANY, _BELOW_ANY) sorts strictly
        # before every real tuple whose timestamp is t, so bisect_left lands on
        # the first edge at or after that timestamp.
        lo = 0 if t_start is None else bisect_left(seq, (t_start, _BELOW_ANY, _BELOW_ANY))
        hi = (
            len(seq)
            if t_end is None
            else bisect_left(seq, (t_end + 1, _BELOW_ANY, _BELOW_ANY))
        )
        return self._gather([pos for (_t, _eid, pos) in seq[lo:hi]])

    def tail(self, account: int, direction: Direction, k: int) -> TransactionBatch:
        entries = self._seq.get(account)
        if entries is None or k <= 0:
            return TransactionBatch.from_records([], num_features=self._F)
        seq = entries[int(direction)]
        chosen = seq[-k:] if k < len(seq) else seq
        return self._gather([pos for (_t, _eid, pos) in chosen])

    def sequence_length(self, account: int, direction: Direction) -> int:
        entries = self._seq.get(account)
        return 0 if entries is None else len(entries[int(direction)])

    def neighbors(self, account: int, direction: Direction) -> npt.NDArray[np.int64]:
        entries = self._nbr.get(account)
        if entries is None:
            return np.empty(0, dtype=ACCOUNT_DTYPE)
        return np.array(sorted(entries[int(direction)]), dtype=ACCOUNT_DTYPE)

    def accounts(self) -> npt.NDArray[np.int64]:
        return np.array(sorted(self._seq.keys()), dtype=ACCOUNT_DTYPE)

    def all_edges(self) -> TransactionBatch:
        if self._n == 0:
            return TransactionBatch.from_records([], num_features=self._F)
        # Columnar log is in insertion order; sort into canonical (t, eid) order.
        order = np.lexsort((self._eid.data, self._t.data))
        return self._gather(order.tolist())

    # -- snapshot ----------------------------------------------------------- #

    def snapshot(self) -> "ReferencePyStore":
        """Return a read-only view isolated from the parent's later inserts.

        The columnar log is append-only, so the snapshot safely *shares* the
        parent's buffers (it only ever reads positions ``< num_edges`` at
        snapshot time, whose contents never change). The per-account sequence
        lists and neighbor sets, which the parent mutates in place on insert,
        are copied so the snapshot's results are frozen.
        """
        snap = ReferencePyStore.__new__(ReferencePyStore)
        snap._F = self._F
        snap._eid = self._eid
        snap._src = self._src
        snap._dst = self._dst
        snap._t = self._t
        snap._attr = self._attr
        snap._label = self._label
        snap._n = self._n
        snap._seq = {a: [list(seqs[0]), list(seqs[1])] for a, seqs in self._seq.items()}
        snap._nbr = {a: [set(s[0]), set(s[1])] for a, s in self._nbr.items()}
        snap._read_only = True
        return snap

    def __repr__(self) -> str:
        kind = "snapshot" if self._read_only else "store"
        return (
            f"ReferencePyStore({kind}, edges={self._n}, "
            f"accounts={self.num_accounts}, F={self._F})"
        )
