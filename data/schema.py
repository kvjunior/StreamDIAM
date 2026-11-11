"""Core data model for StreamDIAM: the temporal transaction multigraph.

This module defines the *logical* unit of data that flows through the entire
system -- a directed, timestamped, attributed transaction (an edge) -- and the
*physical* columnar container (a batch, i.e. a graph delta ``dG``) used to move
many of them efficiently. It is the single source of truth for in-memory
dtypes, canonical event ordering, and label encoding, so that the store's
account id, the model's input tensor, and the evaluator's label agree *by
construction* rather than by convention.

Formal data model (authoritative statement in ``docs/problem_formalization.md``):

    A temporal transaction multigraph ``G = (V, E)`` is an append-only sequence
    of directed edges. Each edge ``e = (eid, u, v, t, x)`` is a transaction from
    account ``u`` to account ``v`` at integer time ``t`` carrying a feature
    vector ``x in R^F``. Parallel edges (several transactions between the same
    ordered pair, possibly at the same ``t``) are permitted -- hence a
    *multi* graph.

    The canonical order over edges is the lexicographic order on ``(t, eid)``.
    ``eid`` is assigned in non-decreasing arrival order, so ``(t, eid)`` is a
    total order that (i) breaks timestamp ties deterministically and (ii) never
    orders an event before one that physically preceded it. This total order is
    what makes "no future leakage" checkable and stream replay deterministic.

Scale note:
    At millions of edges we never materialize millions of ``TransactionRecord``
    objects. ``TransactionRecord`` is the logical / API-boundary type (tests,
    small inputs, readable code); ``TransactionBatch`` is the columnar
    representation the system actually moves and the unit of a graph delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "AccountId",
    "EdgeId",
    "Timestamp",
    "EDGE_ID_DTYPE",
    "ACCOUNT_DTYPE",
    "TIME_DTYPE",
    "ATTR_DTYPE",
    "LABEL_DTYPE",
    "LABEL_UNKNOWN",
    "LABEL_LICIT",
    "LABEL_ILLICIT",
    "SchemaError",
    "Direction",
    "TransactionRecord",
    "TransactionBatch",
]

# --------------------------------------------------------------------------- #
# Canonical dtypes and label encoding.
#
# These are the contract between every subsystem. Account ids and timestamps
# are int64: Ethereum block timestamps are Unix seconds (int32 overflows in
# 2038, so int64 is the safe choice) and int64 also matches torch.long, so
# ids cross into tensors without a copy-inducing cast. Features are float32,
# the standard precision for the model; RAW monetary values (e.g. wei ~ 1e18)
# MUST be normalized upstream (log1p / standardization) before entering ``x``,
# because float32 cannot represent 1e18 to integer precision. Labels are a
# compact int8 channel.
# --------------------------------------------------------------------------- #

AccountId = int
EdgeId = int
Timestamp = int

EDGE_ID_DTYPE: np.dtype = np.dtype(np.int64)
ACCOUNT_DTYPE: np.dtype = np.dtype(np.int64)
TIME_DTYPE: np.dtype = np.dtype(np.int64)
ATTR_DTYPE: np.dtype = np.dtype(np.float32)
LABEL_DTYPE: np.dtype = np.dtype(np.int8)

LABEL_UNKNOWN: int = -1
LABEL_LICIT: int = 0
LABEL_ILLICIT: int = 1
_VALID_LABELS = (LABEL_UNKNOWN, LABEL_LICIT, LABEL_ILLICIT)


class SchemaError(ValueError):
    """Raised when data violates the temporal-multigraph data model.

    Subclasses :class:`ValueError` so that callers already guarding against
    bad input (``except ValueError``) keep working, while code that wants to
    distinguish schema violations specifically can catch this type.
    """


class Direction(IntEnum):
    """Which per-account sequence an edge belongs to, from an account's view.

    For account ``w``, an edge with ``dst == w`` is :attr:`IN` (it is in
    ``S_in(w)``) and an edge with ``src == w`` is :attr:`OUT` (in ``S_out(w)``).
    The integer values are stable and may be used to index length-2 arrays.
    """

    IN = 0
    OUT = 1


# --------------------------------------------------------------------------- #
# The logical unit: a single transaction (one directed, attributed edge).
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True, eq=False)
class TransactionRecord:
    """One directed, timestamped, attributed transaction (a graph edge).

    Identity is the primary key ``edge_id`` alone: two records with the same
    ``edge_id`` denote the same transaction, so equality and hashing use
    ``edge_id`` only (not the attribute vector, which is not hashable and whose
    element-wise comparison would be ambiguous). Ordering is the canonical
    ``(timestamp, edge_id)`` order of the data model.

    The instance is immutable: the dataclass is frozen, and the attribute
    vector is stored as a C-contiguous, read-only ``float32`` array so its
    contents cannot be mutated through an alias either.

    Attributes:
        edge_id: Globally unique id, assigned in non-decreasing arrival order.
        src: Source account id (the sender, ``u``).
        dst: Destination account id (the receiver, ``v``).
        timestamp: Integer event time ``t`` (e.g. Unix seconds).
        attr: Feature vector ``x`` of shape ``(F,)``, dtype ``float32``.
        label: Ground-truth label of the *subject* account for this sample,
            one of ``LABEL_UNKNOWN`` (default), ``LABEL_LICIT``,
            ``LABEL_ILLICIT``. Defaults to unknown because most transactions
            are unlabeled.
    """

    edge_id: int
    src: int
    dst: int
    timestamp: int
    attr: npt.NDArray[np.float32]
    label: int = LABEL_UNKNOWN

    def __post_init__(self) -> None:
        # Normalize scalar fields to plain Python ints so that a record built
        # from numpy scalars (e.g. during ``TransactionBatch.__getitem__``)
        # compares and hashes identically to one built from Python literals.
        # ``object.__setattr__`` is required because the dataclass is frozen.
        object.__setattr__(self, "edge_id", int(self.edge_id))
        object.__setattr__(self, "src", int(self.src))
        object.__setattr__(self, "dst", int(self.dst))
        object.__setattr__(self, "timestamp", int(self.timestamp))
        object.__setattr__(self, "label", int(self.label))

        if self.edge_id < 0:
            raise SchemaError(f"edge_id must be non-negative, got {self.edge_id}")
        if self.src < 0 or self.dst < 0:
            raise SchemaError(
                f"account ids must be non-negative, got src={self.src}, dst={self.dst}"
            )
        if self.label not in _VALID_LABELS:
            raise SchemaError(
                f"label must be one of {_VALID_LABELS}, got {self.label}"
            )

        # Coerce the feature vector to a contiguous float32 1-D array, then
        # freeze it. ``ascontiguousarray`` copies only if necessary.
        attr = np.ascontiguousarray(self.attr, dtype=ATTR_DTYPE)
        if attr.ndim != 1:
            raise SchemaError(
                f"attr must be 1-D of shape (F,), got shape {attr.shape}"
            )
        if attr.size == 0:
            raise SchemaError("attr must have at least one feature (F >= 1)")
        attr.flags.writeable = False
        object.__setattr__(self, "attr", attr)

    @property
    def num_features(self) -> int:
        """Feature dimension ``F``."""
        return int(self.attr.shape[0])

    @property
    def sort_key(self) -> tuple[int, int]:
        """The canonical ordering key ``(timestamp, edge_id)``."""
        return (self.timestamp, self.edge_id)

    def endpoint(self, direction: Direction) -> int:
        """Return the subject account under ``direction``.

        ``IN`` -> the receiver ``dst`` (this edge is in ``S_in(dst)``);
        ``OUT`` -> the sender ``src`` (this edge is in ``S_out(src)``).
        """
        return self.dst if direction is Direction.IN else self.src

    # -- identity: primary key ``edge_id`` only ----------------------------- #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransactionRecord):
            return NotImplemented
        return self.edge_id == other.edge_id

    def __hash__(self) -> int:
        return hash(self.edge_id)

    # -- ordering: canonical ``(timestamp, edge_id)`` ----------------------- #

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TransactionRecord):
            return NotImplemented
        return self.sort_key < other.sort_key

    def __le__(self, other: object) -> bool:
        if not isinstance(other, TransactionRecord):
            return NotImplemented
        return self.sort_key <= other.sort_key

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, TransactionRecord):
            return NotImplemented
        return self.sort_key > other.sort_key

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, TransactionRecord):
            return NotImplemented
        return self.sort_key >= other.sort_key

    def __repr__(self) -> str:
        label_txt = {
            LABEL_UNKNOWN: "?",
            LABEL_LICIT: "licit",
            LABEL_ILLICIT: "illicit",
        }[self.label]
        return (
            f"TransactionRecord(eid={self.edge_id}, {self.src}->{self.dst}, "
            f"t={self.timestamp}, F={self.num_features}, y={label_txt})"
        )


# --------------------------------------------------------------------------- #
# The physical unit: a columnar batch of transactions == a graph delta dG.
# --------------------------------------------------------------------------- #

class TransactionBatch:
    """A columnar, memory-efficient set of transactions; the unit of a delta.

    Stores five parallel arrays of length ``n`` (``edge_id``, ``src``, ``dst``,
    ``timestamp``, and the ``(n, F)`` ``attr`` matrix) plus an optional length
    ``n`` ``label`` array. This is the representation the system moves at scale
    and the concrete form of a graph delta ``dG`` (``n == 0`` is a valid empty
    delta). Construction validates dtypes and shapes by default; pass
    ``validate=False`` on an internally-produced batch (e.g. a slice) to skip
    the O(n) checks.

    The batch is treated as immutable. :meth:`freeze` additionally makes the
    backing arrays read-only; :meth:`from_arrays` and :meth:`from_records`
    return frozen batches.
    """

    __slots__ = ("edge_id", "src", "dst", "timestamp", "attr", "label")

    def __init__(
        self,
        edge_id: npt.NDArray[np.int64],
        src: npt.NDArray[np.int64],
        dst: npt.NDArray[np.int64],
        timestamp: npt.NDArray[np.int64],
        attr: npt.NDArray[np.float32],
        label: npt.NDArray[np.int8] | None = None,
        *,
        validate: bool = True,
    ) -> None:
        self.edge_id = edge_id
        self.src = src
        self.dst = dst
        self.timestamp = timestamp
        self.attr = attr
        self.label = label
        if validate:
            self.validate()

    # -- validation --------------------------------------------------------- #

    def validate(self) -> None:
        """Check the data-model invariants; raise :class:`SchemaError` if any fail.

        Verifies array types, the canonical dtypes, 1-D/2-D shapes, equal
        lengths across all columns, and that labels (if present) are drawn from
        the valid set. Does *not* check ordering -- use :meth:`is_time_sorted`.
        """
        cols: dict[str, tuple[np.ndarray, np.dtype, int]] = {
            "edge_id": (self.edge_id, EDGE_ID_DTYPE, 1),
            "src": (self.src, ACCOUNT_DTYPE, 1),
            "dst": (self.dst, ACCOUNT_DTYPE, 1),
            "timestamp": (self.timestamp, TIME_DTYPE, 1),
            "attr": (self.attr, ATTR_DTYPE, 2),
        }
        for name, (arr, dtype, ndim) in cols.items():
            if not isinstance(arr, np.ndarray):
                raise SchemaError(f"{name} must be a numpy.ndarray, got {type(arr)!r}")
            if arr.dtype != dtype:
                raise SchemaError(
                    f"{name} must have dtype {dtype}, got {arr.dtype}"
                )
            if arr.ndim != ndim:
                raise SchemaError(
                    f"{name} must be {ndim}-D, got {arr.ndim}-D (shape {arr.shape})"
                )

        n = self.edge_id.shape[0]
        for name in ("src", "dst", "timestamp"):
            got = cols[name][0].shape[0]
            if got != n:
                raise SchemaError(
                    f"column length mismatch: edge_id has {n} rows but {name} has {got}"
                )
        if self.attr.shape[0] != n:
            raise SchemaError(
                f"attr must have {n} rows (one per edge), got {self.attr.shape[0]}"
            )
        if n > 0 and self.attr.shape[1] < 1:
            raise SchemaError("attr must have at least one feature column (F >= 1)")

        if self.label is not None:
            if not isinstance(self.label, np.ndarray):
                raise SchemaError(
                    f"label must be a numpy.ndarray or None, got {type(self.label)!r}"
                )
            if self.label.dtype != LABEL_DTYPE:
                raise SchemaError(
                    f"label must have dtype {LABEL_DTYPE}, got {self.label.dtype}"
                )
            if self.label.ndim != 1 or self.label.shape[0] != n:
                raise SchemaError(
                    f"label must be 1-D of length {n}, got shape {self.label.shape}"
                )
            if n > 0:
                lo = int(self.label.min())
                hi = int(self.label.max())
                if lo < LABEL_UNKNOWN or hi > LABEL_ILLICIT:
                    raise SchemaError(
                        f"labels must be in {_VALID_LABELS}, got range [{lo}, {hi}]"
                    )

    def freeze(self) -> "TransactionBatch":
        """Make the backing arrays read-only and return ``self`` (for chaining)."""
        for arr in (self.edge_id, self.src, self.dst, self.timestamp, self.attr):
            arr.flags.writeable = False
        if self.label is not None:
            self.label.flags.writeable = False
        return self

    # -- constructors ------------------------------------------------------- #

    @classmethod
    def from_arrays(
        cls,
        edge_id: npt.ArrayLike,
        src: npt.ArrayLike,
        dst: npt.ArrayLike,
        timestamp: npt.ArrayLike,
        attr: npt.ArrayLike,
        label: npt.ArrayLike | None = None,
    ) -> "TransactionBatch":
        """Build a validated, frozen batch, casting inputs to canonical dtypes.

        Unlike the raw constructor, this coerces each input to the required
        dtype and C-contiguous layout (copying only when necessary), so callers
        may pass e.g. Python lists or int32 arrays.
        """
        attr_arr = np.ascontiguousarray(attr, dtype=ATTR_DTYPE)
        if attr_arr.ndim == 1:
            # Interpret a 1-D attr as a single-feature column, not a single row;
            # a bare list of per-edge scalars is the common accidental input.
            attr_arr = attr_arr.reshape(-1, 1)
        label_arr = (
            None if label is None else np.ascontiguousarray(label, dtype=LABEL_DTYPE)
        )
        batch = cls(
            edge_id=np.ascontiguousarray(edge_id, dtype=EDGE_ID_DTYPE),
            src=np.ascontiguousarray(src, dtype=ACCOUNT_DTYPE),
            dst=np.ascontiguousarray(dst, dtype=ACCOUNT_DTYPE),
            timestamp=np.ascontiguousarray(timestamp, dtype=TIME_DTYPE),
            attr=attr_arr,
            label=label_arr,
            validate=True,
        )
        return batch.freeze()

    @classmethod
    def from_records(
        cls,
        records: Sequence[TransactionRecord],
        num_features: int | None = None,
    ) -> "TransactionBatch":
        """Assemble a frozen batch from logical records.

        Args:
            records: The records; may be empty.
            num_features: Required feature dimension when ``records`` is empty
                (there is no row to infer ``F`` from); otherwise optional and,
                if given, checked against every record.
        """
        n = len(records)
        if n == 0:
            if num_features is None:
                raise SchemaError(
                    "num_features is required to build a batch from zero records"
                )
            if num_features < 1:
                raise SchemaError(f"num_features must be >= 1, got {num_features}")
            return cls(
                edge_id=np.empty(0, dtype=EDGE_ID_DTYPE),
                src=np.empty(0, dtype=ACCOUNT_DTYPE),
                dst=np.empty(0, dtype=ACCOUNT_DTYPE),
                timestamp=np.empty(0, dtype=TIME_DTYPE),
                attr=np.empty((0, num_features), dtype=ATTR_DTYPE),
                label=np.empty(0, dtype=LABEL_DTYPE),
                validate=True,
            ).freeze()

        f = records[0].num_features if num_features is None else int(num_features)
        edge_id = np.empty(n, dtype=EDGE_ID_DTYPE)
        src = np.empty(n, dtype=ACCOUNT_DTYPE)
        dst = np.empty(n, dtype=ACCOUNT_DTYPE)
        timestamp = np.empty(n, dtype=TIME_DTYPE)
        attr = np.empty((n, f), dtype=ATTR_DTYPE)
        label = np.empty(n, dtype=LABEL_DTYPE)
        for i, rec in enumerate(records):
            if rec.num_features != f:
                raise SchemaError(
                    f"record {i} has F={rec.num_features}, expected F={f}"
                )
            edge_id[i] = rec.edge_id
            src[i] = rec.src
            dst[i] = rec.dst
            timestamp[i] = rec.timestamp
            attr[i] = rec.attr
            label[i] = rec.label
        return cls(
            edge_id=edge_id,
            src=src,
            dst=dst,
            timestamp=timestamp,
            attr=attr,
            label=label,
            validate=True,
        ).freeze()

    @classmethod
    def concat(cls, batches: Iterable["TransactionBatch"]) -> "TransactionBatch":
        """Concatenate batches (preserving row order) into one frozen batch.

        All batches must share the feature dimension ``F``. Labels are kept
        only if *every* batch carries them; otherwise the result is unlabeled,
        so that a missing-label batch is never silently backfilled with zeros.
        """
        batches = list(batches)
        if not batches:
            raise SchemaError("concat requires at least one batch")
        f = batches[0].num_features
        for b in batches[1:]:
            if b.num_features != f:
                raise SchemaError(
                    f"cannot concat batches with F={f} and F={b.num_features}"
                )
        has_labels = all(b.label is not None for b in batches)
        label = (
            np.concatenate([b.label for b in batches])  # type: ignore[misc]
            if has_labels
            else None
        )
        return cls(
            edge_id=np.concatenate([b.edge_id for b in batches]),
            src=np.concatenate([b.src for b in batches]),
            dst=np.concatenate([b.dst for b in batches]),
            timestamp=np.concatenate([b.timestamp for b in batches]),
            attr=np.concatenate([b.attr for b in batches], axis=0),
            label=label,
            validate=True,
        ).freeze()

    # -- size / shape ------------------------------------------------------- #

    def __len__(self) -> int:
        return int(self.edge_id.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def num_features(self) -> int:
        """Feature dimension ``F`` (well-defined even for an empty batch)."""
        return int(self.attr.shape[1])

    @property
    def has_labels(self) -> bool:
        return self.label is not None

    def time_span(self) -> tuple[int, int]:
        """Return ``(min_timestamp, max_timestamp)``; raises on an empty batch.

        A time span is undefined with no events; callers guard with ``len`` /
        :attr:`is_empty`. Not assuming the batch is sorted, this scans the
        column rather than reading the endpoints.
        """
        if self.is_empty:
            raise SchemaError("time_span is undefined for an empty batch")
        return (int(self.timestamp.min()), int(self.timestamp.max()))

    # -- ordering ----------------------------------------------------------- #

    def is_time_sorted(self) -> bool:
        """True iff rows are non-decreasing in the canonical ``(t, eid)`` order.

        Runs in O(n): a prefix is canonically sorted iff for every adjacent
        pair either the timestamp strictly increases, or the timestamp is equal
        and the edge id does not decrease. Empty and singleton batches are
        trivially sorted.
        """
        n = len(self)
        if n <= 1:
            return True
        t = self.timestamp
        e = self.edge_id
        t_prev, t_next = t[:-1], t[1:]
        e_prev, e_next = e[:-1], e[1:]
        ok = (t_next > t_prev) | ((t_next == t_prev) & (e_next >= e_prev))
        return bool(np.all(ok))

    def sorted_by_time(self) -> "TransactionBatch":
        """Return a new frozen batch reordered into canonical ``(t, eid)`` order.

        Uses ``np.lexsort`` with ``timestamp`` as the primary key and
        ``edge_id`` as the tie-breaker (lexsort treats its last key as primary).
        Returns ``self`` unchanged when already sorted.
        """
        if self.is_time_sorted():
            return self
        order = np.lexsort((self.edge_id, self.timestamp))
        return self._take(order)

    # -- indexing / iteration ---------------------------------------------- #

    def _take(self, index: npt.NDArray[np.intp]) -> "TransactionBatch":
        """Gather rows by integer index array into a new frozen batch."""
        label = None if self.label is None else self.label[index]
        return TransactionBatch(
            edge_id=self.edge_id[index],
            src=self.src[index],
            dst=self.dst[index],
            timestamp=self.timestamp[index],
            attr=self.attr[index],
            label=label,
            validate=False,  # gather preserves dtypes/shapes/label validity
        ).freeze()

    def record_at(self, i: int) -> TransactionRecord:
        """Materialize the ``i``-th row as a :class:`TransactionRecord`.

        The record receives an independent, read-only copy of the feature row,
        so it never aliases the batch's ``attr`` matrix.
        """
        label = LABEL_UNKNOWN if self.label is None else int(self.label[i])
        return TransactionRecord(
            edge_id=int(self.edge_id[i]),
            src=int(self.src[i]),
            dst=int(self.dst[i]),
            timestamp=int(self.timestamp[i]),
            attr=self.attr[i].copy(),
            label=label,
        )

    def __getitem__(self, key: int | slice) -> "TransactionRecord | TransactionBatch":
        if isinstance(key, (int, np.integer)):
            n = len(self)
            i = int(key)
            if i < 0:
                i += n
            if not 0 <= i < n:
                raise IndexError(f"index {key} out of range for batch of size {n}")
            return self.record_at(i)
        if isinstance(key, slice):
            label = None if self.label is None else self.label[key]
            return TransactionBatch(
                edge_id=self.edge_id[key],
                src=self.src[key],
                dst=self.dst[key],
                timestamp=self.timestamp[key],
                attr=self.attr[key],
                label=label,
                validate=False,  # a slice preserves all invariants
            ).freeze()
        raise TypeError(f"batch indices must be int or slice, not {type(key).__name__}")

    def __iter__(self) -> Iterator[TransactionRecord]:
        for i in range(len(self)):
            yield self.record_at(i)

    def to_records(self) -> list[TransactionRecord]:
        """Materialize every row as a list of records (small inputs / tests)."""
        return list(self)

    def __repr__(self) -> str:
        span = "empty" if self.is_empty else f"t in {self.time_span()}"
        lbl = "labeled" if self.has_labels else "unlabeled"
        return (
            f"TransactionBatch(n={len(self)}, F={self.num_features}, {span}, {lbl})"
        )
