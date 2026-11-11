"""Abstract interface for the temporal multigraph store (C2 — the DB substrate).

The store owns the append-only transaction log and serves the per-account
in/out sequence queries defined in ``docs/problem_formalization.md`` §1.2: for
an account ``w`` and a :class:`~streamdiam.data.schema.Direction`, return the
canonically ordered subsequence ``S_in(w)`` / ``S_out(w)``, optionally
restricted to a time window. These are exactly the reads the ``Edge2Seq``
encoder performs and the reads the incremental maintainer (C3) extends.

Two concrete backends implement this interface:

* ``ReferencePyStore`` — a pure-Python implementation that is simple enough to
  audit line by line. It is the **correctness oracle**: the optional native
  (C++/pybind11) store must return identical results.
* the native store — the latency-optimized backend used for the systems claims.

Design contract (every implementation must honor it):

* **Append-only.** Only ``insert`` grows the store; transactions are never
  updated or deleted (§3.1). Inserts need not arrive in timestamp order; the
  store is responsible for maintaining each per-account sequence in canonical
  ``(timestamp, edge_id)`` order regardless of arrival order.
* **Canonical results.** Every sequence returned is ordered by ``(t, eid)``.
* **Fixed F.** The feature dimension ``F`` is fixed at construction so that an
  empty result is still a well-typed, ``F``-column :class:`TransactionBatch`.
* **Snapshot isolation.** :meth:`snapshot` returns a read-only view whose query
  results are unaffected by inserts applied to the parent afterwards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from streamdiam.data.schema import Direction, TransactionBatch

__all__ = ["TemporalMultigraphStore"]


class TemporalMultigraphStore(ABC):
    """Abstract append-only store over a temporal transaction multigraph.

    Subclasses implement the five abstract primitives (:meth:`insert`,
    :meth:`range_query`, :meth:`sequence_length`, :meth:`neighbors`,
    :meth:`accounts`, :meth:`snapshot`) and the three size properties.
    :meth:`scan` and :meth:`tail` are provided here as template methods built
    on :meth:`range_query`; a backend may override them for efficiency but
    inherits correct behavior for free.
    """

    # -- size / shape ------------------------------------------------------- #

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Feature dimension ``F`` (fixed at construction)."""

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """Total number of transactions inserted so far."""

    @property
    @abstractmethod
    def num_accounts(self) -> int:
        """Number of distinct accounts observed as a source or destination."""

    # -- write -------------------------------------------------------------- #

    @abstractmethod
    def insert(self, batch: TransactionBatch) -> None:
        """Append a delta ``dG`` (a batch of transactions).

        The batch's feature dimension must equal :attr:`num_features`. An empty
        batch is a no-op. Arrival order is free: the store re-establishes
        canonical order per account. Must raise if called on a snapshot.
        """

    # -- read primitives ---------------------------------------------------- #

    @abstractmethod
    def range_query(
        self,
        account: int,
        direction: Direction,
        t_start: int | None = None,
        t_end: int | None = None,
    ) -> TransactionBatch:
        """Return ``S_dir(account)`` restricted to ``t in [t_start, t_end]``.

        ``t_start``/``t_end`` are inclusive; ``None`` means unbounded on that
        side. The result is a frozen :class:`TransactionBatch` in canonical
        ``(timestamp, edge_id)`` order (empty, with ``F`` columns, for an
        unknown account or an empty window). Target cost: ``O(log s + k)`` for a
        sequence of length ``s`` returning ``k`` edges.
        """

    @abstractmethod
    def sequence_length(self, account: int, direction: Direction) -> int:
        """Length of ``S_dir(account)`` (0 for an unknown account)."""

    @abstractmethod
    def neighbors(self, account: int, direction: Direction) -> npt.NDArray[np.int64]:
        """Distinct counterparties of ``account`` along ``direction``.

        For :attr:`Direction.IN`, the senders of ``account``'s incoming edges;
        for :attr:`Direction.OUT`, the receivers of its outgoing edges. Returned
        as a sorted ``int64`` array (empty for an unknown account). Used by the
        MGD message passing and by affected-set computation (C3).
        """

    @abstractmethod
    def accounts(self) -> npt.NDArray[np.int64]:
        """All distinct account ids, sorted ascending."""

    @abstractmethod
    def all_edges(self) -> TransactionBatch:
        """Every transaction in the store, in canonical ``(t, eid)`` order.

        The MGD message passing consumes the full multigraph adjacency; its
        ``edge_index`` is ``[src; dst]`` of this batch. Returned frozen and
        deterministic (empty, with ``F`` columns, when the store is empty).
        """

    @abstractmethod
    def snapshot(self) -> "TemporalMultigraphStore":
        """Return a read-only, point-in-time view with snapshot isolation."""

    # -- template methods (built on range_query) ---------------------------- #

    def scan(self, account: int, direction: Direction) -> TransactionBatch:
        """Return the full canonical sequence ``S_dir(account)``."""
        return self.range_query(account, direction, None, None)

    def tail(self, account: int, direction: Direction, k: int) -> TransactionBatch:
        """Return the last ``k`` edges of ``S_dir(account)`` in canonical order.

        ``k <= 0`` yields an empty batch; ``k`` larger than the sequence yields
        the whole sequence. Backends with an indexed sequence should override
        this to avoid materializing the full scan.
        """
        full = self.scan(account, direction)
        start = max(0, len(full) - k) if k > 0 else len(full)
        result = full[start:]
        assert isinstance(result, TransactionBatch)  # slice of a batch is a batch
        return result
