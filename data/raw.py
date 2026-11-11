"""Canonical in-memory form a dataset loader produces, before streaming.

Every dataset reader (Elliptic, and later the Ethereum/Bitcoin account
multigraphs) parses its raw files into one :class:`RawTemporalGraph`: a single
time-sorted edge table plus a per-account ground-truth label vector over a
contiguous account-id space. Downstream code (temporal split, stream replay,
the store, the evaluator) consumes only this type, so no component depends on a
dataset's on-disk quirks.

Account ids in a ``RawTemporalGraph`` are always a contiguous range
``[0, num_accounts)`` so the model can index arrays and embeddings by id
directly; :func:`remap_accounts` performs the (bijective) remap and records the
original ids for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from streamdiam.data.schema import (
    ACCOUNT_DTYPE,
    LABEL_DTYPE,
    LABEL_ILLICIT,
    LABEL_UNKNOWN,
    SchemaError,
    TransactionBatch,
)

__all__ = ["RawTemporalGraph", "remap_accounts"]


def remap_accounts(
    original: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Map arbitrary integer account ids to a contiguous ``[0, N)`` code space.

    Returns ``(codes, uniques)`` where ``codes[i]`` is the contiguous id of
    ``original[i]`` and ``uniques`` is the sorted array of distinct original
    ids, so ``uniques[codes] == original`` and ``uniques[c]`` recovers the
    original id of code ``c``. The mapping is a bijection between the distinct
    original ids and ``[0, N)``.
    """
    uniques, inverse = np.unique(np.asarray(original), return_inverse=True)
    return inverse.ravel().astype(ACCOUNT_DTYPE), uniques.astype(np.int64)


@dataclass(frozen=True, eq=False)
class RawTemporalGraph:
    """A fully-parsed temporal multigraph on a contiguous account-id space.

    Attributes:
        name: Dataset identifier (e.g. ``"elliptic"``).
        edges: All transactions as one frozen, canonically ``(t, eid)``-sorted
            :class:`TransactionBatch` with ``edge_id == 0..M-1``. Edges are
            unlabeled; ground truth is per-account, in :attr:`account_label`.
        account_label: Length-``num_accounts`` ``int8`` vector indexed by
            account id, each entry in ``{LABEL_UNKNOWN, LABEL_LICIT,
            LABEL_ILLICIT}``.
        num_accounts: Size ``N`` of the account-id space ``[0, N)``.
        num_features: Edge feature dimension ``F``.
        extra: Dataset-specific provenance (original-id map, node times, counts
            of dropped rows, number of time steps, …); never read by the model.
    """

    name: str
    edges: TransactionBatch
    account_label: npt.NDArray[np.int8]
    num_accounts: int
    num_features: int
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_accounts < 0:
            raise SchemaError(f"num_accounts must be >= 0, got {self.num_accounts}")
        if self.edges.num_features != self.num_features:
            raise SchemaError(
                f"edges have F={self.edges.num_features} but graph declares "
                f"F={self.num_features}"
            )
        if self.account_label.dtype != LABEL_DTYPE:
            raise SchemaError(
                f"account_label must be {LABEL_DTYPE}, got {self.account_label.dtype}"
            )
        if self.account_label.shape != (self.num_accounts,):
            raise SchemaError(
                f"account_label must have shape ({self.num_accounts},), "
                f"got {self.account_label.shape}"
            )
        if len(self.edges) > 0:
            if int(self.edges.src.max()) >= self.num_accounts or int(
                self.edges.dst.max()
            ) >= self.num_accounts:
                raise SchemaError("edge endpoints reference ids outside [0, N)")
            if int(self.edges.src.min()) < 0 or int(self.edges.dst.min()) < 0:
                raise SchemaError("edge endpoints must be non-negative")
            if not self.edges.is_time_sorted():
                raise SchemaError("edges must be in canonical (t, eid) order")

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def label_counts(self) -> dict[int, int]:
        """Counts of illicit / licit / unknown accounts."""
        values, counts = np.unique(self.account_label, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    @property
    def illicit_ratio(self) -> float:
        """Fraction of *labeled* accounts that are illicit (0 if none labeled)."""
        counts = self.label_counts()
        pos = counts.get(LABEL_ILLICIT, 0)
        labeled = self.num_accounts - counts.get(LABEL_UNKNOWN, 0)
        return pos / labeled if labeled else 0.0

    def __repr__(self) -> str:
        return (
            f"RawTemporalGraph(name={self.name!r}, N={self.num_accounts}, "
            f"M={self.num_edges}, F={self.num_features}, "
            f"illicit_ratio={self.illicit_ratio:.4f})"
        )
