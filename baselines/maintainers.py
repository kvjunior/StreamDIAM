"""View-maintenance strategies for the streaming harness.

Every strategy here is a :class:`~streamdiam.eval.protocol.StreamingMaintainer`,
so all of them run through the identical prequential loop and their numbers are
directly comparable (the point of C5). Three strategies share the loop:

* ``from_scratch`` (in ``eval.protocol``) -- recompute the whole view each tick.
  The correctness oracle and the from-scratch baseline.
* :class:`IncrementalStreamingMaintainer` -- **the proposed method (C3)**:
  incremental Edge2Seq state extension (B7) + exact affected-set MGD maintenance
  (B8). Produces the same scores as from-scratch (proven per-tick in
  ``test_baselines``) for a fraction of the work.
* :class:`FrozenViewMaintainer` -- compute once, never update on new edges. The
  "does inference maintenance even matter?" ablation: its scores go stale as the
  graph grows, so any gap to the maintained strategies is the value of C3.

All three pick up a hot-swapped model on :meth:`rebuild`, so they compose with
the decoupled retraining of C4.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

from streamdiam.data.schema import TransactionBatch
from streamdiam.eval.protocol import StreamingMaintainer
from streamdiam.maintain.from_scratch import FromScratchMaintainer
from streamdiam.maintain.incremental_edge2seq import IncrementalEdge2Seq
from streamdiam.maintain.incremental_mgd import IncrementalMGD
from streamdiam.model.diam import DIAM
from streamdiam.store.interface import TemporalMultigraphStore

__all__ = ["IncrementalStreamingMaintainer", "FrozenViewMaintainer"]


class IncrementalStreamingMaintainer(StreamingMaintainer):
    """The proposed C3 maintainer, wired into the streaming harness.

    On each delta it (1) extends the affected accounts' Edge2Seq GRU states in
    O(c) per edge, (2) reads their updated codes, and (3) refreshes exactly the
    affected L-hop MGD closure -- never touching the rest of the graph. The
    result equals the from-scratch view within floating-point summation order.

    ``rebuild`` (initial view, or after a hot-swap) reconstructs the incremental
    state from the current model by replaying the store once; this is the only
    full pass and it happens rarely (startup and drift-triggered swaps).
    """

    def __init__(
        self,
        model_provider: Callable[[], DIAM],
        store: TemporalMultigraphStore,
        universe_size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self._model_provider = model_provider
        self._store = store
        self._n = int(universe_size)
        self._device = torch.device(device)
        self._e2s: IncrementalEdge2Seq | None = None
        self._mgd: IncrementalMGD | None = None

    @property
    def name(self) -> str:
        return "incremental"

    def rebuild(self) -> None:
        model = self._model_provider()
        # Fresh incremental state under the (possibly swapped) model: replay the
        # whole store so per-account GRU states match a full pass.
        e2s = IncrementalEdge2Seq(model.edge2seq, self._device)
        all_edges = self._store.all_edges()
        if len(all_edges) > 0:
            e2s.on_batch(all_edges)
        codes = e2s.codes(self._n)
        mgd = IncrementalMGD(model, self._n, self._device)
        mgd.rebuild(codes, self._store)
        self._e2s, self._mgd = e2s, mgd

    def ingest(self, delta: TransactionBatch) -> None:
        assert self._e2s is not None and self._mgd is not None, "rebuild() first"
        # 1) extend GRU states for the delta's endpoints
        self._e2s.on_batch(delta)
        # 2) gather the changed accounts' updated codes
        changed = np.unique(np.concatenate([delta.src, delta.dst])).astype(np.int64)
        new_codes = torch.stack([self._e2s.code(int(a)) for a in changed], dim=0)
        # 3) exact affected-set MGD refresh (store already holds the delta)
        self._mgd.update(changed, new_codes, self._store)

    def scores(self) -> npt.NDArray[np.float64]:
        assert self._mgd is not None, "rebuild() first"
        return self._mgd.scores().detach().cpu().numpy().astype(np.float64)


class FrozenViewMaintainer(StreamingMaintainer):
    """Ablation: compute the view once and never update it on new edges.

    ``ingest`` is a no-op, so scores reflect only the graph as it stood at the
    last ``rebuild``. Isolates the value of inference maintenance: whatever
    accuracy the maintained strategies hold that this one loses is exactly what
    C3 buys. (Distinct from frozen *weights* -- that is a retrain policy.)
    """

    def __init__(
        self,
        model_provider: Callable[[], DIAM],
        store: TemporalMultigraphStore,
        universe_size: int,
        t_max: int = 10_000,
        chunk_size: int = 10_000,
    ) -> None:
        self._model_provider = model_provider
        self._store = store
        self._n = int(universe_size)
        self._t_max = t_max
        self._chunk = chunk_size
        self._scores = np.zeros(self._n, dtype=np.float64)

    @property
    def name(self) -> str:
        return "frozen_view"

    def rebuild(self) -> None:
        view = FromScratchMaintainer(
            self._model_provider(), t_max=self._t_max, chunk_size=self._chunk
        ).compute_view(self._store, self._n)
        self._scores = view.scores.detach().cpu().numpy().astype(np.float64)

    def ingest(self, delta: TransactionBatch) -> None:
        return  # deliberately stale: the view is not updated for new edges

    def scores(self) -> npt.NDArray[np.float64]:
        return self._scores
