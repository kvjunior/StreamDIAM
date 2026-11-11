"""The from-scratch maintainer: recompute the full inference view Phi(G).

This is the semantics the incremental maintainer (C3) must reproduce cheaply,
and the oracle that ``tests/test_maintain.py`` and future equivalence tests
check against. It runs the model exactly as a full forward pass would, but:

* Edge2Seq codes are computed in **chunks** of accounts to bound the padded-
  sequence memory (encodings are per-account independent, so the result is
  identical to a single pass); then
* the MGD layers run once over the whole node tensor and the whole edge index.

No gradients and no caching: this is a reference computation, deliberately
simple so its correctness is self-evident.
"""

from __future__ import annotations

import numpy as np
import torch

from streamdiam.maintain.inputs import build_sequence_inputs, edge_index_from_store
from streamdiam.maintain.view import InferenceView
from streamdiam.model.diam import DIAM
from streamdiam.store.interface import TemporalMultigraphStore

__all__ = ["FromScratchMaintainer"]


class FromScratchMaintainer:
    """Recompute the entire inference view over the current graph."""

    def __init__(
        self,
        model: DIAM,
        t_max: int = 32,
        chunk_size: int = 4096,
        device: torch.device | str = "cpu",
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        self.model = model
        self.t_max = t_max
        self.chunk_size = chunk_size
        self.device = torch.device(device)

    @torch.no_grad()
    def compute_view(
        self, store: TemporalMultigraphStore, universe_size: int
    ) -> InferenceView:
        """Materialize ``Phi(G)`` over accounts ``[0, universe_size)``.

        ``universe_size`` must exceed every account id present in the store's
        edges (the model indexes node tensors by id).
        """
        if store.num_features != self.model.in_dim:
            raise ValueError(
                f"store F={store.num_features} != model in_dim={self.model.in_dim}"
            )
        n = int(universe_size)
        if n < 1:
            raise ValueError(f"universe_size must be >= 1, got {n}")
        edges = store.all_edges()
        if len(edges) > 0 and int(max(edges.src.max(), edges.dst.max())) >= n:
            raise ValueError("universe_size must exceed every account id in the store")

        self.model.eval()
        model = self.model.to(self.device)
        c = model.hidden_dim

        # 1) Edge2Seq codes for every account, computed chunk by chunk.
        codes = torch.empty((n, 2 * c), device=self.device)
        for start in range(0, n, self.chunk_size):
            accts = np.arange(start, min(start + self.chunk_size, n), dtype=np.int64)
            si = build_sequence_inputs(store, accts, self.t_max, self.device)
            codes[start : start + accts.shape[0]] = model.edge2seq(
                si.in_seq, si.in_len, si.out_seq, si.out_len
            )

        # 2) MGD layers over the whole graph, then the risk head.
        edge_index = edge_index_from_store(store, self.device)
        h = codes
        layer_hidden: list[torch.Tensor] = []
        for layer in model.layers:
            h = layer(h, edge_index)
            layer_hidden.append(h)
        logits = model.classifier(h).squeeze(-1)
        scores = torch.sigmoid(logits)
        return InferenceView(
            codes=codes,
            layer_hidden=layer_hidden,
            logits=logits,
            scores=scores,
            universe_size=n,
        )
