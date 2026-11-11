"""Exact incremental MGD maintenance (C3, Theorem 2 of docs/proofs.md).

Maintains the per-layer MGD representations of every account under a stream of
edge insertions, recomputing only the nodes whose value can change. On a delta
whose changed Edge2Seq codes are at the endpoint set ``S_0``, layer ``l`` is
refreshed on ``R_l = S_{l-1} ∪ N(S_{l-1})`` -- the only nodes whose layer-``l``
output can differ -- and every other node keeps its stored value. After ``L``
layers the touched set is the ``L``-hop closure of the delta's endpoints.

Each refreshed node is recomputed *exactly* from the current representations of
its multiset neighborhood (parallel edges counted), using the model's own MGD
weights, so the maintained view equals the from-scratch oracle in real
arithmetic. Floating-point results differ only by summation order (per-node sum
here vs. edge-scatter in the oracle).

This is the exact foundation; threshold-based pruning for high-degree hubs
(bounded ε for bounded work) layers on top and is added separately.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from streamdiam.data.schema import Direction
from streamdiam.maintain.affected_set import neighbor_closure
from streamdiam.maintain.inputs import edge_index_from_store
from streamdiam.model.diam import DIAM
from streamdiam.model.mgd import MGDLayer
from streamdiam.store.interface import TemporalMultigraphStore

__all__ = ["IncrementalMGD"]


class IncrementalMGD:
    """Maintain MGD layer representations incrementally over a stream."""

    def __init__(self, model: DIAM, universe_size: int, device: torch.device | str = "cpu"):
        self.model = model.eval()
        self.n = int(universe_size)
        self.c = model.hidden_dim
        self.num_layers = model.num_layers
        self.device = torch.device(device)
        self.codes: Tensor | None = None       # [N, 2c] layer-0 input (Edge2Seq)
        self.h: list[Tensor] = []              # per-layer [N, c]
        self._z = torch.empty((self.n, self.c), device=self.device)  # scratch z buffer

    # -- initialization ----------------------------------------------------- #

    @torch.no_grad()
    def rebuild(self, codes: Tensor, store: TemporalMultigraphStore) -> None:
        """Set the full view by a from-scratch MGD pass over ``codes`` + graph."""
        self.codes = codes.clone().to(self.device)
        edge_index = edge_index_from_store(store, self.device)
        h = self.codes
        self.h = []
        for layer in self.model.layers:
            h = layer(h, edge_index)
            self.h.append(h)

    # -- incremental update ------------------------------------------------- #

    @torch.no_grad()
    def update(
        self,
        changed_nodes: npt.NDArray[np.integer],
        new_codes: Tensor,
        store: TemporalMultigraphStore,
    ) -> npt.NDArray[np.int64]:
        """Apply a delta and return the touched node set.

        Args:
            changed_nodes: accounts whose Edge2Seq code changed (delta endpoints).
            new_codes: ``[len(changed_nodes), 2c]`` the updated code rows, in the
                same order as ``changed_nodes``.
            store: the store **after** the delta was inserted (current adjacency).
        """
        assert self.codes is not None, "call rebuild() before update()"
        changed = np.unique(np.asarray(changed_nodes, dtype=np.int64))
        if changed.size == 0:
            return changed
        idx = torch.from_numpy(changed).to(self.device)
        self.codes[idx] = new_codes.to(self.device)

        s = changed  # nodes whose layer-(l-1) representation changed
        for ell in range(self.num_layers):
            r = neighbor_closure(store, s)  # nodes to refresh at this layer
            h_prev = self.codes if ell == 0 else self.h[ell - 1]
            layer = cast(MGDLayer, self.model.layers[ell])
            # z for every node we will read: R and its neighbors.
            needed = neighbor_closure(store, r)
            nidx = torch.from_numpy(needed).to(self.device)
            self._z[nidx] = layer.lin_z(h_prev[nidx])
            for v in r.tolist():
                self.h[ell][v] = self._recompute_node(v, self._z, layer, store)
            s = r  # touched set propagates one hop per layer
        return s

    def _recompute_node(
        self, v: int, z: Tensor, layer: MGDLayer, store: TemporalMultigraphStore
    ) -> Tensor:
        """Exact layer output for node ``v`` from the current ``z`` of its neighbors."""
        zv = z[v]
        r_in = z.new_zeros(self.c)
        inb = store.scan(v, Direction.IN)
        if len(inb) > 0:
            su = torch.from_numpy(np.array(inb.src, dtype=np.int64)).to(self.device)
            zu = z[su]
            r_in = layer.msg_in(torch.cat([zu, zv.unsqueeze(0) - zu], dim=-1)).sum(0)
        r_out = z.new_zeros(self.c)
        outb = store.scan(v, Direction.OUT)
        if len(outb) > 0:
            du = torch.from_numpy(np.array(outb.dst, dtype=np.int64)).to(self.device)
            zu = z[du]
            r_out = layer.msg_out(torch.cat([zu, zv.unsqueeze(0) - zu], dim=-1)).sum(0)
        stacked = torch.stack([zv, r_in, r_out], dim=0)  # [3, c]
        scores = layer.att_q(torch.tanh(layer.att_w(stacked)))  # [3, 1]
        alpha = torch.softmax(scores, dim=0)
        return (alpha * stacked).sum(0)  # [c]; dropout is identity in eval

    # -- outputs ------------------------------------------------------------ #

    @torch.no_grad()
    def hidden(self) -> Tensor:
        """Final-layer representations ``[N, c]``."""
        return self.h[-1]

    @torch.no_grad()
    def scores(self) -> Tensor:
        """Risk probabilities ``[N]`` from the maintained final representation."""
        logits = self.model.classifier(self.h[-1]).squeeze(-1)
        return torch.sigmoid(logits)
