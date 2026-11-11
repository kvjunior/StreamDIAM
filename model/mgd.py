"""Multigraph Discrepancy layer (DIAM §4.2), reused faithfully.

An MGD layer refines node representations by propagating not just neighbor
representations but the *discrepancy* between a node and its neighbors, so that
closely connected illicit and licit accounts can still be pushed apart. For
layer ``l`` with input ``h`` (Eq. 4-5 of DIAM):

    z_v          = W2 h_v + b2
    r_in(v)      = sum_{u : u->v}  W3_in ( z_u || (z_v - z_u) )
    r_out(v)     = sum_{u : v->u}  W3_out( z_u || (z_v - z_u) )
    a_{v,1..3}   = softmax( q^T tanh(W_a [z_v, r_in(v), r_out(v)]) )
    h'_v         = a_{v,1} z_v + a_{v,2} r_in(v) + a_{v,3} r_out(v)

The sums run over the in/out neighbor **multisets**, so parallel edges are
counted with multiplicity -- the "multi" in multigraph. ``W3`` is shared across
the two directions by default, matching DIAM's single-symbol notation; set
``shared_message=False`` for independent in/out message maps.

No state is cached inside this layer: every forward recomputes from ``h`` and
``edge_index``. Incremental reuse is the maintainer's responsibility (C3) and is
done without breaking the autograd graph -- unlike the detached per-position
cache that silently stopped gradients in the prior prototype.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["MGDLayer"]


class MGDLayer(nn.Module):
    """One discrepancy-aware message-passing layer: ``[N, in_dim] -> [N, c]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        shared_message: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1:
            raise ValueError("in_dim and hidden_dim must be >= 1")
        self.hidden_dim = hidden_dim
        self.lin_z = nn.Linear(in_dim, hidden_dim)  # W2, b2
        self.msg_in = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)  # W3 (in)
        self.msg_out = (
            self.msg_in
            if shared_message
            else nn.Linear(2 * hidden_dim, hidden_dim, bias=False)  # W3 (out)
        )
        self.att_w = nn.Linear(hidden_dim, hidden_dim)
        self.att_q = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """Refine representations.

        Args:
            h: ``[N, in_dim]`` node representations (``2c`` at layer 1, else ``c``).
            edge_index: ``[2, E]`` int64; row 0 is the source ``u``, row 1 the
                destination ``v`` of each directed edge ``u -> v`` (with
                multiplicity for parallel edges).

        Returns:
            ``[N, c]`` refined representations.
        """
        n = h.size(0)
        z = self.lin_z(h)  # [N, c]
        src, dst = edge_index[0], edge_index[1]

        # Incoming messages aggregate at the destination v over edges u -> v.
        m_in = self.msg_in(torch.cat([z[src], z[dst] - z[src]], dim=-1))  # [E, c]
        r_in = z.new_zeros((n, self.hidden_dim)).index_add(0, dst, m_in)  # [N, c]

        # Outgoing messages aggregate at the source v over edges v -> u.
        m_out = self.msg_out(torch.cat([z[dst], z[src] - z[dst]], dim=-1))  # [E, c]
        r_out = z.new_zeros((n, self.hidden_dim)).index_add(0, src, m_out)  # [N, c]

        # Attention over the three aspects: self, incoming, outgoing.
        stacked = torch.stack([z, r_in, r_out], dim=1)  # [N, 3, c]
        scores = self.att_q(torch.tanh(self.att_w(stacked)))  # [N, 3, 1]
        alpha = torch.softmax(scores, dim=1)  # [N, 3, 1]
        h_new = (alpha * stacked).sum(dim=1)  # [N, c]
        return self.dropout(h_new)
