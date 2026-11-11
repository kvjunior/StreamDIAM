"""The assembled DIAM model: Edge2Seq -> L x MGD -> 2-layer MLP classifier.

This is a faithful reimplementation of DIAM (CIKM'24), reused as StreamDIAM's
detector rather than claimed as a contribution. Defaults follow the paper:
hidden ``c = 128``, ``L = 2`` MGD layers, dropout ``0.2``. The classifier emits
a single logit per node for a binary (illicit vs. licit) decision trained with
``BCEWithLogitsLoss``; the sigmoid lives in the loss, so ``forward`` returns raw
logits.

The inference view the maintainer (C3) incrementally maintains is exactly the
set of tensors this module computes: the Edge2Seq codes, each MGD layer's
output, and the final logits. ``forward`` here is the from-scratch oracle those
increments must match.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from streamdiam.model.edge2seq import Edge2Seq
from streamdiam.model.mgd import MGDLayer

__all__ = ["DIAM"]


class DIAM(nn.Module):
    """Edge2Seq encoder, a stack of MGD layers, and an MLP risk head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        shared_message: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.edge2seq = Edge2Seq(in_dim, hidden_dim)
        # Layer 1 consumes the 2c-dim Edge2Seq code; later layers consume c.
        layers = [MGDLayer(2 * hidden_dim, hidden_dim, shared_message, dropout)]
        layers += [
            MGDLayer(hidden_dim, hidden_dim, shared_message, dropout)
            for _ in range(num_layers - 1)
        ]
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        in_seq: Tensor,
        in_len: Tensor,
        out_seq: Tensor,
        out_len: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Run Edge2Seq + all MGD layers, returning node representations ``[N, c]``."""
        h = self.edge2seq(in_seq, in_len, out_seq, out_len)  # [N, 2c]
        for layer in self.layers:
            h = layer(h, edge_index)  # [N, c]
        return h

    def forward(
        self,
        in_seq: Tensor,
        in_len: Tensor,
        out_seq: Tensor,
        out_len: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Return per-node risk **logits** ``[N]`` (apply sigmoid for probability)."""
        h = self.encode(in_seq, in_len, out_seq, out_len, edge_index)
        return self.classifier(h).squeeze(-1)  # [N]

    @torch.no_grad()
    def predict_proba(
        self,
        in_seq: Tensor,
        in_len: Tensor,
        out_seq: Tensor,
        out_len: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Return per-node illicit probabilities ``[N]`` in ``[0, 1]``."""
        self.eval()
        logits = self.forward(in_seq, in_len, out_seq, out_len, edge_index)
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
