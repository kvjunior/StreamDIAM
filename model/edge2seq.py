"""Edge2Seq encoder (DIAM §4.1), reused faithfully.

For a node ``v`` the store yields two chronologically ordered edge-attribute
sequences -- incoming ``X_in(v)`` and outgoing ``X_out(v)`` (§1.2 of
``docs/problem_formalization.md``). Edge2Seq encodes each sequence with a
per-edge linear map followed by a GRU, max-pools the GRU outputs over time, and
concatenates the two directional codes into the node representation

    h_v = [ maxpool_t GRU_in(Linear_in(X_in(v))) ;
            maxpool_t GRU_out(Linear_out(X_out(v))) ]  in R^{2c}.

The max-pool over time (rather than taking the last state) matches DIAM's
stated computation and its complexity analysis. Incoming and outgoing use
separate linear maps and GRUs because a node behaves differently as receiver
vs. sender. Sequences are capped at ``T_max`` most-recent edges by the caller;
nodes with no edges in a direction get a self-loop so every length is >= 1.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["Edge2Seq"]


class Edge2Seq(nn.Module):
    """Encode per-account in/out edge-attribute sequences into ``h_v in R^{2c}``."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1:
            raise ValueError("in_dim and hidden_dim must be >= 1")
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.lin_out = nn.Linear(in_dim, hidden_dim)
        self.gru_in = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru_out = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    @staticmethod
    def _encode(
        seq: Tensor, lengths: Tensor, lin: nn.Linear, gru: nn.GRU
    ) -> Tensor:
        """Encode one padded sequence batch into a max-pooled code ``[N, c]``.

        Args:
            seq: ``[N, T, d]`` padded edge attributes.
            lengths: ``[N]`` valid lengths, each in ``[1, T]``.
            lin: per-edge linear map ``d -> c``.
            gru: GRU with input/hidden ``c``.
        """
        n, t, _ = seq.shape
        z = lin(seq)  # [N, T, c]
        # Pack so the GRU ignores padding; lengths must be on CPU and >= 1.
        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=t
        )  # [N, T, c], padded steps are zero
        # Max-pool over valid time steps only: mask padded steps to -inf.
        valid = torch.arange(t, device=seq.device)[None, :] < lengths[:, None]  # [N,T]
        out = out.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled, _ = out.max(dim=1)  # [N, c]
        return pooled

    def forward(
        self, in_seq: Tensor, in_len: Tensor, out_seq: Tensor, out_len: Tensor
    ) -> Tensor:
        """Return node representations ``h_v`` of shape ``[N, 2c]``.

        Args:
            in_seq: ``[N, T, d]`` incoming edge attributes (padded).
            in_len: ``[N]`` incoming lengths, each >= 1.
            out_seq: ``[N, T, d]`` outgoing edge attributes (padded).
            out_len: ``[N]`` outgoing lengths, each >= 1.
        """
        if int(in_len.min()) < 1 or int(out_len.min()) < 1:
            raise ValueError(
                "every sequence length must be >= 1 (add a self-loop for empty nodes)"
            )
        h_in = self._encode(in_seq, in_len, self.lin_in, self.gru_in)
        h_out = self._encode(out_seq, out_len, self.lin_out, self.gru_out)
        return torch.cat([h_in, h_out], dim=-1)  # [N, 2c]
