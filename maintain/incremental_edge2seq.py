"""Incremental Edge2Seq maintenance (C3, Theorem 1 of docs/proofs.md).

Maintains the growing-window Edge2Seq code of every account in O(c) per edge,
exactly matching the from-scratch oracle. Per account and direction it keeps
only the last GRU state ``h`` and the running time max-pool ``m`` (plus a count);
on a new edge it does one linear map, one GRU cell step, and one element-wise
max. The edge ``u -> v`` updates exactly ``(u, out)`` and ``(v, in)``.

This is the correctness-first reference maintainer: it reuses the model's own
``Edge2Seq`` weights via single-step GRU calls, so the maintained state is bit-
for-bit the recurrence the oracle computes. A throughput-optimized version
(batched cell steps) can replace the per-edge Python loop later without changing
the semantics.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from streamdiam.data.schema import Direction, TransactionBatch
from streamdiam.model.edge2seq import Edge2Seq

__all__ = ["IncrementalEdge2Seq"]

_IN, _OUT = int(Direction.IN), int(Direction.OUT)


class _DirState:
    """Per-(account, direction) maintained state: last GRU state, max-pool, count."""

    __slots__ = ("h", "m", "count")

    def __init__(self) -> None:
        self.h: Tensor | None = None  # [c] last GRU hidden state
        self.m: Tensor | None = None  # [c] running element-wise max of hidden states
        self.count: int = 0


class IncrementalEdge2Seq:
    """Maintain every account's Edge2Seq code under a stream of edge insertions."""

    def __init__(self, edge2seq: Edge2Seq, device: torch.device | str = "cpu"):
        self.e2s = edge2seq.eval()
        self.device = torch.device(device)
        self.c = edge2seq.hidden_dim
        self.d = edge2seq.in_dim
        # account -> [ _DirState(IN), _DirState(OUT) ]
        self._state: dict[int, list[_DirState]] = {}
        # Fixed self-loop codes for empty directions (encode a single zero row).
        with torch.no_grad():
            self._empty_in = self._encode_zero_row(_IN)
            self._empty_out = self._encode_zero_row(_OUT)
            self._empty_code = torch.cat([self._empty_in, self._empty_out])

    # -- internals ---------------------------------------------------------- #

    def _modules(self, direction: int) -> tuple[torch.nn.Linear, torch.nn.GRU]:
        if direction == _IN:
            return self.e2s.lin_in, self.e2s.gru_in
        return self.e2s.lin_out, self.e2s.gru_out

    def _encode_zero_row(self, direction: int) -> Tensor:
        """Edge2Seq code of a single zero edge -- the self-loop value ``s_dir``."""
        lin, gru = self._modules(direction)
        z = lin(torch.zeros(1, self.d, device=self.device))  # [1, c]
        out, _ = gru(z.view(1, 1, self.c))  # h_0 defaults to zeros
        return out.reshape(self.c).clone()

    def _gru_step(self, direction: int, x: Tensor, h_prev: Tensor | None) -> Tensor:
        """One GRU cell step using the model's GRU weights; returns ``h`` as ``[c]``."""
        lin, gru = self._modules(direction)
        z = lin(x.view(1, self.d)).view(1, 1, self.c)  # [1,1,c]
        h0 = None if h_prev is None else h_prev.view(1, 1, self.c)
        out, _ = gru(z, h0)
        return out.reshape(self.c)

    def _step(self, account: int, direction: int, x: Tensor) -> None:
        st = self._state[account][direction]
        h_new = self._gru_step(direction, x, st.h)
        st.m = h_new.clone() if st.count == 0 else torch.maximum(st.m, h_new)  # type: ignore[arg-type]
        st.h = h_new
        st.count += 1

    def _ensure(self, account: int) -> None:
        if account not in self._state:
            self._state[account] = [_DirState(), _DirState()]

    # -- public API --------------------------------------------------------- #

    @torch.no_grad()
    def on_edge(self, src: int, dst: int, attr: Tensor) -> None:
        """Apply one edge ``src -> dst`` with feature row ``attr`` of shape ``[d]``."""
        self._ensure(src)
        self._ensure(dst)
        self._step(src, _OUT, attr)
        self._step(dst, _IN, attr)

    @torch.no_grad()
    def on_batch(self, batch: TransactionBatch) -> None:
        """Apply a delta, processing rows in their (canonical) order."""
        if batch.num_features != self.d:
            raise ValueError(f"batch F={batch.num_features} != model d={self.d}")
        if len(batch) == 0:
            return
        attrs = torch.from_numpy(np.array(batch.attr, dtype=np.float32)).to(self.device)
        src = batch.src.tolist()
        dst = batch.dst.tolist()
        for i in range(len(batch)):
            self.on_edge(src[i], dst[i], attrs[i])

    @torch.no_grad()
    def code(self, account: int) -> Tensor:
        """Current Edge2Seq code ``[2c]`` for one account."""
        st = self._state.get(account)
        if st is None:
            return self._empty_code.clone()
        m_in, m_out = st[_IN].m, st[_OUT].m
        code_in = m_in if m_in is not None else self._empty_in
        code_out = m_out if m_out is not None else self._empty_out
        return torch.cat([code_in, code_out])

    @torch.no_grad()
    def codes(self, universe_size: int) -> Tensor:
        """Edge2Seq codes ``[N, 2c]`` for all accounts in ``[0, universe_size)``."""
        out = self._empty_code.unsqueeze(0).repeat(universe_size, 1).clone()
        for account, st in self._state.items():
            if account >= universe_size:
                continue
            m_in, m_out = st[_IN].m, st[_OUT].m
            if m_in is not None:
                out[account, : self.c] = m_in
            if m_out is not None:
                out[account, self.c :] = m_out
        return out
