"""Adapter: turn store reads into the tensors the model consumes.

Two kinds of input are prepared from the temporal multigraph store:

* **Edge2Seq sequence inputs** for a *chunk* of accounts -- each account's most
  recent ``T_max`` incoming and outgoing edge-attribute rows, padded to a dense
  ``[B, T_max, d]`` tensor with per-account lengths. An account with no edges in
  a direction is given a single zero row (a self-loop), so every length is >= 1
  as Edge2Seq requires.
* the **edge index** ``[2, E]`` of the whole graph, which the MGD layers consume.

Chunking keeps the padded-sequence memory bounded (the full universe times
``T_max`` times ``d`` would be enormous), while the per-account encodings are
independent, so the chunked result is identical to encoding everything at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from streamdiam.data.schema import Direction
from streamdiam.store.interface import TemporalMultigraphStore

__all__ = ["SequenceInputs", "build_sequence_inputs", "edge_index_from_store"]


@dataclass(frozen=True)
class SequenceInputs:
    """Padded Edge2Seq inputs for one chunk of accounts."""

    accounts: Tensor  # [B] int64 account ids, in request order
    in_seq: Tensor    # [B, T, d] float32
    in_len: Tensor    # [B] int64, each >= 1
    out_seq: Tensor   # [B, T, d] float32
    out_len: Tensor   # [B] int64, each >= 1


def build_sequence_inputs(
    store: TemporalMultigraphStore,
    accounts: np.ndarray,
    t_max: int,
    device: torch.device | str = "cpu",
) -> SequenceInputs:
    """Build padded in/out sequence tensors for ``accounts``.

    Args:
        store: the temporal multigraph store to read from.
        accounts: 1-D array of account ids to encode.
        t_max: maximum (most-recent) sequence length kept per direction.
        device: destination device for the returned tensors.
    """
    if t_max < 1:
        raise ValueError(f"t_max must be >= 1, got {t_max}")
    accts = np.ascontiguousarray(accounts, dtype=np.int64)
    b = int(accts.shape[0])
    d = store.num_features

    # Gather the (capped) tails first so we can size the padded tensor to the
    # longest sequence actually present -- never the full t_max, which may be
    # huge. Padding width does not affect the result (padded steps are masked),
    # so this is a pure memory optimization.
    in_tails = [store.tail(a, Direction.IN, t_max) for a in accts.tolist()]
    out_tails = [store.tail(a, Direction.OUT, t_max) for a in accts.tolist()]
    width = 1
    for tb in in_tails:
        width = max(width, len(tb))
    for tb in out_tails:
        width = max(width, len(tb))

    # Stage in writable numpy, then hand to torch in one shot (no copy warnings).
    in_np = np.zeros((b, width, d), dtype=np.float32)
    out_np = np.zeros((b, width, d), dtype=np.float32)
    in_len = np.ones(b, dtype=np.int64)   # default length 1 == self-loop (zeros)
    out_len = np.ones(b, dtype=np.int64)

    for i in range(b):
        inc = in_tails[i]
        k = len(inc)
        if k > 0:
            in_np[i, :k] = inc.attr
            in_len[i] = k
        out = out_tails[i]
        k = len(out)
        if k > 0:
            out_np[i, :k] = out.attr
            out_len[i] = k

    dev = torch.device(device)
    return SequenceInputs(
        accounts=torch.from_numpy(accts).to(dev),
        in_seq=torch.from_numpy(in_np).to(dev),
        in_len=torch.from_numpy(in_len).to(dev),
        out_seq=torch.from_numpy(out_np).to(dev),
        out_len=torch.from_numpy(out_len).to(dev),
    )


def edge_index_from_store(
    store: TemporalMultigraphStore, device: torch.device | str = "cpu"
) -> Tensor:
    """Return the whole graph's ``edge_index`` ``[2, E]`` (row 0 src, row 1 dst)."""
    edges = store.all_edges()
    # Copy out of the store's frozen (read-only) arrays so the tensor owns
    # writable memory and never aliases immutable store state.
    src = torch.from_numpy(np.array(edges.src, dtype=np.int64))
    dst = torch.from_numpy(np.array(edges.dst, dtype=np.int64))
    return torch.stack([src, dst], dim=0).to(torch.device(device))
