"""Affected-set computation for incremental MGD (C3).

When a node's representation changes at MGD layer ``l-1``, the nodes whose
layer-``l`` representation can change are that node together with all of its
graph neighbors: a node ``v``'s layer-``l`` output depends on ``z_v`` and on the
messages from its in/out neighbors, so a change to ``z_w`` can only reach ``w``
itself and the nodes adjacent to ``w``. One layer therefore expands the changed
set by one hop; ``L`` layers expand it by ``L`` hops (Theorem 2, docs/proofs.md).

``neighbor_closure`` computes ``S ∪ N(S)`` -- the one-hop expansion used once per
layer. Neighbors are taken in *both* directions because ``z_w`` enters both the
messages ``w`` sends and those it receives.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from streamdiam.data.schema import Direction
from streamdiam.store.interface import TemporalMultigraphStore

__all__ = ["neighbor_closure"]


def neighbor_closure(
    store: TemporalMultigraphStore, nodes: npt.NDArray[np.integer]
) -> npt.NDArray[np.int64]:
    """Return the sorted unique set ``nodes ∪ N(nodes)`` (both-direction neighbors)."""
    seed = np.unique(np.asarray(nodes, dtype=np.int64))
    acc: set[int] = set(seed.tolist())
    for v in seed.tolist():
        acc.update(store.neighbors(v, Direction.IN).tolist())
        acc.update(store.neighbors(v, Direction.OUT).tolist())
    return np.array(sorted(acc), dtype=np.int64)
