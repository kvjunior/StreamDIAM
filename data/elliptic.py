"""Loader for the Elliptic Bitcoin data set into the canonical multigraph.

Honest modeling note (read before using this for results)
----------------------------------------------------------
Elliptic is *natively a transaction graph*: its nodes are Bitcoin
transactions (each with a time step in ``1..49`` and 165 features) and its
edges are directed payment flows between transactions; the labels
(illicit / licit / unknown) are on nodes. It is a standard temporal
node-classification benchmark.

Our system is defined over an *account* multigraph with attributed edges. We
therefore adopt one explicit, documented construction — not a silent
reinterpretation:

* each Elliptic transaction-node becomes an **account** (the entity we
  classify), on a contiguous id space;
* each directed flow ``u -> v`` becomes a **transaction edge** whose
  **timestamp is the time step of the source** ``u``. This is faithful to
  Bitcoin semantics: a transaction ``u`` is a single event that creates all of
  its outputs at one time, so ``u``'s outgoing edges legitimately share
  ``time(u)``, while ``v``'s incoming edges spread across the times of the
  sources that fund it;
* each edge's **feature vector is the 165 features of the source** ``u`` — the
  counterparty signal the destination observes. Elliptic supplies no native
  edge features, so some construction is unavoidable; this one is stated here
  and nowhere assumed implicitly.

This loader lets the whole pipeline run and be tested on *real* data. The
native DIAM Ethereum / Bitcoin account multigraphs (true per-transaction
timestamps and amounts) are the target for the paper's headline results and
are obtained separately; they slot into the same :class:`RawTemporalGraph` API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from streamdiam.data.raw import RawTemporalGraph, remap_accounts
from streamdiam.data.schema import (
    ATTR_DTYPE,
    LABEL_DTYPE,
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    TIME_DTYPE,
    TransactionBatch,
)

__all__ = ["load_elliptic"]

_FEATURES_CSV = "elliptic_txs_features.csv"
_CLASSES_CSV = "elliptic_txs_classes.csv"
_EDGELIST_CSV = "elliptic_txs_edgelist.csv"

# Elliptic class strings -> our label encoding.
_CLASS_TO_LABEL = {"1": LABEL_ILLICIT, "2": LABEL_LICIT, "unknown": LABEL_UNKNOWN}


def load_elliptic(root: str | Path) -> RawTemporalGraph:
    """Parse the three Elliptic CSVs under ``root`` into a RawTemporalGraph.

    Args:
        root: Directory containing ``elliptic_txs_features.csv``,
            ``elliptic_txs_classes.csv`` and ``elliptic_txs_edgelist.csv``.

    Returns:
        The canonical temporal multigraph (N = 203,769 accounts, F = 165).
    """
    root = Path(root)
    feats_df = pd.read_csv(root / _FEATURES_CSV, header=None)
    # Column 0 is the node id, column 1 the time step, the rest are features.
    node_ids = feats_df.iloc[:, 0].to_numpy(dtype=np.int64)
    node_time = feats_df.iloc[:, 1].to_numpy(dtype=TIME_DTYPE)
    node_feat = np.ascontiguousarray(feats_df.iloc[:, 2:].to_numpy(dtype=ATTR_DTYPE))
    num_features = node_feat.shape[1]

    # Contiguous account ids. Node ids are unique (one feature row per node), so
    # the universe of accounts is exactly the feature-file ids.
    codes, uniques = remap_accounts(node_ids)
    n_accounts = int(uniques.shape[0])
    # Reindex per-node arrays into contiguous-id order.
    time_by_id = np.empty(n_accounts, dtype=TIME_DTYPE)
    feat_by_id = np.empty((n_accounts, num_features), dtype=ATTR_DTYPE)
    time_by_id[codes] = node_time
    feat_by_id[codes] = node_feat

    # Labels (per account). Default unknown; the classes file may omit nodes.
    account_label = np.full(n_accounts, LABEL_UNKNOWN, dtype=LABEL_DTYPE)
    classes_df = pd.read_csv(root / _CLASSES_CSV)  # columns: txId, class
    cls_ids = classes_df["txId"].to_numpy(dtype=np.int64)
    cls_codes = np.searchsorted(uniques, cls_ids)
    cls_found = (cls_codes < n_accounts) & (uniques[np.clip(cls_codes, 0, n_accounts - 1)] == cls_ids)
    for txid, code, found, cls in zip(
        cls_ids, cls_codes, cls_found, classes_df["class"].astype(str)
    ):
        if found:
            account_label[code] = _CLASS_TO_LABEL[cls]

    # Edges: map endpoint ids to codes via binary search on the sorted uniques.
    edges_df = pd.read_csv(root / _EDGELIST_CSV)  # columns: txId1, txId2
    e1 = edges_df.iloc[:, 0].to_numpy(dtype=np.int64)
    e2 = edges_df.iloc[:, 1].to_numpy(dtype=np.int64)
    u = np.searchsorted(uniques, e1)
    v = np.searchsorted(uniques, e2)
    u_ok = (u < n_accounts) & (uniques[np.clip(u, 0, n_accounts - 1)] == e1)
    v_ok = (v < n_accounts) & (uniques[np.clip(v, 0, n_accounts - 1)] == e2)
    keep = u_ok & v_ok
    dropped = int((~keep).sum())  # edges referencing an unknown node (expected: 0)
    u, v = u[keep].astype(np.int64), v[keep].astype(np.int64)

    # Edge timestamp = time of the source; edge features = source node features.
    t_edge = time_by_id[u]
    # Canonical arrival order: by (timestamp, src, dst), then assign dense eids.
    order = np.lexsort((v, u, t_edge))
    u, v, t_edge = u[order], v[order], t_edge[order]
    attr = feat_by_id[u]
    edge_id = np.arange(u.shape[0], dtype=np.int64)

    edges = TransactionBatch.from_arrays(
        edge_id=edge_id, src=u, dst=v, timestamp=t_edge, attr=attr
    )

    return RawTemporalGraph(
        name="elliptic",
        edges=edges,
        account_label=account_label,
        num_accounts=n_accounts,
        num_features=num_features,
        extra={
            "n_time_steps": int(node_time.max()),
            "dropped_edges": dropped,
            "node_time": time_by_id,  # provenance: account id -> Elliptic time step
            "original_ids": uniques,  # account id -> original Elliptic txId
            "edge_timestamp_rule": "source_node_time_step",
            "edge_feature_rule": "source_node_165_features",
        },
    )
