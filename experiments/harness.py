"""Shared experiment harness: load a dataset, run a strategy, save results.

Thin glue over the eval harness (C5). Every experiment driver (rq1, rq2, rq3)
builds its runs from these helpers so the dataset path and the result format are
identical across research questions -- one code path, no per-RQ divergence.

Results are written as JSONL under ``<out>/<rq>/<dataset>/<run_id>.jsonl`` via the
eval harness's own writer. Nothing here plots; figures read these files later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from streamdiam.data import TransactionBatch, load_elliptic
from streamdiam.eval import RunConfig, RunResult, run_streaming, write_run_jsonl
from streamdiam.model import DIAM

__all__ = ["Dataset", "load_dataset", "run_strategy", "save_result"]


@dataclass(frozen=True)
class Dataset:
    """A loaded, compact streaming dataset ready for the harness."""

    name: str
    edges: TransactionBatch
    labels: npt.NDArray[np.int8]
    universe_size: int
    num_features: int


def load_dataset(
    name: str = "elliptic",
    root: str | None = None,
    per_step: int | None = None,
) -> Dataset:
    """Load a dataset into a compact, remapped account universe.

    Args:
        name: currently ``"elliptic"``.
        root: dataset root (Elliptic CSV directory).
        per_step: if set, keep only the first ``per_step`` edges of each
            timestep -- a fast, timestep-spanning development subsample. ``None``
            loads the full graph (headline runs).
    """
    if name != "elliptic":
        raise ValueError(f"unknown dataset: {name!r}")
    if root is None:
        raise ValueError("elliptic requires root=<CSV dir>")
    graph = load_elliptic(root)
    e = graph.edges
    if per_step is None:
        return Dataset(name, e, graph.account_label, graph.num_accounts, graph.num_features)

    ts = np.asarray(e.timestamp)
    _, starts, counts = np.unique(ts, return_index=True, return_counts=True)
    sel = np.concatenate(
        [np.arange(s, s + min(per_step, c)) for s, c in zip(starts, counts)]
    )
    src = np.asarray(e.src)[sel]
    dst = np.asarray(e.dst)[sel]
    inv = np.unique(np.concatenate([src, dst]))
    remap = -np.ones(graph.num_accounts, dtype=np.int64)
    remap[inv] = np.arange(inv.size)
    batch = TransactionBatch.from_arrays(
        np.arange(sel.size), remap[src], remap[dst], ts[sel], np.asarray(e.attr)[sel]
    )
    labels = graph.account_label[inv].copy()
    return Dataset(name, batch, labels, int(inv.size), graph.num_features)


def run_strategy(
    dataset: Dataset,
    *,
    maintainer: str,
    retrain_policy: str = "never",
    hidden_dim: int = 32,
    num_layers: int = 2,
    warmup_fraction: float = 0.3,
    epochs: int = 20,
    seed: int = 0,
    config_overrides: dict | None = None,
) -> RunResult:
    """Build a fresh model and run one (maintainer, policy) through the harness."""
    from streamdiam.retrain import TrainConfig
    from streamdiam.utils.seeding import seed_everything

    seed_everything(seed)
    model = DIAM(in_dim=dataset.num_features, hidden_dim=hidden_dim, num_layers=num_layers)
    cfg = RunConfig(
        maintainer=maintainer,
        retrain_policy=retrain_policy,
        warmup_fraction=warmup_fraction,
        train_config=TrainConfig(epochs=epochs, seed=seed),
    )
    for k, v in (config_overrides or {}).items():
        setattr(cfg, k, v)
    enable_retrain = retrain_policy != "never"
    return run_streaming(
        dataset.edges, dataset.labels, dataset.universe_size, model, cfg,
        enable_retrain=enable_retrain,
    )


def save_result(
    result: RunResult, out: str | Path, rq: str, dataset: str, run_id: str
) -> Path:
    """Write a run to ``<out>/<rq>/<dataset>/<run_id>.jsonl`` and return the path."""
    path = Path(out) / rq / dataset / f"{run_id}.jsonl"
    return write_run_jsonl(result, path)
