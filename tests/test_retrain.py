"""Tests for the retrain package (C4 part 2).

Recency weighting is tested directly. The trainer and hot-swap are tested on a
**real** Elliptic subgraph (remapped to a compact account universe so it carries
genuine illicit labels): the trainer must actually reduce loss and raise average
precision on the real labels -- the blueprint's "reproduce DIAM / sanity anchor"
step -- and the full train -> swap -> rebuild loop must improve the served view.
"""

from __future__ import annotations

import copy
import functools
import os
import sys

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamdiam.data import TransactionBatch, load_elliptic  # noqa: E402
from streamdiam.maintain import FromScratchMaintainer  # noqa: E402
from streamdiam.model import DIAM  # noqa: E402
from streamdiam.retrain import (  # noqa: E402
    BackgroundTrainer,
    HotSwapRegistry,
    RecencyWeighting,
    TrainConfig,
    recency_weights,
)
from streamdiam.store import ReferencePyStore  # noqa: E402

ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")


# --------------------------------------------------------------------------- #
# Recency weighting
# --------------------------------------------------------------------------- #
def test_recency_decay_and_uniform() -> None:
    times = np.array([0, 5, 10], dtype=np.int64)
    w = recency_weights(times, ref_time=10, lam=0.1)
    assert w[2] > w[1] > w[0]  # more recent -> larger weight
    assert np.allclose(recency_weights(times, 10, lam=0.0), 1.0)  # uniform


def test_recency_from_half_life() -> None:
    sched = RecencyWeighting.from_half_life(4.0)
    w = sched.weights(np.array([6.0]), ref_time=10.0)  # age = 4 = one half-life
    assert abs(float(w[0]) - 0.5) < 1e-9
    assert abs(sched.half_life - 4.0) < 1e-9
    assert RecencyWeighting.uniform().half_life == float("inf")


def test_recency_future_is_clamped() -> None:
    w = recency_weights(np.array([20.0]), ref_time=10.0, lam=0.5)  # future sample
    assert float(w[0]) == 1.0  # age clamped at 0, never look-ahead


def test_recency_normalize_mean_one() -> None:
    w = recency_weights(np.arange(100), ref_time=99, lam=0.05, normalize=True)
    assert abs(w.mean() - 1.0) < 1e-9


def test_recency_validation() -> None:
    with pytest.raises(ValueError):
        recency_weights(np.array([1]), 1, lam=-0.1)
    with pytest.raises(ValueError):
        RecencyWeighting.from_half_life(0.0)
    with pytest.raises(ValueError):
        RecencyWeighting(lam=-1.0)


# --------------------------------------------------------------------------- #
# Real Elliptic subgraph fixture (remapped to a compact labelled universe)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _load_graph():
    # Parse the full Elliptic CSVs once for the whole module (expensive).
    return load_elliptic(ELLIPTIC_ROOT)


def _elliptic_subgraph(n_edges: int = 2500):
    graph = _load_graph()
    e = graph.edges
    src = np.asarray(e.src)[:n_edges]
    dst = np.asarray(e.dst)[:n_edges]
    t = np.asarray(e.timestamp)[:n_edges]
    attr = np.asarray(e.attr)[:n_edges]
    inv = np.unique(np.concatenate([src, dst]))
    remap = -np.ones(graph.num_accounts, dtype=np.int64)
    remap[inv] = np.arange(inv.size)
    batch = TransactionBatch.from_arrays(
        np.arange(n_edges), remap[src], remap[dst], t, attr
    )
    labels = graph.account_label[inv].copy()
    store = ReferencePyStore(graph.num_features)
    store.insert(batch)
    return store, labels, int(inv.size), int(t.max()), graph.num_features


def _ap(model: DIAM, store, labels, universe, nfeat) -> float:
    view = FromScratchMaintainer(model, t_max=10_000, chunk_size=10_000).compute_view(
        store, universe
    )
    scores = view.scores.numpy()
    mask = labels >= 0
    return float(average_precision_score((labels[mask] == 1).astype(int), scores[mask]))


# --------------------------------------------------------------------------- #
# Background trainer
# --------------------------------------------------------------------------- #
@needs_elliptic
def test_trainer_reduces_loss() -> None:
    store, labels, universe, ref, nfeat = _elliptic_subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=2)
    trainer = BackgroundTrainer()
    ckpt = trainer.train(model, store, labels, universe, ref, TrainConfig(epochs=30))
    assert ckpt.final_loss < ckpt.loss_history[0]  # learning happened
    assert ckpt.n_pos > 0 and ckpt.n_train > ckpt.n_pos


@needs_elliptic
def test_trainer_does_not_mutate_live_model() -> None:
    store, labels, universe, ref, nfeat = _elliptic_subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=2)
    before = copy.deepcopy(model.state_dict())
    BackgroundTrainer().train(model, store, labels, universe, ref, TrainConfig(epochs=10))
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])  # live model untouched during fine-tune


@needs_elliptic
def test_trainer_learns_on_real_elliptic() -> None:
    """Training-set average precision rises well above the random baseline."""
    store, labels, universe, ref, nfeat = _elliptic_subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=2)
    ap_before = _ap(model, store, labels, universe, nfeat)
    ckpt = trainer_ckpt = BackgroundTrainer().train(
        model, store, labels, universe, ref, TrainConfig(epochs=50)
    )
    trained = copy.deepcopy(model)
    trained.load_state_dict(ckpt.state_dict)
    ap_after = _ap(trained, store, labels, universe, nfeat)
    assert ap_after > ap_before
    assert ap_after > 0.05  # a real, non-trivial separation of the 10 positives
    assert trainer_ckpt.version == 1


@needs_elliptic
def test_trainer_requires_labels() -> None:
    store, labels, universe, ref, nfeat = _elliptic_subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1)
    mask = np.zeros(universe, dtype=bool)  # no trainable accounts
    with pytest.raises(ValueError):
        BackgroundTrainer().train(
            model, store, labels, universe, ref, train_mask=mask
        )


@needs_elliptic
def test_trainer_recency_path_runs() -> None:
    store, labels, universe, ref, nfeat = _elliptic_subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1)
    cfg = TrainConfig(epochs=15, recency=RecencyWeighting.from_half_life(10.0))
    ckpt = BackgroundTrainer().train(model, store, labels, universe, ref, cfg)
    assert ckpt.final_loss < ckpt.loss_history[0]  # recency-weighted loss trains


# --------------------------------------------------------------------------- #
# Hot swap
# --------------------------------------------------------------------------- #
@needs_elliptic
def test_hotswap_installs_new_version_and_changes_output() -> None:
    store, labels, universe, ref, nfeat = _elliptic_subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=2)
    reg = HotSwapRegistry(model)
    assert reg.version == 0
    before = _ap(reg.model, store, labels, universe, nfeat)

    ckpt = BackgroundTrainer().train(model, store, labels, universe, ref, TrainConfig(epochs=50))
    event = reg.swap(ckpt)
    assert event.old_version == 0 and event.new_version == 1 and reg.version == 1
    after = _ap(reg.model, store, labels, universe, nfeat)
    assert after > before  # the served model now reflects the fine-tune


@needs_elliptic
def test_hotswap_rollback() -> None:
    store, labels, universe, ref, nfeat = _elliptic_subgraph()
    model = DIAM(in_dim=nfeat, hidden_dim=16, num_layers=1)
    reg = HotSwapRegistry(model)
    ckpt = BackgroundTrainer().train(model, store, labels, universe, ref, TrainConfig(epochs=5))
    reg.swap(ckpt)
    assert reg.version == 1
    reg.rollback()
    assert reg.version == 0
    with pytest.raises(RuntimeError):
        reg.rollback()  # only one level of undo


def test_hotswap_requires_previous_for_rollback() -> None:
    model = DIAM(in_dim=4, hidden_dim=8, num_layers=1)
    reg = HotSwapRegistry(model)
    with pytest.raises(RuntimeError):
        reg.rollback()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
