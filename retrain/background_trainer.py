"""Decoupled background trainer (C4).

This is the off-critical-path half of the retraining story. When the drift
trigger fires, the trainer fine-tunes the model on a recency-weighted window of
history and returns a versioned checkpoint. Crucially it trains a **copy** of
the live model, never the live model itself, so the inference view (C3) keeps
serving current scores throughout training -- the decoupling the paper claims is
realised here by construction, not by assertion.

The objective is a weighted ``BCEWithLogits`` over **labelled** accounts only
(Elliptic is ~23% labelled; unlabelled accounts are masked out, not guessed),
combining two weightings:

* **class** -- ``pos_weight = n_neg / n_pos`` (auto) counters the 2-10% illicit
  imbalance (R4 O6);
* **recency** -- :mod:`streamdiam.retrain.recency_weight` down-weights stale
  accounts (R4 O7).

Training reuses the model's own forward pass over the store snapshot, so the
weights produced are exactly the weights the maintainer will then serve -- no
train/serve skew.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from streamdiam.data.schema import LABEL_ILLICIT, LABEL_LICIT
from streamdiam.maintain.inputs import build_sequence_inputs, edge_index_from_store
from streamdiam.model.diam import DIAM
from streamdiam.retrain.recency_weight import RecencyWeighting
from streamdiam.store.interface import TemporalMultigraphStore
from streamdiam.utils.seeding import seed_everything

__all__ = ["TrainConfig", "ModelCheckpoint", "BackgroundTrainer"]


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one fine-tune, every default documented.

    Attributes:
        lr: Adam learning rate.
        weight_decay: Adam L2 regularisation.
        epochs: full-batch passes over the window.
        grad_clip: max global grad norm (``None`` disables).
        pos_weight: positive-class weight; ``None`` auto-computes ``n_neg/n_pos``.
        recency: recency schedule; ``None`` means uniform weighting.
        seed: RNG seed for reproducible fine-tunes.
    """

    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 30
    grad_clip: float | None = 5.0
    pos_weight: float | None = None
    recency: RecencyWeighting | None = None
    seed: int = 0


@dataclass
class ModelCheckpoint:
    """A trained model version produced by one fine-tune.

    ``state_dict`` is a CPU copy of the trained (cloned) model's parameters, so
    it can be hot-swapped into the maintainer without disturbing the live model.
    """

    version: int
    state_dict: dict[str, torch.Tensor]
    ref_time: float
    n_train: int
    n_pos: int
    final_loss: float
    loss_history: list[float] = field(default_factory=list)


class BackgroundTrainer:
    """Fine-tune a copy of the model on a recency-weighted labelled window."""

    def __init__(
        self,
        t_max: int = 10_000,
        device: torch.device | str = "cpu",
    ) -> None:
        self.t_max = t_max
        self.device = torch.device(device)
        self._version = 0

    def _last_activity_times(
        self, store: TemporalMultigraphStore, n: int
    ) -> npt.NDArray[np.int64]:
        """Most recent activity time per account (``-1`` if the account is unseen)."""
        last = np.full(n, -1, dtype=np.int64)
        edges = store.all_edges()
        if len(edges) > 0:
            t = np.asarray(edges.timestamp, dtype=np.int64)
            np.maximum.at(last, np.asarray(edges.src, dtype=np.int64), t)
            np.maximum.at(last, np.asarray(edges.dst, dtype=np.int64), t)
        return last

    def train(
        self,
        model: DIAM,
        store: TemporalMultigraphStore,
        labels: npt.NDArray[np.integer],
        universe_size: int,
        ref_time: float,
        config: TrainConfig | None = None,
        train_mask: npt.NDArray[np.bool_] | None = None,
    ) -> ModelCheckpoint:
        """Fine-tune a clone of ``model`` and return its checkpoint.

        Args:
            model: the live model; it is **cloned**, not mutated.
            store: the training-window snapshot (adjacency + sequences).
            labels: ``[universe_size]`` in ``{-1 unknown, 0 licit, 1 illicit}``.
            universe_size: account-id space size ``N``.
            ref_time: the retrain "now" for recency weighting.
            config: hyperparameters (defaults if ``None``).
            train_mask: optional ``[N]`` bool restricting which accounts may
                train; combined with "is labelled". ``None`` => all labelled.
        """
        cfg = config or TrainConfig()
        seed_everything(cfg.seed)
        n = int(universe_size)

        # Train a copy: the live model keeps serving during the fine-tune.
        clone = copy.deepcopy(model).to(self.device)
        clone.train()

        # Full-universe forward inputs (same path the maintainer/oracle use).
        accts = np.arange(n, dtype=np.int64)
        si = build_sequence_inputs(store, accts, self.t_max, self.device)
        edge_index = edge_index_from_store(store, self.device)

        lab = np.asarray(labels)
        labelled = (lab == LABEL_ILLICIT) | (lab == LABEL_LICIT)
        if train_mask is not None:
            labelled = labelled & np.asarray(train_mask, dtype=bool)
        if not labelled.any():
            raise ValueError("no labelled accounts in the training window")

        target = torch.from_numpy((lab == LABEL_ILLICIT).astype(np.float32)).to(
            self.device
        )
        mask_f = torch.from_numpy(labelled.astype(np.float32)).to(self.device)

        # Recency weights over labelled accounts; uniform if no schedule.
        if cfg.recency is not None:
            times = self._last_activity_times(store, n)
            rec = cfg.recency.weights(times, ref_time, normalize=True)
            rec_t = torch.from_numpy(rec.astype(np.float32)).to(self.device)
        else:
            rec_t = torch.ones(n, device=self.device)
        sample_w = mask_f * rec_t

        n_pos = int((lab == LABEL_ILLICIT)[labelled].sum())
        n_neg = int((lab == LABEL_LICIT)[labelled].sum())
        if cfg.pos_weight is not None:
            pw = cfg.pos_weight
        else:
            pw = (n_neg / n_pos) if n_pos > 0 else 1.0
        pos_weight = torch.tensor(pw, device=self.device)

        opt = torch.optim.Adam(
            clone.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        denom = float(sample_w.sum().item())
        loss_history: list[float] = []
        for _ in range(cfg.epochs):
            opt.zero_grad()
            logits = clone(si.in_seq, si.in_len, si.out_seq, si.out_len, edge_index)
            per = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight, reduction="none"
            )
            loss = (per * sample_w).sum() / denom
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(clone.parameters(), cfg.grad_clip)
            opt.step()
            loss_history.append(float(loss.item()))

        self._version += 1
        clone.eval()
        state = {k: v.detach().cpu().clone() for k, v in clone.state_dict().items()}
        return ModelCheckpoint(
            version=self._version,
            state_dict=state,
            ref_time=ref_time,
            n_train=int(labelled.sum()),
            n_pos=n_pos,
            final_loss=loss_history[-1] if loss_history else float("nan"),
            loss_history=loss_history,
        )
