"""Atomic model hot-swap (C4).

When the background trainer produces a new checkpoint, it must replace the model
the maintainer serves **without stalling the inference path**. The registry holds
a single current :class:`VersionedModel`; :meth:`swap` builds the new version off
to the side and then replaces the pointer in one assignment -- atomic under
CPython's single writer, so a reader (the maintainer) always sees a fully-formed
model, never a half-loaded one.

Separation of concerns: the registry swaps the *model*; it does not recompute the
inference view. After a swap the caller rebuilds the view from the new model
(``FromScratchMaintainer`` for an exact refresh, or the incremental maintainer
re-seeded). The swap emits a :class:`SwapEvent` so the harness logs exactly when
weights changed -- another line that feeds the RQ3 story.

One previous version is retained so a bad fine-tune can be rolled back.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from streamdiam.model.diam import DIAM
from streamdiam.retrain.background_trainer import ModelCheckpoint

__all__ = ["VersionedModel", "SwapEvent", "HotSwapRegistry"]


@dataclass(frozen=True)
class VersionedModel:
    """A model paired with its version number."""

    model: DIAM
    version: int


@dataclass(frozen=True)
class SwapEvent:
    """Record of one swap, for the experiment log."""

    old_version: int
    new_version: int
    ref_time: float


class HotSwapRegistry:
    """Holds the live model version and swaps new checkpoints in atomically."""

    def __init__(self, model: DIAM) -> None:
        self._current = VersionedModel(model=model, version=0)
        self._previous: VersionedModel | None = None

    @property
    def current(self) -> VersionedModel:
        return self._current

    @property
    def model(self) -> DIAM:
        """The live model the maintainer should serve."""
        return self._current.model

    @property
    def version(self) -> int:
        return self._current.version

    def swap(self, checkpoint: ModelCheckpoint) -> SwapEvent:
        """Install ``checkpoint`` as the new live version; return the event.

        The new model is built from a deep copy of the current one (same
        architecture) with the checkpoint's parameters loaded, then published in
        one pointer assignment. The checkpoint's own version number is used so
        the live version tracks the trainer's counter.
        """
        new_model = copy.deepcopy(self._current.model)
        new_model.load_state_dict(checkpoint.state_dict)
        new_model.eval()
        old = self._current
        new = VersionedModel(model=new_model, version=checkpoint.version)
        # Atomic publish: single reference rebind (CPython single writer).
        self._previous = old
        self._current = new
        return SwapEvent(
            old_version=old.version,
            new_version=new.version,
            ref_time=checkpoint.ref_time,
        )

    def rollback(self) -> SwapEvent:
        """Revert to the previously-live version (one level of undo)."""
        if self._previous is None:
            raise RuntimeError("no previous version to roll back to")
        old = self._current
        self._current = self._previous
        self._previous = None
        return SwapEvent(
            old_version=old.version,
            new_version=self._current.version,
            ref_time=float("nan"),
        )
