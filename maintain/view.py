"""The inference view Phi(G): everything the model computes over the graph.

``docs/problem_formalization.md`` §4 defines the inference view as the per-account
Edge2Seq codes, the MGD layer outputs, and the risk scores. The from-scratch
oracle (``from_scratch.py``) materializes it in full; the incremental maintainer
(C3) maintains a copy of it and must stay within the epsilon-correctness bound
of this object after every delta.

All tensors are indexed by account id on the fixed universe ``[0, universe_size)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

__all__ = ["InferenceView"]


@dataclass(frozen=True)
class InferenceView:
    """Materialized model state over the current graph.

    Attributes:
        codes: ``[N, 2c]`` Edge2Seq node codes (the first MGD layer's input).
        layer_hidden: list of ``[N, c]`` outputs, one per MGD layer, in order;
            ``layer_hidden[-1]`` is the final representation fed to the head.
        logits: ``[N]`` pre-sigmoid risk logits.
        scores: ``[N]`` risk probabilities ``sigmoid(logits)`` in ``[0, 1]``.
        universe_size: ``N``, the number of accounts the view covers.
    """

    codes: Tensor
    layer_hidden: list[Tensor]
    logits: Tensor
    scores: Tensor
    universe_size: int

    @property
    def hidden(self) -> Tensor:
        """Final MGD representation ``[N, c]``."""
        return self.layer_hidden[-1]

    def __repr__(self) -> str:
        c = self.hidden.shape[1]
        return (
            f"InferenceView(N={self.universe_size}, layers={len(self.layer_hidden)}, "
            f"c={c})"
        )
