"""Recency weighting for retraining (C4).

When the background trainer fine-tunes on a window of history, not all of that
history is equally relevant: recent behaviour should count for more than stale
behaviour. We weight each training account by an exponential decay of its age,

    ``w_i = exp(-lambda * (t_ref - t_i))``,

where ``t_i`` is the account's most recent activity time and ``t_ref`` is the
retrain reference time ("now"). ``lambda = 0`` recovers uniform weighting (the
ablation baseline).

The decay is parameterised by a **half-life** -- the age at which weight falls to
one half, ``lambda = ln 2 / half_life`` -- because a half-life in the dataset's
own time units (Elliptic time steps, Unix seconds) is interpretable and directly
sweepable, which is what the old paper's "unclear curriculum" (R4 O7) lacked. A
documented half-life *is* the curriculum.

Future-dated samples (``t_i > t_ref``) are clamped to zero age (weight 1): a
retrain never sees the future, so this only guards against an off-by-one in the
caller, it is not look-ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["recency_weights", "RecencyWeighting"]


def recency_weights(
    times: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ref_time: float,
    lam: float,
    *,
    normalize: bool = False,
) -> npt.NDArray[np.float64]:
    """Exponential-decay weights for samples aged ``ref_time - times``.

    Args:
        times: per-sample most-recent-activity time.
        ref_time: the retrain "now"; ages are ``ref_time - times``, clamped at 0.
        lam: decay rate ``>= 0``. ``0`` gives all-ones (uniform).
        normalize: if set, rescale so the weights average 1 (keeps the loss
            scale comparable across windows / lambda values).

    Returns:
        ``float64`` weights, same shape as ``times``.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")
    t = np.asarray(times, dtype=np.float64)
    age = np.maximum(ref_time - t, 0.0)
    w = np.exp(-lam * age)
    if normalize and w.sum() > 0:
        w = w * (w.size / w.sum())
    return w


@dataclass(frozen=True)
class RecencyWeighting:
    """A recency schedule, parameterised by decay rate or half-life.

    Prefer :meth:`from_half_life`: a half-life in the data's time units is the
    interpretable knob the sensitivity study sweeps. ``lam = 0`` is uniform.
    """

    lam: float

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError("lam must be >= 0")

    @classmethod
    def from_half_life(cls, half_life: float) -> RecencyWeighting:
        """Build a schedule whose weight halves every ``half_life`` time units."""
        if half_life <= 0:
            raise ValueError("half_life must be > 0")
        return cls(lam=math.log(2.0) / half_life)

    @classmethod
    def uniform(cls) -> RecencyWeighting:
        """The ablation baseline: every sample weighted equally."""
        return cls(lam=0.0)

    @property
    def half_life(self) -> float:
        """Age at which weight falls to one half (``inf`` when uniform)."""
        return math.inf if self.lam == 0.0 else math.log(2.0) / self.lam

    def weights(
        self,
        times: npt.NDArray[np.integer] | npt.NDArray[np.floating],
        ref_time: float,
        *,
        normalize: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Weights for ``times`` relative to ``ref_time`` (see :func:`recency_weights`)."""
        return recency_weights(times, ref_time, self.lam, normalize=normalize)
