"""Strict time-ordered train / validation / test split (no future leakage).

For a streaming detector the split must respect time: everything used for
training must precede everything used for testing. This module splits *edges*
by timestamp at two boundaries chosen so the cumulative edge fractions match
the requested proportions. By construction every test edge has a strictly
greater timestamp than every training and validation edge, which is the
``docs/problem_formalization.md`` §3.2 no-future-leakage property, and
``tests/test_data.py`` checks it on real data.

(The common random node split used by the old pipeline is exactly what lets the
future leak into the past; it is deliberately not offered here.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from streamdiam.data.schema import SchemaError

__all__ = ["TemporalSplit", "temporal_split"]


@dataclass(frozen=True)
class TemporalSplit:
    """Boolean edge masks for the three splits plus the timestamp boundaries.

    The masks partition the edges exactly (disjoint, covering). ``t_train`` is
    the last timestamp assigned to training and ``t_val`` the last assigned to
    validation, so ``train = {t <= t_train}``, ``val = {t_train < t <= t_val}``,
    ``test = {t > t_val}``.
    """

    train_mask: npt.NDArray[np.bool_]
    val_mask: npt.NDArray[np.bool_]
    test_mask: npt.NDArray[np.bool_]
    t_train: int
    t_val: int

    @property
    def sizes(self) -> tuple[int, int, int]:
        return (
            int(self.train_mask.sum()),
            int(self.val_mask.sum()),
            int(self.test_mask.sum()),
        )


def temporal_split(
    timestamps: npt.NDArray[np.integer],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> TemporalSplit:
    """Split edges by time into (train, val, test) at cumulative ``fractions``.

    Args:
        timestamps: Per-edge timestamps (any order).
        fractions: Target (train, val, test) proportions; must be positive and
            sum to 1.

    Returns:
        A :class:`TemporalSplit`. Because whole timestamps are kept together,
        realized sizes may differ slightly from the targets when many edges
        share a boundary timestamp; the ordering guarantee is exact regardless.
    """
    if len(fractions) != 3:
        raise SchemaError(f"fractions must have 3 entries, got {len(fractions)}")
    if any(f <= 0 for f in fractions):
        raise SchemaError(f"fractions must be positive, got {fractions}")
    if not np.isclose(sum(fractions), 1.0):
        raise SchemaError(f"fractions must sum to 1, got sum={sum(fractions)}")

    ts = np.asarray(timestamps)
    m = ts.shape[0]
    if m == 0:
        raise SchemaError("cannot split an empty edge set")

    f_train, f_val, _ = fractions
    sorted_t = np.sort(ts)
    # Index of the boundary edge in time order; clip to the last valid index.
    i_train = min(m - 1, int(f_train * m))
    i_val = min(m - 1, int((f_train + f_val) * m))
    t_train = int(sorted_t[i_train])
    t_val = max(int(sorted_t[i_val]), t_train)  # keep t_val >= t_train

    train_mask = ts <= t_train
    test_mask = ts > t_val
    val_mask = ~train_mask & ~test_mask
    return TemporalSplit(train_mask, val_mask, test_mask, t_train, t_val)
