"""Streaming evaluation metrics (C5).

Illicit accounts are a 2-10% minority, so accuracy is meaningless and the metrics
are imbalance-aware: F1 for the illicit (positive) class at a threshold, plus
threshold-free ROC-AUC and average precision (AP). AP is the headline -- it is
the area under precision-recall, which is what matters when positives are rare.

Degenerate windows are the norm early in a stream (a tick with no labelled
accounts, or only negatives). They are handled explicitly: AUC needs both
classes and is ``nan`` otherwise; AP needs at least one positive; F1 is 0 when
the positive class is absent from either predictions or truth. ``nan`` metrics
are carried through and excluded from averages rather than crashing the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = ["WindowMetrics", "compute_classification_metrics", "PrequentialAccumulator"]


@dataclass(frozen=True)
class WindowMetrics:
    """Classification metrics over one set of (label, score) pairs."""

    n: int
    n_pos: int
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float  # nan if only one class present
    ap: float  # nan if no positives
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "n_pos": self.n_pos,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "ap": self.ap,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def compute_classification_metrics(
    y_true: npt.NDArray[np.integer],
    y_score: npt.NDArray[np.floating],
    threshold: float = 0.5,
) -> WindowMetrics:
    """Imbalance-aware metrics for binary illicit detection (positive = illicit).

    ``y_true`` is in ``{0, 1}`` (filter unknowns before calling). Empty input
    yields an all-zero/``nan`` record rather than raising.
    """
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score, dtype=np.float64)
    n = int(yt.size)
    if n == 0:
        return WindowMetrics(0, 0, threshold, 0.0, 0.0, 0.0, float("nan"),
                             float("nan"), 0, 0, 0, 0)

    yp = (ys >= threshold).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    n_pos = int((yt == 1).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    # AUC needs both classes; AP needs at least one positive.
    both_classes = 0 < n_pos < n
    roc_auc = float(roc_auc_score(yt, ys)) if both_classes else float("nan")
    ap = float(average_precision_score(yt, ys)) if n_pos > 0 else float("nan")

    return WindowMetrics(
        n=n, n_pos=n_pos, threshold=threshold,
        precision=precision, recall=recall, f1=f1,
        roc_auc=roc_auc, ap=ap, tp=tp, fp=fp, fn=fn, tn=tn,
    )


class PrequentialAccumulator:
    """Accumulates leakage-free first-appearance (label, score) decisions.

    Each account is scored exactly once, at its first appearance in the stream,
    *before* it can enter any training window. The accumulated pairs give one
    strict, leakage-free summary metric for the whole run, complementing the
    per-tick timeline.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._labels: list[int] = []
        self._scores: list[float] = []

    def add(self, label: int, score: float) -> None:
        """Record one first-appearance decision for a labelled account."""
        self._labels.append(int(label))
        self._scores.append(float(score))

    def __len__(self) -> int:
        return len(self._labels)

    def metrics(self) -> WindowMetrics:
        return compute_classification_metrics(
            np.asarray(self._labels), np.asarray(self._scores), self.threshold
        )
