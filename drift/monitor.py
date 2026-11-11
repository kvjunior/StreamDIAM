"""Streaming drift monitor (C4).

The monitor watches the live system and turns its behaviour into drift signals.
It has two independent branches, because in illicit-account detection the two
informative streams arrive on very different clocks:

* **Supervised branch.** Ground-truth labels, when they arrive (often *late* --
  an account is flagged as illicit well after its activity), feed a prequential
  (test-then-train) error estimate, which feeds a supervised detector (DDM by
  default). This is the Gama et al. 2013 interleaved-test-then-train estimate,
  with an optional fading factor so old errors decay.

* **Unsupervised branch.** The risk scores the model emits for *every* event are
  available immediately, with no label needed. Their running mean feeds a
  real-valued detector (ADWIN by default), catching a shift in the score
  distribution before any label confirms it.

Keeping the branches separate -- :meth:`observe_label` and
:meth:`observe_score` are distinct calls -- is deliberate: the evaluation harness
can push scores every event and labels only when they materialise, which is the
honest model of this domain's label latency.

The monitor only *reports* status; deciding whether that status should launch a
retrain (cooldown, confirmation) is :mod:`streamdiam.drift.trigger`'s job.
"""

from __future__ import annotations

from streamdiam.drift.detectors import ADWIN, DDM, DriftDetector, DriftStatus

__all__ = ["PrequentialError", "DriftMonitor"]


class _DefaultSentinel:
    """Marker so ``None`` can mean 'disable a branch' while omission means default."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<default>"


# Defined before DriftMonitor so it can serve as a default argument value.
_DEFAULT: DriftDetector = _DefaultSentinel()  # type: ignore[assignment]


class PrequentialError:
    """Interleaved test-then-train error with an optional fading factor.

    Gama, Sebastião & Rodrigues, *On evaluating stream learning algorithms*,
    Machine Learning 2013. With fading factor ``alpha`` in ``(0, 1]`` the
    estimate is

        ``E_i = (alpha * S_{i-1} + loss_i) / (alpha * B_{i-1} + 1)``

    where ``S`` accumulates faded loss and ``B`` accumulates faded counts.
    ``alpha = 1`` recovers the plain cumulative error. Unknown labels
    (``y_true is None``) are skipped and leave the estimate unchanged.
    """

    def __init__(self, fading_factor: float = 1.0) -> None:
        if not 0.0 < fading_factor <= 1.0:
            raise ValueError("fading_factor must be in (0, 1]")
        self.fading_factor = float(fading_factor)
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0  # faded accumulated loss
        self._count = 0.0  # faded accumulated weight
        self._n = 0  # number of labelled instances seen

    @property
    def error(self) -> float:
        """Current prequential error estimate in ``[0, 1]``."""
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def n(self) -> int:
        """Number of labelled instances incorporated."""
        return self._n

    def update(self, y_true: int, y_pred: int) -> float:
        """Incorporate one labelled prediction; return the per-step loss (0/1).

        The caller feeds the returned loss to a supervised detector. The running
        :attr:`error` is updated as a side effect.
        """
        loss = 0.0 if int(y_true) == int(y_pred) else 1.0
        a = self.fading_factor
        self._sum = a * self._sum + loss
        self._count = a * self._count + 1.0
        self._n += 1
        return loss


class DriftMonitor:
    """Two-branch drift monitor over the live stream.

    Args:
        supervised: detector fed the 0/1 prequential-loss stream. Defaults to
            :class:`DDM`. Pass ``None`` to disable the supervised branch.
        unsupervised: detector fed the running mean risk score. Defaults to
            :class:`ADWIN`. Pass ``None`` to disable the unsupervised branch.
        fading_factor: forgetting for the prequential error (see
            :class:`PrequentialError`).

    The combined :meth:`status` is ``DRIFT`` if either branch has just drifted,
    else ``WARNING`` if the supervised branch warns, else ``STABLE``.
    """

    def __init__(
        self,
        supervised: DriftDetector | None = _DEFAULT,
        unsupervised: DriftDetector | None = _DEFAULT,
        fading_factor: float = 1.0,
    ) -> None:
        # Sentinel distinguishes "omitted -> use default" from explicit None
        # (disable the branch) from an explicit detector instance (use it).
        self.supervised = DDM() if supervised is _DEFAULT else supervised
        self.unsupervised = ADWIN() if unsupervised is _DEFAULT else unsupervised
        self.prequential = PrequentialError(fading_factor)
        self._sup_status = DriftStatus.STABLE
        self._unsup_status = DriftStatus.STABLE

    @property
    def supervised_status(self) -> DriftStatus:
        return self._sup_status

    @property
    def unsupervised_status(self) -> DriftStatus:
        return self._unsup_status

    @property
    def prequential_error(self) -> float:
        return self.prequential.error

    def status(self) -> DriftStatus:
        """Combined status from the most recent observations of each branch."""
        if DriftStatus.DRIFT in (self._sup_status, self._unsup_status):
            return DriftStatus.DRIFT
        if self._sup_status == DriftStatus.WARNING:
            return DriftStatus.WARNING
        return DriftStatus.STABLE

    def observe_label(self, y_true: int, y_pred: int) -> DriftStatus:
        """Feed one (possibly delayed) ground-truth label to the supervised branch."""
        loss = self.prequential.update(y_true, y_pred)
        if self.supervised is not None:
            self._sup_status = self.supervised.update(loss)
        return self._sup_status

    def observe_score(self, score: float) -> DriftStatus:
        """Feed one immediate risk score to the unsupervised branch."""
        if self.unsupervised is not None:
            self._unsup_status = self.unsupervised.update(score)
        return self._unsup_status

    def reset(self) -> None:
        """Reset both branches and the prequential estimate (post-retrain)."""
        self.prequential.reset()
        if self.supervised is not None:
            self.supervised.reset()
        if self.unsupervised is not None:
            self.unsupervised.reset()
        self._sup_status = DriftStatus.STABLE
        self._unsup_status = DriftStatus.STABLE
