"""Concept-drift detectors (C4).

Three faithful, classical detectors, each implemented to its published
definition so the paper can cite the algorithm and a reviewer can check the
code against the paper:

* :class:`DDM` -- Drift Detection Method, Gama, Medas, Castillo & Rodrigues,
  *Learning with Drift Detection*, SBIA 2004. Consumes a **binary error**
  stream (``1`` = the model was wrong on this instance).
* :class:`ADWIN` -- ADaptive WINdowing, Bifet & Gavaldà, *Learning from
  Time-Changing Data with Adaptive Windowing*, SDM 2007. Consumes a
  **real-valued** stream and detects a change in its mean with a rigorous
  Hoeffding-style bound. We implement the exact ("ADWIN0") variant that keeps
  the full window and tests every split point; the exponential-histogram is a
  memory optimisation of the *same* test and is layered separately if needed.
* :class:`PageHinkley` -- the Page-Hinkley test (Page, *Continuous Inspection
  Schemes*, Biometrika 1954), as used for drift by Mouss et al. 2004.
  Consumes a **real-valued** stream and flags a persistent shift of the mean.

Design choice -- exact first. ADWIN0 keeps every value and computes the window
variance exactly, so its cut decision is the detector's true decision, not an
approximation. Memory is bounded by ``max_window`` with a documented guard.

The detectors are pure Python/NumPy with no torch dependency: they sit on the
streaming critical path and must stay cheap.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import IntEnum

__all__ = ["DriftStatus", "DriftDetector", "DDM", "ADWIN", "PageHinkley"]


class DriftStatus(IntEnum):
    """Result of feeding one observation to a detector.

    ``STABLE`` -- no change. ``WARNING`` -- a change may be starting (DDM only;
    a signal to start buffering training data, not yet to retrain). ``DRIFT`` --
    a change is confirmed.
    """

    STABLE = 0
    WARNING = 1
    DRIFT = 2


class DriftDetector(ABC):
    """Interface shared by every detector.

    Contract: :meth:`update` is called once per stream element with a single
    scalar and returns the :class:`DriftStatus` for that step. A detector that
    returns ``DRIFT`` resets its own internal baseline so detection can
    continue on the post-change stream. :attr:`status` echoes the most recent
    return value.
    """

    def __init__(self) -> None:
        self._status = DriftStatus.STABLE

    @property
    def status(self) -> DriftStatus:
        """Status returned by the most recent :meth:`update`."""
        return self._status

    @property
    def drift_detected(self) -> bool:
        return self._status == DriftStatus.DRIFT

    @property
    def warning_detected(self) -> bool:
        return self._status == DriftStatus.WARNING

    @abstractmethod
    def update(self, value: float) -> DriftStatus:
        """Feed one observation; return this step's status."""

    @abstractmethod
    def reset(self) -> None:
        """Restore the detector to its initial state."""


class DDM(DriftDetector):
    """Drift Detection Method (Gama et al. 2004).

    Monitors the online error rate of a classifier. With ``i`` instances seen,
    the error rate ``p_i`` has standard deviation ``s_i = sqrt(p_i (1-p_i)/i)``.
    Because error *decreases* as a stationary learner improves, the method
    records the minimum of ``p_i + s_i`` (``p_min + s_min``) and flags change
    when the current ``p_i + s_i`` rises well above that minimum:

    * warning when ``p_i + s_i >= p_min + warn_level * s_min`` (default 2 sigma),
    * drift   when ``p_i + s_i >= p_min + drift_level * s_min`` (default 3 sigma).

    The minimum is only tracked, and warnings/drifts only raised, after
    ``min_instances`` observations, so early noise does not fix a bad baseline.
    On drift the baseline is reset.

    Input: a binary error indicator (``1`` if the prediction was wrong).
    """

    def __init__(
        self,
        min_instances: int = 30,
        warn_level: float = 2.0,
        drift_level: float = 3.0,
    ) -> None:
        super().__init__()
        if not 0.0 < warn_level <= drift_level:
            raise ValueError("require 0 < warn_level <= drift_level")
        if min_instances < 1:
            raise ValueError("min_instances must be >= 1")
        self.min_instances = int(min_instances)
        self.warn_level = float(warn_level)
        self.drift_level = float(drift_level)
        self.reset()

    def reset(self) -> None:
        self._status = DriftStatus.STABLE
        self._n = 0
        self._p = 1.0  # running error mean
        self._p_min = math.inf
        self._s_min = math.inf

    @property
    def error_rate(self) -> float:
        """Current online error estimate ``p_i``."""
        return self._p

    @property
    def n(self) -> int:
        return self._n

    def update(self, value: float) -> DriftStatus:
        error = 1.0 if value else 0.0
        # Incremental mean of the 0/1 error stream.
        self._n += 1
        self._p += (error - self._p) / self._n
        s = math.sqrt(self._p * (1.0 - self._p) / self._n)

        if self._n < self.min_instances:
            self._status = DriftStatus.STABLE
            return self._status

        if self._p + s <= self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = s

        threshold = self._p + s
        if threshold >= self._p_min + self.drift_level * self._s_min:
            # Reset the baseline, then report DRIFT: reset() clears _status, so
            # the assignment must come after it or the drift is swallowed.
            self.reset()
            self._status = DriftStatus.DRIFT
        elif threshold >= self._p_min + self.warn_level * self._s_min:
            self._status = DriftStatus.WARNING
        else:
            self._status = DriftStatus.STABLE
        return self._status


class ADWIN(DriftDetector):
    """ADaptive WINdowing, exact variant (Bifet & Gavaldà 2007).

    Keeps a window ``W`` of the most recent real values. After each insertion it
    asks: is there a split of ``W`` into an older ``W0`` and a newer ``W1`` whose
    means differ by more than could be explained by chance? The test uses the
    variance-aware bound from the paper: with ``n0, n1`` the sub-window sizes,
    ``m = 1 / (1/n0 + 1/n1)``, window variance ``v``, and
    ``dd = ln(2 / delta')`` where ``delta' = delta / |W|`` (a Bonferroni
    correction over the split points),

        ``eps_cut = sqrt((2/m) * v * dd) + (2/3) * dd / m``.

    If ``|mean(W0) - mean(W1)| > eps_cut`` for any split, the change is real:
    the detector drops ``W0``, keeps ``W1``, and returns ``DRIFT``. ADWIN has no
    warning level.

    Input: any real-valued stream (e.g. a per-event risk score, or a per-window
    error rate). ``max_window`` caps memory; when exceeded the oldest value is
    evicted (a documented guard, not part of the statistical test).
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window: int = 10,
        max_window: int = 5000,
    ) -> None:
        super().__init__()
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if min_window < 2:
            raise ValueError("min_window must be >= 2")
        if max_window < 2 * min_window:
            raise ValueError("max_window must be >= 2 * min_window")
        self.delta = float(delta)
        self.min_window = int(min_window)
        self.max_window = int(max_window)
        self.reset()

    def reset(self) -> None:
        self._status = DriftStatus.STABLE
        self._window: list[float] = []

    @property
    def width(self) -> int:
        """Current window length."""
        return len(self._window)

    @property
    def estimation(self) -> float:
        """Mean over the current window (the adaptive estimate)."""
        w = self._window
        return sum(w) / len(w) if w else 0.0

    def update(self, value: float) -> DriftStatus:
        self._window.append(float(value))
        if len(self._window) > self.max_window:
            self._window.pop(0)

        self._status = DriftStatus.STABLE
        # Shrink greedily: a single value can create at most one valid cut, but
        # keep checking so a long run of stale data is fully flushed.
        while self._try_cut():
            self._status = DriftStatus.DRIFT
        return self._status

    def _try_cut(self) -> bool:
        w = self._window
        total = len(w)
        if total < 2 * self.min_window:
            return False

        mean = sum(w) / total
        var = sum((x - mean) ** 2 for x in w) / total
        delta_prime = self.delta / total
        dd = math.log(2.0 / delta_prime)

        # Prefix sums let every split be tested in O(1); scan oldest-to-newest.
        prefix = 0.0
        for n0 in range(self.min_window, total - self.min_window + 1):
            prefix += w[n0 - 1]
            n1 = total - n0
            mean0 = prefix / n0
            mean1 = (mean * total - prefix) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps_cut = math.sqrt((2.0 / m) * var * dd) + (2.0 / 3.0) * dd / m
            if abs(mean0 - mean1) > eps_cut:
                # Change is real: drop the stale prefix W0, keep the recent W1.
                del self._window[:n0]
                return True
        return False


class PageHinkley(DriftDetector):
    """Page-Hinkley test (Page 1954; Mouss et al. 2004 for drift).

    Tracks the cumulative difference between each value and the running mean,
    minus a tolerance ``delta`` that the mean is allowed to wander by without
    alarm. For an *increase*, ``m_T = sum_t (x_t - xbar_t - delta)`` and the
    test fires when ``m_T - min_t m_T > lambda_``; the decrease case is
    symmetric with ``max``. ``direction`` selects which to watch.

    Input: any real-valued stream. Fires ``DRIFT`` (no warning level) and resets
    on firing.
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        direction: str = "both",
        min_instances: int = 30,
    ) -> None:
        super().__init__()
        if direction not in ("increase", "decrease", "both"):
            raise ValueError("direction must be increase, decrease or both")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be > 0")
        self.delta = float(delta)
        self.lambda_ = float(lambda_)
        self.direction = direction
        self.min_instances = int(min_instances)
        self.reset()

    def reset(self) -> None:
        self._status = DriftStatus.STABLE
        self._n = 0
        self._mean = 0.0
        self._m_inc = 0.0  # cumulative sum for increase test
        self._m_dec = 0.0  # cumulative sum for decrease test
        self._m_inc_min = 0.0
        self._m_dec_max = 0.0

    @property
    def n(self) -> int:
        return self._n

    def update(self, value: float) -> DriftStatus:
        x = float(value)
        self._n += 1
        self._mean += (x - self._mean) / self._n

        # Increase branch: accumulate positive excursions above mean + delta.
        self._m_inc += x - self._mean - self.delta
        self._m_inc_min = min(self._m_inc_min, self._m_inc)
        # Decrease branch: accumulate negative excursions below mean - delta.
        self._m_dec += x - self._mean + self.delta
        self._m_dec_max = max(self._m_dec_max, self._m_dec)

        self._status = DriftStatus.STABLE
        if self._n < self.min_instances:
            return self._status

        up = (self.direction in ("increase", "both")) and (
            self._m_inc - self._m_inc_min > self.lambda_
        )
        down = (self.direction in ("decrease", "both")) and (
            self._m_dec_max - self._m_dec > self.lambda_
        )
        if up or down:
            # reset() clears _status; assign DRIFT after it so it is reported.
            self.reset()
            self._status = DriftStatus.DRIFT
        return self._status
