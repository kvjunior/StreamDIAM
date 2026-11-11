"""Drift-triggered retrain policy (C4).

The trigger turns the monitor's per-step status into *retrain decisions*. It is
the component that makes the paper's thesis concrete: retraining is **decoupled
and drift-triggered**, never on the critical path. The trigger itself does no
training -- it emits a :class:`RetrainSignal` and records a :class:`TriggerEvent`.
The background trainer (``retrain/``, built next) consumes the signal; the
evaluation harness reads the event log, whose firing count and timing are the
headline RQ3 result ("how often is retraining actually needed?").

Policy knobs, each a defensible default the sensitivity study sweeps:

* ``cooldown`` -- after a firing, suppress new firings for this many steps. While
  a background retrain is in flight, re-firing would only thrash; the cooldown
  models the retrain-and-swap latency and keeps the firing log honest.
* ``require_confirmation`` -- fire only on ``DRIFT``, treating ``WARNING`` as a
  logged-but-not-fired heads-up (the DDM warning zone). Off => warnings never
  fire regardless, since only ``DRIFT`` reaches the fire path anyway; the flag
  exists so a future policy can choose to act on sustained warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from streamdiam.drift.detectors import DriftStatus
from streamdiam.drift.monitor import DriftMonitor

__all__ = [
    "RetrainSignal",
    "TriggerEvent",
    "Trigger",
    "RetrainTrigger",
    "PeriodicTrigger",
]


@runtime_checkable
class Trigger(Protocol):
    """Structural interface the streaming harness calls once per tick.

    Both the drift-driven :class:`RetrainTrigger` and the fixed-schedule
    :class:`PeriodicTrigger` satisfy it, so either policy drops into the same
    loop. ``events`` / ``firing_count`` expose the firing log (an RQ3 result).
    """

    events: list[TriggerEvent]

    @property
    def firing_count(self) -> int: ...

    def step(self, stream_time: int | None = ...) -> RetrainSignal | None: ...


@dataclass(frozen=True)
class RetrainSignal:
    """Emitted when the policy decides a retrain should start.

    Attributes:
        step_index: stream position (0-based event count) at the firing.
        stream_time: domain timestamp at the firing, if the caller supplies one.
        reason: which branch drove the firing (``"supervised"`` /
            ``"unsupervised"``).
        prequential_error: the supervised error estimate at the firing.
    """

    step_index: int
    stream_time: int | None
    reason: str
    prequential_error: float


@dataclass(frozen=True)
class TriggerEvent:
    """One logged firing (a superset of :class:`RetrainSignal` for analysis)."""

    step_index: int
    stream_time: int | None
    reason: str
    prequential_error: float
    supervised_status: DriftStatus
    unsupervised_status: DriftStatus


@dataclass
class RetrainTrigger:
    """Drift-triggered retrain policy over a :class:`DriftMonitor`.

    Call :meth:`step` once per stream element *after* feeding that element's
    label and/or score to the monitor. It returns a :class:`RetrainSignal` on
    the steps where a retrain should launch, and ``None`` otherwise. Every
    firing is appended to :attr:`events`.
    """

    monitor: DriftMonitor
    cooldown: int = 0
    require_confirmation: bool = True
    events: list[TriggerEvent] = field(default_factory=list)
    _step: int = field(default=0, repr=False)
    _last_fire: int | None = field(default=None, repr=False)

    @property
    def firing_count(self) -> int:
        """Number of retrains the policy has triggered -- the RQ3 headline."""
        return len(self.events)

    def _in_cooldown(self, idx: int) -> bool:
        return (
            self._last_fire is not None and idx - self._last_fire < self.cooldown
        )

    def step(self, stream_time: int | None = None) -> RetrainSignal | None:
        """Advance one step; emit a signal iff the policy fires here."""
        sup = self.monitor.supervised_status
        unsup = self.monitor.unsupervised_status
        idx = self._step
        self._step += 1

        if self._in_cooldown(idx):
            return None

        fired = sup == DriftStatus.DRIFT or unsup == DriftStatus.DRIFT
        if self.require_confirmation and not fired:
            return None
        if not fired:
            return None

        reason = "supervised" if sup == DriftStatus.DRIFT else "unsupervised"
        err = self.monitor.prequential_error
        self._last_fire = idx
        self.events.append(
            TriggerEvent(
                step_index=idx,
                stream_time=stream_time,
                reason=reason,
                prequential_error=err,
                supervised_status=sup,
                unsupervised_status=unsup,
            )
        )
        return RetrainSignal(
            step_index=idx,
            stream_time=stream_time,
            reason=reason,
            prequential_error=err,
        )

    def reset(self) -> None:
        """Clear the firing log and counters (not the monitor)."""
        self.events = []
        self._step = 0
        self._last_fire = None


@dataclass
class PeriodicTrigger:
    """Fixed-schedule trigger: retrain every ``period`` ticks, drift-blind.

    This is the DIAM *full-retrain-per-window* baseline's policy -- the opposite
    extreme from drift-triggered retraining. Running it through the same harness
    lets RQ3 quantify how much work drift-triggering saves versus retraining on a
    fixed schedule, at what accuracy cost. It reads no monitor.
    """

    period: int
    events: list[TriggerEvent] = field(default_factory=list)
    _step: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError("period must be >= 1")

    @property
    def firing_count(self) -> int:
        return len(self.events)

    def step(self, stream_time: int | None = None) -> RetrainSignal | None:
        idx = self._step
        self._step += 1
        # Fire at ticks period, 2*period, ... (never on the very first tick).
        if idx == 0 or idx % self.period != 0:
            return None
        self.events.append(
            TriggerEvent(
                step_index=idx,
                stream_time=stream_time,
                reason="periodic",
                prequential_error=float("nan"),
                supervised_status=DriftStatus.STABLE,
                unsupervised_status=DriftStatus.STABLE,
            )
        )
        return RetrainSignal(
            step_index=idx,
            stream_time=stream_time,
            reason="periodic",
            prequential_error=float("nan"),
        )

    def reset(self) -> None:
        self.events = []
        self._step = 0
