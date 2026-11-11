"""Prequential (test-then-train) streaming protocol (C5).

One honest loop produces every streaming number, which is what resolves the old
paper's incremental-vs-full-retrain F1 contradiction (R2 O7): there are not two
tables from two harnesses, there is one harness.

Per stream tick (one timestamp's delta):

1. **Test.** Score the active accounts with the *current* view -- the model as it
   stands before this tick's edges or any retrain that uses them. Accounts seen
   for the first time are recorded as strict, leakage-free prequential decisions;
   all labelled active accounts feed the per-tick window metric (the F1-over-time
   timeline).
2. **Update.** Insert the delta into the store and update the maintained view
   incrementally (timed -- this latency is RQ1).
3. **Monitor.** Feed the refreshed scores and any known labels to the drift
   monitor.
4. **Retrain (decoupled).** If the trigger fires, call the retrain hook (which
   fine-tunes and hot-swaps), then rebuild the view and reset the monitor.

Ordering guarantees no look-ahead: scoring precedes the store/view update and
precedes the retrain, so the model that scores tick ``t`` was last trained strictly
before ``t``. ``test_eval`` checks this invariant on real data.

The protocol is maintainer-agnostic: any object satisfying
:class:`StreamingMaintainer` plugs in (from-scratch here; incremental and the
baselines in later batches), so every strategy runs through the identical loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from streamdiam.data.schema import TransactionBatch
from streamdiam.drift import DriftMonitor, Trigger
from streamdiam.eval.metrics import (
    PrequentialAccumulator,
    WindowMetrics,
    compute_classification_metrics,
)
from streamdiam.maintain.from_scratch import FromScratchMaintainer
from streamdiam.model.diam import DIAM
from streamdiam.store.interface import TemporalMultigraphStore
from streamdiam.utils.memory import peak_rss_bytes
from streamdiam.utils.timing import LatencyRecorder, LatencySummary, Stopwatch

__all__ = [
    "StreamingMaintainer",
    "FromScratchStreamingMaintainer",
    "TickRecord",
    "RunResult",
    "PrequentialProtocol",
]


class StreamingMaintainer(ABC):
    """What the protocol needs from any view-maintenance strategy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result records."""

    @abstractmethod
    def scores(self) -> npt.NDArray[np.float64]:
        """Current risk probabilities ``[universe_size]``."""

    @abstractmethod
    def ingest(self, delta: TransactionBatch) -> None:
        """Update the view for ``delta`` (already inserted into the store)."""

    @abstractmethod
    def rebuild(self) -> None:
        """Recompute the view from the current model and store (after a swap)."""


class FromScratchStreamingMaintainer(StreamingMaintainer):
    """Reference strategy: recompute the whole view each tick.

    Always correct (it is the oracle), so it anchors the harness and doubles as
    the ``from-scratch`` baseline for RQ1's speed comparison. ``model_provider``
    is read fresh each recompute so a hot-swap is picked up automatically.
    """

    def __init__(
        self,
        model_provider: Callable[[], DIAM],
        store: TemporalMultigraphStore,
        universe_size: int,
        t_max: int = 10_000,
        chunk_size: int = 10_000,
    ) -> None:
        self._model_provider = model_provider
        self._store = store
        self._n = int(universe_size)
        self._t_max = t_max
        self._chunk = chunk_size
        self._scores = np.zeros(self._n, dtype=np.float64)

    @property
    def name(self) -> str:
        return "from_scratch"

    def _recompute(self) -> None:
        view = FromScratchMaintainer(
            self._model_provider(), t_max=self._t_max, chunk_size=self._chunk
        ).compute_view(self._store, self._n)
        self._scores = view.scores.detach().cpu().numpy().astype(np.float64)

    def ingest(self, delta: TransactionBatch) -> None:
        self._recompute()

    def rebuild(self) -> None:
        self._recompute()

    def scores(self) -> npt.NDArray[np.float64]:
        return self._scores


@dataclass(frozen=True)
class TickRecord:
    """Per-tick result row (JSONL-serializable via :meth:`as_dict`)."""

    t: int
    n_active: int
    n_new_labelled: int
    latency_ms: float
    drift_status: int
    retrained: bool
    model_version: int
    window: WindowMetrics

    def as_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "t": self.t,
            "n_active": self.n_active,
            "n_new_labelled": self.n_new_labelled,
            "latency_ms": self.latency_ms,
            "drift_status": self.drift_status,
            "retrained": self.retrained,
            "model_version": self.model_version,
        }
        d.update({f"window_{k}": v for k, v in self.window.as_dict().items()})
        return d


@dataclass
class RunResult:
    """The whole run's results -- the only thing a run emits (no plots here)."""

    maintainer: str
    ticks: list[TickRecord]
    prequential: WindowMetrics
    latency: LatencySummary
    throughput_eps: float
    n_retrains: int
    retrain_times: list[int] = field(default_factory=list)
    peak_rss_bytes: int = 0

    def summary_dict(self) -> dict[str, object]:
        return {
            "maintainer": self.maintainer,
            "n_ticks": len(self.ticks),
            "prequential": self.prequential.as_dict(),
            "latency": self.latency.as_dict(),
            "throughput_eps": self.throughput_eps,
            "n_retrains": self.n_retrains,
            "retrain_times": list(self.retrain_times),
            "peak_rss_bytes": self.peak_rss_bytes,
        }


class PrequentialProtocol:
    """Run one maintainer through the test-then-train streaming loop."""

    def __init__(
        self,
        store: TemporalMultigraphStore,
        universe_size: int,
        maintainer: StreamingMaintainer,
        labels: npt.NDArray[np.integer],
        *,
        monitor: DriftMonitor | None = None,
        trigger: Trigger | None = None,
        retrain_hook: Callable[[int, npt.NDArray[np.bool_]], int | None] | None = None,
        version_provider: Callable[[], int] | None = None,
        threshold: float = 0.5,
        initial_seen: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        self.store = store
        self.n = int(universe_size)
        self.maintainer = maintainer
        self.labels = np.asarray(labels)
        self.monitor = monitor
        self.trigger = trigger
        self.retrain_hook = retrain_hook
        self.version_provider = version_provider or (lambda: 0)
        self.threshold = threshold
        # Accounts already trained on during warmup: excluded from the strict
        # leakage-free first-appearance metric.
        self.initial_seen = (
            np.zeros(self.n, dtype=bool)
            if initial_seen is None
            else np.asarray(initial_seen, dtype=bool).copy()
        )

    def run(self, stream: Iterable[TransactionBatch]) -> RunResult:
        self.maintainer.rebuild()  # initial view over the starting store
        seen = self.initial_seen.copy()
        preq = PrequentialAccumulator(self.threshold)
        latency = LatencyRecorder()
        ticks: list[TickRecord] = []
        retrain_times: list[int] = []

        for delta in stream:
            t = int(delta.timestamp[0])
            active = np.unique(np.concatenate([delta.src, delta.dst])).astype(np.int64)
            scores_now = self.maintainer.scores()
            version_used = int(self.version_provider())

            # 1a) strict leakage-free decision for first-appearance accounts
            fresh = active[~seen[active]]
            fresh_lbl = self.labels[fresh]
            for acc, lab in zip(fresh.tolist(), fresh_lbl.tolist()):
                if lab >= 0:
                    preq.add(int(lab), float(scores_now[acc]))
            # 1b) per-tick window over all labelled active accounts
            act_lbl = self.labels[active]
            known = act_lbl >= 0
            window = compute_classification_metrics(
                act_lbl[known], scores_now[active][known], self.threshold
            )
            n_new_labelled = int((fresh_lbl >= 0).sum())
            seen[active] = True

            # 2) update store + view (timed)
            self.store.insert(delta)
            with Stopwatch() as sw:
                self.maintainer.ingest(delta)
            latency.record(sw.elapsed)

            # 3) feed the drift monitor on the refreshed scores
            drift_status = 0
            if self.monitor is not None:
                new_scores = self.maintainer.scores()
                for acc in active.tolist():
                    self.monitor.observe_score(float(new_scores[acc]))
                    lab = int(self.labels[acc])
                    if lab >= 0:
                        pred = 1 if new_scores[acc] >= self.threshold else 0
                        self.monitor.observe_label(lab, pred)
                drift_status = int(self.monitor.status())

            # 4) decoupled retrain on trigger
            retrained = False
            if self.trigger is not None:
                sig = self.trigger.step(stream_time=t)
                if sig is not None and self.retrain_hook is not None:
                    new_version = self.retrain_hook(t, seen.copy())
                    if new_version is not None:
                        self.maintainer.rebuild()
                        if self.monitor is not None:
                            self.monitor.reset()
                        retrained = True
                        retrain_times.append(t)

            ticks.append(
                TickRecord(
                    t=t,
                    n_active=int(active.size),
                    n_new_labelled=n_new_labelled,
                    latency_ms=sw.elapsed * 1000.0,
                    drift_status=drift_status,
                    retrained=retrained,
                    model_version=version_used,
                    window=window,
                )
            )

        return RunResult(
            maintainer=self.maintainer.name,
            ticks=ticks,
            prequential=preq.metrics(),
            latency=latency.summary(),
            throughput_eps=latency.throughput_eps,
            n_retrains=len(retrain_times),
            retrain_times=retrain_times,
            peak_rss_bytes=peak_rss_bytes(),
        )
