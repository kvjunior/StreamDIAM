"""Tests for the drift package (C4).

Two kinds of test here, and the distinction matters:

* **Algorithm-correctness tests** feed *controlled synthetic input streams* to a
  detector to check its published mathematical property (a stationary stream
  must not raise a false alarm; an injected mean/error shift must be caught, and
  caught *after* the change). These exercise the detector math; they do not
  produce any paper result, so no "no-synthetic-in-results" rule applies.
* **Real-data tests** run on the real Elliptic stream: ADWIN detecting the
  dataset's genuine opening regime shift at its true change point (timestep 7),
  and the monitor running end-to-end on real risk scores from the from-scratch
  oracle. These are the ones that back any claim about behaviour on real data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from streamdiam.data import TransactionBatch, load_elliptic  # noqa: E402
from streamdiam.drift import (  # noqa: E402
    ADWIN,
    DDM,
    DriftMonitor,
    DriftStatus,
    PageHinkley,
    PrequentialError,
    RetrainTrigger,
)
from streamdiam.maintain import FromScratchMaintainer  # noqa: E402
from streamdiam.model import DIAM  # noqa: E402
from streamdiam.store import ReferencePyStore  # noqa: E402

ELLIPTIC_ROOT = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
needs_elliptic = pytest.mark.skipif(ELLIPTIC_ROOT is None, reason="set STREAMDIAM_ELLIPTIC_ROOT")

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# DDM
# --------------------------------------------------------------------------- #
def test_ddm_no_drift_on_stationary() -> None:
    d = DDM()
    fired = any(
        d.update(1.0 if rng.random() < 0.1 else 0.0) == DriftStatus.DRIFT
        for _ in range(2000)
    )
    assert not fired  # a stationary 10%-error stream must not alarm


def test_ddm_detects_error_increase() -> None:
    d = DDM()
    for _ in range(500):  # calm regime: 5% error
        d.update(1.0 if rng.random() < 0.05 else 0.0)
    drift_step = None
    for i in range(500):  # degraded regime: 60% error
        if d.update(1.0 if rng.random() < 0.6 else 0.0) == DriftStatus.DRIFT:
            drift_step = i
            break
    assert drift_step is not None and drift_step < 100  # caught, and promptly


def test_ddm_warning_precedes_drift() -> None:
    d = DDM()
    for _ in range(300):
        d.update(1.0 if rng.random() < 0.05 else 0.0)
    saw_warning = False
    for _ in range(300):
        s = d.update(1.0 if rng.random() < 0.7 else 0.0)
        if s == DriftStatus.WARNING:
            saw_warning = True
        if s == DriftStatus.DRIFT:
            break
    assert saw_warning  # DDM passes through the warning zone on the way to drift


# --------------------------------------------------------------------------- #
# ADWIN
# --------------------------------------------------------------------------- #
def test_adwin_stable_on_stationary() -> None:
    a = ADWIN(delta=0.002)
    fired = any(a.update(rng.normal(0.0, 1.0)) == DriftStatus.DRIFT for _ in range(3000))
    assert not fired


def test_adwin_detects_mean_shift_and_tracks() -> None:
    a = ADWIN(delta=0.002)
    for _ in range(1000):
        a.update(rng.normal(0.0, 0.1))
    drift_step = None
    for i in range(1000):
        if a.update(rng.normal(1.0, 0.1)) == DriftStatus.DRIFT:
            drift_step = i
            break
    assert drift_step is not None
    # After a cut the stale low-mean prefix is dropped; estimate moves up.
    for _ in range(200):
        a.update(rng.normal(1.0, 0.1))
    assert a.estimation > 0.5


def test_adwin_window_shrinks_on_cut() -> None:
    a = ADWIN(delta=0.002, min_window=5)
    for _ in range(500):
        a.update(0.0)
    w_before = a.width
    for _ in range(500):
        a.update(5.0)
    assert a.width < w_before  # the old homogeneous window was discarded


# --------------------------------------------------------------------------- #
# Page-Hinkley
# --------------------------------------------------------------------------- #
def test_page_hinkley_stable_on_stationary() -> None:
    # PH is designed for bounded signals (error rates / scores): test it at that
    # scale. Against lambda=50 the zero-mean random walk stays far from alarm.
    ph = PageHinkley(delta=0.005, lambda_=50.0)
    fired = any(ph.update(rng.normal(0.0, 0.05)) == DriftStatus.DRIFT for _ in range(3000))
    assert not fired


def test_page_hinkley_detects_increase() -> None:
    ph = PageHinkley(delta=0.005, lambda_=5.0, direction="increase")
    for _ in range(200):
        ph.update(rng.normal(0.0, 0.05))
    caught = any(
        ph.update(rng.normal(0.5, 0.05)) == DriftStatus.DRIFT for _ in range(400)
    )
    assert caught


def test_page_hinkley_increase_ignores_decrease() -> None:
    ph = PageHinkley(delta=0.005, lambda_=5.0, direction="increase")
    for _ in range(200):
        ph.update(rng.normal(0.0, 0.05))
    fired = any(
        ph.update(rng.normal(-0.5, 0.05)) == DriftStatus.DRIFT for _ in range(400)
    )
    assert not fired  # an increase-only detector must ignore a downward shift


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_detector_param_validation() -> None:
    with pytest.raises(ValueError):
        DDM(warn_level=4.0, drift_level=3.0)
    with pytest.raises(ValueError):
        ADWIN(delta=0.0)
    with pytest.raises(ValueError):
        PageHinkley(direction="sideways")


# --------------------------------------------------------------------------- #
# PrequentialError + DriftMonitor
# --------------------------------------------------------------------------- #
def test_prequential_error_basic_and_fading() -> None:
    pe = PrequentialError()
    for _ in range(100):
        pe.update(1, 1)  # all correct
    assert pe.error == 0.0 and pe.n == 100
    pe = PrequentialError()
    for i in range(100):
        pe.update(1, i % 2)  # wrong half the time
    assert 0.4 < pe.error < 0.6
    # Fading weights recent instances: after a correct run the faded estimate
    # sits below the unfaded cumulative estimate.
    faded, plain = PrequentialError(0.95), PrequentialError(1.0)
    for _ in range(50):
        faded.update(1, 0)
        plain.update(1, 0)  # all wrong
    for _ in range(50):
        faded.update(1, 1)
        plain.update(1, 1)  # all right
    assert faded.error < plain.error


def test_monitor_supervised_branch_detects() -> None:
    m = DriftMonitor(unsupervised=None)  # supervised-only
    for _ in range(400):
        y = 1 if rng.random() < 0.95 else 0
        m.observe_label(y_true=1, y_pred=y)  # ~5% error
    drifted = False
    for _ in range(400):
        y = 1 if rng.random() < 0.4 else 0
        if m.observe_label(y_true=1, y_pred=y) == DriftStatus.DRIFT:
            drifted = True
            break
    assert drifted and m.status() == DriftStatus.DRIFT


def test_monitor_unsupervised_branch_detects() -> None:
    m = DriftMonitor(supervised=None)  # unsupervised-only
    for _ in range(800):
        m.observe_score(float(rng.normal(0.1, 0.05)))
    drifted = any(
        m.observe_score(float(rng.normal(0.8, 0.05))) == DriftStatus.DRIFT
        for _ in range(800)
    )
    assert drifted


def test_monitor_disable_and_reset() -> None:
    m = DriftMonitor(supervised=None)
    assert m.supervised is None and m.unsupervised is not None
    m.observe_score(0.5)
    m.observe_label(1, 0)  # no supervised branch -> prequential still tracks
    assert m.prequential_error == 1.0
    m.reset()
    assert m.prequential_error == 0.0 and m.status() == DriftStatus.STABLE


# --------------------------------------------------------------------------- #
# RetrainTrigger
# --------------------------------------------------------------------------- #
class _StubMonitor:
    """Minimal stand-in exposing the three attributes the trigger reads."""

    def __init__(self) -> None:
        self.supervised_status = DriftStatus.STABLE
        self.unsupervised_status = DriftStatus.STABLE
        self.prequential_error = 0.0


def test_trigger_fires_logs_and_signal() -> None:
    mon = _StubMonitor()
    trig = RetrainTrigger(mon, cooldown=0)  # type: ignore[arg-type]
    assert trig.step(stream_time=10) is None  # stable -> no fire
    mon.supervised_status = DriftStatus.DRIFT
    mon.prequential_error = 0.42
    sig = trig.step(stream_time=11)
    assert sig is not None
    assert sig.reason == "supervised" and sig.prequential_error == 0.42
    assert sig.step_index == 1 and sig.stream_time == 11
    assert trig.firing_count == 1 and trig.events[0].reason == "supervised"


def test_trigger_cooldown_suppresses_refires() -> None:
    mon = _StubMonitor()
    trig = RetrainTrigger(mon, cooldown=5)  # type: ignore[arg-type]
    mon.unsupervised_status = DriftStatus.DRIFT
    fires = [trig.step() is not None for _ in range(10)]
    # One fire, then suppressed for `cooldown` steps, then allowed again.
    assert fires[0] is True
    assert sum(fires[1:5]) == 0
    assert fires[5] is True
    assert trig.firing_count == 2


def test_trigger_warning_does_not_fire() -> None:
    mon = _StubMonitor()
    trig = RetrainTrigger(mon, cooldown=0)  # type: ignore[arg-type]
    mon.supervised_status = DriftStatus.WARNING
    assert trig.step() is None and trig.firing_count == 0


# --------------------------------------------------------------------------- #
# Real Elliptic data
# --------------------------------------------------------------------------- #
@needs_elliptic
def test_real_elliptic_adwin_detects_regime_shift() -> None:
    """ADWIN catches Elliptic's genuine low->high opening shift at its true step.

    Steps 1-6 have a ~0.7% illicit rate among labelled source accounts; from
    step 7 it jumps above 8%. Fed the per-event indicator stream in canonical
    order, ADWIN's first cut must land at timestep 7 (not during the flat
    opening), tying the detection to the real change point.
    """
    graph = load_elliptic(ELLIPTIC_ROOT)
    e = graph.edges
    src = np.asarray(e.src)
    t = np.asarray(e.timestamp)
    lab = graph.account_label[src]
    known = lab >= 0
    indicator = (lab[known] == 1).astype(float)
    times = t[known]

    # The real shift exists in the data, independent of any detector.
    assert indicator[times <= 6].mean() < 0.02 < indicator[(times >= 7)].mean()

    a = ADWIN(delta=0.002)
    first_fire_time = None
    for x, tt in zip(indicator, times):
        if a.update(float(x)) == DriftStatus.DRIFT and first_fire_time is None:
            first_fire_time = int(tt)
    assert first_fire_time == 7  # detected exactly at the real change point


@needs_elliptic
def test_real_elliptic_monitor_end_to_end() -> None:
    """The monitor runs on real risk scores from the oracle without any mocking."""
    graph = load_elliptic(ELLIPTIC_ROOT)
    e = graph.edges
    K = 400
    idx = np.flatnonzero((e.src < K) & (e.dst < K))
    sub = TransactionBatch.from_arrays(
        np.arange(idx.size), e.src[idx], e.dst[idx], e.timestamp[idx], e.attr[idx]
    )
    store = ReferencePyStore(graph.num_features)
    store.insert(sub)
    model = DIAM(in_dim=graph.num_features, hidden_dim=24, num_layers=2)
    view = FromScratchMaintainer(model, t_max=10_000, chunk_size=10_000).compute_view(
        store, universe_size=K
    )
    scores = view.scores.tolist()

    monitor = DriftMonitor()
    trigger = RetrainTrigger(monitor, cooldown=50)
    active = np.unique(np.concatenate([sub.src, sub.dst]))
    for acc in active.tolist():
        s = scores[acc]
        monitor.observe_score(float(s))
        y_true = int(graph.account_label[acc])
        if y_true >= 0:  # labelled accounts feed the supervised branch
            monitor.observe_label(y_true=y_true, y_pred=1 if s > 0.5 else 0)
        trigger.step(stream_time=int(acc))

    assert 0.0 <= monitor.prequential_error <= 1.0
    assert isinstance(trigger.firing_count, int)  # well-formed log, mock-free run


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
