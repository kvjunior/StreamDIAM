# Drift-triggered retraining: design (C4)

This document states the design of the drift layer and the reasoning a reviewer
will probe. It is the empirical heart that answers R1 O1 / R2 O8 ("why retrain in
real time at all?"): the answer is *you mostly don't* -- retraining is decoupled,
drift-triggered, and off the critical path, and we measure how rarely it fires.

Code: `src/streamdiam/drift/{detectors,monitor,trigger}.py`. Tests: `tests/test_drift.py`.

## 1. What the drift layer is for

The inference view (C3) keeps risk scores fresh continuously without retraining.
But a *model* trained on an old regime can still become miscalibrated when the
data-generating process shifts (new laundering patterns, exchange behaviour,
market events). The drift layer decides **when** that has happened and a
background retrain is warranted. It never trains; it emits a signal and logs it.

## 2. Two branches, two clocks

Illicit-account labels arrive with large and variable latency: an account is
confirmed illicit long after it transacts. A purely supervised drift monitor
would therefore react late, or not at all on recent data. We split the monitor:

* **Unsupervised branch** -- consumes the risk score the model emits for *every*
  event, immediately, no label required. Its running mean feeds ADWIN, which
  flags a shift in the score distribution. This is the fast, always-available
  signal.
* **Supervised branch** -- consumes ground-truth labels *when they arrive*,
  computes a prequential (interleaved test-then-train) error with an optional
  fading factor (Gama et al. 2013), and feeds the 0/1 loss stream to DDM. This
  is the slower, authoritative signal.

The two are updated by separate calls (`observe_score`, `observe_label`) so the
evaluation harness can push scores every event and labels only when they
materialise -- the honest model of this domain's label latency, not an
assumption that labels are synchronous (which the old paper implicitly made).

The combined status is `DRIFT` if either branch drifts, else `WARNING` if the
supervised branch is in DDM's warning zone, else `STABLE`.

## 3. Detectors (faithful, cited, exact-first)

All three are implemented to their published definitions so the paper cites the
algorithm and the code is checkable against it:

| Detector | Input | Property | Reference |
|---|---|---|---|
| DDM | binary error | warning/drift at 2/3 sigma above the min error+std | Gama et al. 2004 |
| ADWIN | real value | adaptive window; cut when sub-window means differ beyond a variance-aware Hoeffding bound | Bifet & Gavaldà 2007 |
| Page-Hinkley | real value | cumulative deviation from running mean exceeds lambda | Page 1954; Mouss et al. 2004 |

ADWIN is implemented in its exact ("ADWIN0") form: the full window is kept and
every split point is tested against the variance-aware bound
`eps_cut = sqrt((2/m) v dd) + (2/3) dd/m`, `dd = ln(2|W|/delta)`. The
exponential-histogram is a memory optimisation of the *same* decision and is
layered separately if latency demands it -- the same exact-first discipline used
for incremental MGD (C3).

A detector that fires resets its own baseline so detection continues on the
post-change stream. (Implementation note, learned the hard way and now tested:
the reset must precede the status assignment, or the drift is swallowed.)

## 4. Trigger policy

`RetrainTrigger` turns per-step status into retrain decisions:

* **Decoupling.** It emits a `RetrainSignal` and appends a `TriggerEvent`; it
  does not train. The background trainer (C4, `retrain/`) consumes the signal.
  This is what keeps retraining off the critical path by construction.
* **Cooldown.** After firing, new firings are suppressed for `cooldown` steps.
  While a background retrain is in flight, re-firing would only thrash; the
  cooldown models retrain-and-swap latency and keeps the firing log honest.
* **Confirmation.** Only `DRIFT` fires; `WARNING` is a logged heads-up.

The firing log (`events`, `firing_count`) **is a result**: RQ3 reports how often
retraining actually fires and the accuracy cost of deferring between firings.

## 5. What is measured (RQ3), and the honest caveats

- Trigger frequency per dataset under the default policy, and its sensitivity to
  `delta` / `lambda` / `cooldown` (the sensitivity study).
- Accuracy-over-time between firings: the cost of *not* retraining, which
  quantifies how much the reframing ("maintain, rarely retrain") actually saves.

Caveats stated up front: drift detection on delayed labels is itself subject to
that delay; the unsupervised branch detects score-distribution shift, which is a
proxy for, not a guarantee of, accuracy drift. Both are reported, neither is
overclaimed.
