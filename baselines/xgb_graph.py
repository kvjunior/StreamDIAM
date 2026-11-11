"""XGB-Graph baseline: gradient-boosted trees on graph-augmented features.

GADBench found tree ensembles to be strong, often state-of-the-art, baselines on
graph anomaly tasks, so a credible paper must compare against one -- for real, not
mocked. This is a self-contained detector (no DIAM, no shared weights) that plugs
into the *same* streaming harness through the ``StreamingMaintainer`` interface,
so it is measured by the identical prequential protocol as everything else.

Per-account features (streaming-maintainable):

* **own** -- the account's own feature vector, the running mean of its outgoing
  edges' attributes (in Elliptic an edge carries its source's features, so an
  account's out-edges carry its own), extended in O(delta) per tick;
* **neighbour** -- the mean of its in/out neighbours' own-features, a one-hop
  aggregation recomputed from the store.

The classifier is a histogram gradient-boosted tree, refit on the drift/periodic
trigger over the accounts seen so far, with class-balancing and optional recency
sample weights. It has its own ``retrain``/``version``; the run helper wires it to
the protocol exactly like the DIAM path wires the background trainer.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import HistGradientBoostingClassifier

from streamdiam.data.schema import LABEL_ILLICIT, LABEL_LICIT, TransactionBatch
from streamdiam.eval.protocol import RunResult, StreamingMaintainer
from streamdiam.retrain.recency_weight import RecencyWeighting
from streamdiam.store.interface import TemporalMultigraphStore

__all__ = ["XGBGraphMaintainer", "run_xgb_graph"]


class XGBGraphMaintainer(StreamingMaintainer):
    """Tree-ensemble detector on own + neighbour-aggregated account features."""

    def __init__(
        self,
        store: TemporalMultigraphStore,
        universe_size: int,
        labels: npt.NDArray[np.integer],
        *,
        recency: RecencyWeighting | None = None,
        seed: int = 0,
        max_iter: int = 100,
    ) -> None:
        self._store = store
        self._n = int(universe_size)
        self._f = store.num_features
        self._labels = np.asarray(labels)
        self._recency = recency
        self._seed = seed
        self._max_iter = max_iter
        self._own_sum = np.zeros((self._n, self._f), dtype=np.float64)
        self._own_cnt = np.zeros(self._n, dtype=np.float64)
        self._clf: HistGradientBoostingClassifier | None = None
        self._version = 0
        self._scores_cache = np.zeros(self._n, dtype=np.float64)
        self._dirty = True

    @property
    def name(self) -> str:
        return "xgb_graph"

    @property
    def version(self) -> int:
        return self._version

    # -- features ---------------------------------------------------------- #
    def _own(self) -> npt.NDArray[np.float64]:
        cnt = np.maximum(self._own_cnt, 1.0)[:, None]
        return self._own_sum / cnt

    def _features(self) -> npt.NDArray[np.float64]:
        own = self._own()
        nbr_sum = np.zeros((self._n, self._f), dtype=np.float64)
        nbr_cnt = np.zeros(self._n, dtype=np.float64)
        edges = self._store.all_edges()
        if len(edges) > 0:
            src = np.asarray(edges.src, dtype=np.int64)
            dst = np.asarray(edges.dst, dtype=np.int64)
            np.add.at(nbr_sum, dst, own[src])  # u is an in-neighbour of v
            np.add.at(nbr_cnt, dst, 1.0)
            np.add.at(nbr_sum, src, own[dst])  # v is an out-neighbour of u
            np.add.at(nbr_cnt, src, 1.0)
        nbr = nbr_sum / np.maximum(nbr_cnt, 1.0)[:, None]
        return np.concatenate([own, nbr], axis=1)

    # -- StreamingMaintainer ---------------------------------------------- #
    def rebuild(self) -> None:
        self._own_sum[:] = 0.0
        self._own_cnt[:] = 0.0
        edges = self._store.all_edges()
        if len(edges) > 0:
            attrs = np.asarray(edges.attr, dtype=np.float64)
            src = np.asarray(edges.src, dtype=np.int64)
            np.add.at(self._own_sum, src, attrs)
            np.add.at(self._own_cnt, src, 1.0)
        self._dirty = True

    def ingest(self, delta: TransactionBatch) -> None:
        attrs = np.asarray(delta.attr, dtype=np.float64)
        src = np.asarray(delta.src, dtype=np.int64)
        np.add.at(self._own_sum, src, attrs)
        np.add.at(self._own_cnt, src, 1.0)
        self._dirty = True

    def scores(self) -> npt.NDArray[np.float64]:
        if self._clf is None:
            return np.zeros(self._n, dtype=np.float64)
        if self._dirty:
            proba = self._clf.predict_proba(self._features())[:, 1]
            self._scores_cache = proba.astype(np.float64)
            self._dirty = False
        return self._scores_cache

    # -- its own retraining ----------------------------------------------- #
    def _last_activity(self) -> npt.NDArray[np.int64]:
        last = np.full(self._n, -1, dtype=np.int64)
        edges = self._store.all_edges()
        if len(edges) > 0:
            t = np.asarray(edges.timestamp, dtype=np.int64)
            np.maximum.at(last, np.asarray(edges.src, dtype=np.int64), t)
            np.maximum.at(last, np.asarray(edges.dst, dtype=np.int64), t)
        return last

    def retrain(self, seen_mask: npt.NDArray[np.bool_], ref_time: float) -> int | None:
        """Refit the tree over seen+labelled accounts; return the new version.

        Returns ``None`` (no version bump) if the window has fewer than two
        classes to learn from -- honest: there is nothing to fit yet.
        """
        lab = self._labels
        labelled = ((lab == LABEL_ILLICIT) | (lab == LABEL_LICIT)) & np.asarray(
            seen_mask, dtype=bool
        )
        y = (lab == LABEL_ILLICIT).astype(int)
        if np.unique(y[labelled]).size < 2:
            return None

        x = self._features()[labelled]
        yt = y[labelled]
        n_pos = int(yt.sum())
        n_neg = int(yt.size - n_pos)
        w = np.where(yt == 1, (n_neg / max(n_pos, 1)), 1.0).astype(np.float64)
        if self._recency is not None:
            times = self._last_activity()[labelled].astype(np.float64)
            w = w * self._recency.weights(times, ref_time, normalize=True)

        clf = HistGradientBoostingClassifier(
            max_iter=self._max_iter, random_state=self._seed
        )
        clf.fit(x, yt, sample_weight=w)
        self._clf = clf
        self._version += 1
        self._dirty = True
        return self._version


def run_xgb_graph(
    edges: TransactionBatch,
    labels: npt.NDArray[np.integer],
    universe_size: int,
    num_features: int,
    *,
    warmup_fraction: float = 0.3,
    retrain_policy: str = "drift",
    retrain_period: int = 10,
    cooldown: int = 5,
    recency: RecencyWeighting | None = None,
    threshold: float = 0.5,
    seed: int = 0,
) -> RunResult:
    """Run the XGB-Graph baseline through the prequential harness."""
    # Imported here to avoid any import-order coupling with the eval package.
    from streamdiam.data.stream_replay import replay_by_time
    from streamdiam.drift import DriftMonitor, PeriodicTrigger, RetrainTrigger, Trigger
    from streamdiam.eval.protocol import PrequentialProtocol
    from streamdiam.store.reference_py import ReferencePyStore

    n = int(universe_size)
    labels = np.asarray(labels)
    store = ReferencePyStore(num_features)
    maintainer = XGBGraphMaintainer(store, n, labels, recency=recency, seed=seed)

    initial_seen = np.zeros(n, dtype=bool)
    ts = np.asarray(edges.timestamp)
    distinct = np.unique(ts)
    stream_edges = edges
    if warmup_fraction > 0.0 and distinct.size > 1:
        cut = max(1, int(distinct.size * warmup_fraction))
        t_cut = int(distinct[cut - 1])
        split = int(np.searchsorted(ts, t_cut, side="right"))
        warm = edges[:split]
        assert isinstance(warm, TransactionBatch)
        rest = edges[split:]
        assert isinstance(rest, TransactionBatch)
        stream_edges = rest
        if len(warm) > 0:
            store.insert(warm)
            maintainer.rebuild()
            warm_acc = np.unique(np.concatenate([warm.src, warm.dst]))
            initial_seen[warm_acc] = True
            maintainer.retrain(initial_seen, float(t_cut))

    monitor: DriftMonitor | None = None
    trigger: Trigger | None = None
    if retrain_policy == "drift":
        monitor = DriftMonitor()
        trigger = RetrainTrigger(monitor, cooldown=cooldown)
    elif retrain_policy == "periodic":
        trigger = PeriodicTrigger(period=retrain_period)
    elif retrain_policy != "never":
        raise ValueError(f"unknown retrain_policy: {retrain_policy!r}")

    def retrain_hook(ref_time: int, seen_mask: npt.NDArray[np.bool_]) -> int | None:
        return maintainer.retrain(seen_mask, float(ref_time))

    protocol = PrequentialProtocol(
        store, n, maintainer, labels,
        monitor=monitor, trigger=trigger, retrain_hook=retrain_hook,
        version_provider=lambda: maintainer.version,
        threshold=threshold, initial_seen=initial_seen,
    )
    return protocol.run(replay_by_time(stream_edges))
