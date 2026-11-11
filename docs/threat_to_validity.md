# Threats to validity

Stated plainly, because the previous submission's problems were exactly the ones
hidden here. Each threat names what we do about it and what remains a genuine
limitation.

## Class imbalance (R4 O6)

Illicit accounts are 2-10% of labelled nodes, and the ratio shifts over time
(Elliptic: ~0.7% in the opening timesteps, >8% in the middle). Accuracy is
therefore meaningless; we report F1 for the illicit class, ROC-AUC, and **average
precision (AP) as the headline**, and training class-balances with
`pos_weight = n_neg/n_pos`. Degenerate windows (no positives) are reported as
`nan`, not silently dropped or imputed. Remaining limitation: AP on a small
subsample is low for *every* method; headline accuracy requires the full graph
with proper training.

## Label latency (R3 O5)

In this domain labels arrive long after the activity they describe. The monitor
reflects this: the unsupervised branch acts on scores immediately, the supervised
branch only when a label materialises, on separate clocks. The strict prequential
metric scores each account at first appearance, before it can enter any training
window. Remaining limitation: supervised drift detection is itself delayed by the
label delay; we lean on the unsupervised branch for early signal and do not claim
supervised drift is timely.

## Drift assumptions (R1 O1 / R2 O8)

The reframing — maintain continuously, retrain rarely — assumes drift is
detectable and infrequent. We *measure* the trigger frequency rather than assume
it, and contrast drift-triggering against a periodic schedule. Remaining
limitation: a detector has false positives and negatives; thresholds are defaults
exposed for a sensitivity sweep (RQ5, to run), not tuned per-dataset.

## Scale of the reference implementation

The store and maintainers are pure Python, chosen for correctness and
portability. They demonstrate the scaling *trend* (RQ7) — the incremental
advantage grows with graph size and from-scratch recompute becomes
memory-infeasible — but are not the vehicle for full 203k-account Elliptic or the
DIAM-native Ethereum/Bitcoin graphs. Those absolute headline numbers need the
native hot path (future work). We report what the reference measures and never
extrapolate numbers we did not run. The latency *ratio* and *trend* are
machine-robust; absolute wall-clock values are machine-dependent and reported as
such (`reproducibility/hardware.md`).

## Dataset coverage

Only Elliptic is both real and temporally structured in the current runs. The
DIAM-native account multigraphs (Ethereum/Bitcoin) targeted for the paper's
headline detection numbers must be obtained and run separately; until then,
detection-quality claims are scoped to Elliptic.

## Baseline completeness

Maintenance strategies (from-scratch, incremental, frozen), retrain policies
(drift, periodic, never), and the XGB-Graph detector are real and run through the
one harness. TGN / APAN / Spade are not yet wired to their upstream code and
appear in **no** results table — never as placeholder numbers. Comparative claims
against them are deferred until their real implementations run here.

## Exactness scope of the correctness theorem

Incremental maintenance is *exact* for the recurrent (Edge2Seq) and message
(MGD) computations up to floating-point summation order; the proofs state this
precisely and do not claim bit-equality or O(1)/O(log n) where neighbourhood
expansion makes the work O(affected-set). The threshold-pruned (bounded-ε) MGD
variant is a separate, optional layer on top of the exact path, kept out of the
oracle.
