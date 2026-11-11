# Problem Formalization (C1)

This document fixes the definitions that the code implements. It is the
authority for terminology, the data model, the streaming/delta semantics, and
the correctness criterion. `src/streamdiam/data/schema.py` is the executable
encoding of §1–§3; the incremental-maintenance proofs (`docs/proofs.md`) build
on §4.

Notation: `[n] = {0, 1, …, n−1}`. `R^F` is the `F`-dimensional reals. `⊎` is
multiset union (it keeps duplicates).

---

## 1. The temporal transaction multigraph

A **temporal transaction multigraph** is an append-only sequence of directed,
timestamped, attributed edges

    G = (V, E),  E = ⟨ e_0, e_1, … , e_{m−1} ⟩,

where `V ⊆ N` is a set of **accounts** (nodes) and each **transaction** (edge)

    e = (eid, u, v, t, x)

has a unique edge id `eid ∈ N`, a source account `u ∈ V`, a destination
account `v ∈ V`, an integer timestamp `t ∈ Z`, and a feature vector
`x ∈ R^F`. `F` is fixed for a given graph.

`E` is a **multiset**: several transactions may share the same ordered pair
`(u, v)` and even the same timestamp `t` (e.g. multiple transfers mined in one
Ethereum block). This is what makes `G` a *multi*graph rather than a simple
graph, and it is a first-class property of the domain, not an anomaly to be
deduplicated away.

Self-loops (`u = v`) are permitted by the type but are rare in practice; the
data loader records them without special-casing.

### 1.1 Canonical event order

Edge ids are assigned in **non-decreasing arrival order**: if `e_i` is ingested
before `e_j` then `eid(e_i) < eid(e_j)`. Define the **canonical order** `≺` as
the lexicographic order on the pair `(t, eid)`:

    e ≺ e'   ⟺   ( t(e),  eid(e) )  <_lex  ( t(e'),  eid(e') ).

Because `eid` is globally unique, `≺` is a **total order** on `E`. It has two
properties the system depends on:

- **(deterministic tie-break)** transactions sharing a timestamp are ordered by
  arrival, so replay and all downstream computation are deterministic;
- **(causal soundness)** `≺` never places an event before one that physically
  preceded it, because `t` is non-decreasing within a source and `eid` is
  non-decreasing in arrival. This is the property that makes *no future
  leakage* (§3.2) a checkable invariant rather than a convention.

`schema.py` encodes `≺` as `TransactionRecord.__lt__` and
`TransactionBatch.is_time_sorted()`.

### 1.2 Per-account in/out sequences

For an account `w ∈ V`, its **incoming sequence** and **outgoing sequence** are
the `≺`-ordered subsequences

    S_in(w)  = ⟨ e ∈ E : v(e) = w ⟩  ordered by ≺,
    S_out(w) = ⟨ e ∈ E : u(e) = w ⟩  ordered by ≺.

These per-account ordered sequences — not the raw edge set — are the objects
the model consumes: the `Edge2Seq` encoder reads `S_in(w)` and `S_out(w)` to
produce a node representation for `w`, and the maintainer (C3) extends them one
element at a time as new transactions arrive. `Direction ∈ {IN, OUT}` in
`schema.py` selects between them.

---

## 2. Labels and the detection task

Each account `w` carries a ground-truth label `y(w) ∈ {licit, illicit}`, encoded
`{0, 1}`, with `−1` reserved for **unknown** (unlabeled). The detection task is
to assign each account a **risk score** `ŝ(w) ∈ [0, 1]` estimating
`P(y(w) = illicit)`.

Labels are typically **sparse** (most accounts unknown) and arrive with
**latency** (an account is confirmed illicit well after it acted). The schema
therefore carries labels as an optional, nullable channel defaulting to
unknown, and the evaluation protocol (C5) is responsible for respecting label
availability time; the data model never assumes a label is present.

---

## 3. The streaming / delta model

### 3.1 Graph deltas

The graph is revealed over time as a stream of **deltas**. A delta `ΔG` is a
finite, `≺`-ordered multiset of new transactions applied by append-only
multiset union:

    G ⊕ ΔG = ( V ∪ V(ΔG),  E ⊎ ΔG ).

In this work `ΔG` contains **only insertions** — on a public blockchain a
confirmed transaction is never retracted — so the maintainer never handles
edge deletion. (Account state can still *change*; what is immutable is the
transaction history.) A delta is represented by `TransactionBatch`; the empty
delta (`|ΔG| = 0`, a tick with no transactions) is valid and meaningful.

### 3.2 No future leakage

Let `t*` be the timestamp of the most recent processed transaction. Any
computation that produces a score at time `t*` may read only edges
`e` with `e ≺ (t*, +∞)`, i.e. only the prefix of `E` under `≺` up to the
current point. No edge with a strictly greater `(t, eid)` may influence a past
or present score. The canonical order of §1.1 makes this prefix well-defined
and the invariant testable (`tests/test_temporal_split.py`).

---

## 4. The inference view and the correctness criterion (forward reference)

Running the model from scratch after every delta is the semantics we must
match but cannot afford. Define the **inference view** `Φ(G)` as everything the
model computes on `G`:

    Φ(G) = ( { h_seq(w) }_{w∈V},  { m_mgd(w) }_{w∈V},  { ŝ(w) }_{w∈V} ),

the per-account `Edge2Seq` recurrent states `h_seq`, the multigraph-discrepancy
messages `m_mgd`, and the risk scores `ŝ`. The **maintainer** (C3) holds a
materialized view `Φ̃` and, on each delta `ΔG`, updates it incrementally to
`Φ̃'` instead of recomputing `Φ(G ⊕ ΔG)` wholesale.

**Correctness criterion.** For a fixed model `θ`, the maintainer is
**ε-correct** if after every delta

    ‖ Φ̃'  −  Φ(G ⊕ ΔG; θ) ‖_∞  ≤  ε,

with `ε = 0` (exactness) required on the components that are exactly
incrementalizable and a bounded `ε > 0` permitted only where a documented
pruning rule trades a provable error bound for work. `maintain/from_scratch.py`
computes `Φ(G ⊕ ΔG; θ)` and is the test oracle; `tests/test_maintain_equiv.py`
checks the criterion on real data. The per-component split (what is exact vs.
ε-bounded) and the bound on `ε` are established in `docs/proofs.md` once the
maintenance algorithm is defined.

This section is a forward reference only; the schema in §1–§3 does not depend
on it.
