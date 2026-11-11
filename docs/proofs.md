# Correctness and complexity of incremental maintenance (C3)

This document states, for each incrementally-maintained component, what the
maintainer keeps, why its update is cheap, and why the maintained value equals
(or stays within ε of) the from-scratch oracle `maintain/from_scratch.py`. Each
theorem is paired with the code that implements it and an equivalence test that
checks it on real data.

Notation follows `docs/problem_formalization.md`. The model parameters θ are
fixed during maintenance (retraining is the separate, decoupled path of C4).

---

## 1. Edge2Seq is exactly maintainable in O(c) per edge

### 1.1 Streaming formulation: growing window, not sliding window

In the static setting DIAM caps each account's sequence to its `T_max` most
recent edges. A *sliding* window of fixed size is **not** O(1)-maintainable:
when an edge leaves the window the running max-pool can lose its argmax and the
GRU's starting state changes, forcing an O(T_max) recompute.

In the streaming setting we instead maintain the **growing window** — the full
history of each account's in/out sequence. This is a deliberate modeling choice,
not an approximation: the `T_max` cap existed only to bound static computation,
and streaming maintenance removes that need because the maintained state is O(c)
regardless of history length. The growing window uses at least as much history
as the cap, and (Theorem 1) is maintainable exactly and in O(c) per edge.

The from-scratch oracle is run with `t_max` ≥ the longest sequence, so it
computes the same growing-window Edge2Seq; that is what the equivalence test
compares against.

### 1.2 What is maintained

For each account `w` and direction `dir ∈ {in, out}` with a non-empty sequence
of edge features `x_1, …, x_k` in canonical order, define

    z_i   = Linear_dir(x_i)                       (per-edge linear map)
    h_i   = GRU_dir(z_i, h_{i-1}),  h_0 = 0       (recurrent state)
    m     = elementwise-max_{i=1..k} h_i          (max-pool over time)

Edge2Seq's code for `w` is `code(w) = [ m_in(w) ; m_out(w) ] ∈ R^{2c}`. The
maintainer stores, per `(w, dir)`, only the pair `(h_k, m)` and the count `k`
— O(c) state. An account with `k = 0` in a direction contributes the fixed
*self-loop code* `s_dir = h_1` obtained by encoding a single zero row (exactly
what the oracle's self-loop padding produces), so empty directions need no
stored state.

### 1.3 Update

When a new edge with features `x` is appended to `(w, dir)`:

    z        = Linear_dir(x)
    h_{k+1}  = GRU_dir(z, h_k)       (h_k := 0 when k = 0)
    m'       = h_{k+1}               if k = 0
             = max(m, h_{k+1})       if k ≥ 1

Each step is one linear map, one GRU cell application, and one element-wise max
— all O(c), independent of `k`. The edge `u → v` triggers exactly two such
updates: `(u, out)` and `(v, in)`.

### 1.4 Theorem 1 (exactness)

*After processing an account's edges in canonical order, the maintained
`(h_k, m)` equal the last GRU state and the time max-pool computed by a
from-scratch growing-window Edge2Seq over the same edges; hence the maintained
`code(w)` equals the oracle's, exactly (up to floating-point reproducibility of
identical operations).*

**Proof.** The recurrence for `h_i` is identical to the one the maintainer
applies step by step, over the same inputs in the same order, using the same
GRU weights, so `h_k` matches. The max-pool `m = max_i h_i` is an element-wise
maximum over a set; since element-wise max is associative and commutative,
folding it one element at a time (`m' = max(m, h_{k+1})`) yields the same result
as reducing the whole set, so `m` matches. Concatenation of the two directional
max-pools gives `code(w)`. For `k = 0`, both the maintainer and the oracle encode
one zero row, giving the same `s_dir`. ∎

Because the per-account subsequence of a canonically-ordered stream is itself in
canonical order, feeding the maintainer a canonically-ordered delta stream
reproduces the oracle's per-account order, which the proof requires.

### 1.5 Code and test

Implemented in `maintain/incremental_edge2seq.py`; `tests/test_incremental.py`
replays a stream through the maintainer and asserts the codes equal
`FromScratchMaintainer` (full-history) within 1e-5 on constructed streams and on
a real Elliptic subgraph. ε = 0 for this component.

---

## 2. MGD is exactly maintainable on the L-hop closure of a delta

### 2.1 What is maintained

The maintainer stores, for the current graph, the layer-0 codes `h^0 = codes`
(kept exact by §1) and each MGD layer's output `h^1, …, h^L ∈ R^{N×c}`. For a
single layer `l` with input `g = h^{l-1}` the model computes, for every node `v`,

    z_v      = W2 g_v + b2
    r_in(v)  = Σ_{u : u→v} W3_in ( z_u ‖ (z_v − z_u) )      (multiset sum)
    r_out(v) = Σ_{u : v→u} W3_out( z_u ‖ (z_v − z_u) )
    h^l_v    = attention( z_v, r_in(v), r_out(v) ).

### 2.2 Locality lemma

*If the layer input changes only on a set `C` (i.e. `g_w` differs for `w ∈ C`
and is unchanged elsewhere), then the layer output `h^l_v` can change only for
`v ∈ C ∪ N(v)∩C ≠ ∅`, i.e. for `v ∈ C ∪ N(C)`.*

**Proof.** `h^l_v` is a function of `z_v` and of `{z_u : u ∈ N(v)}` only (the
message sums range over `v`'s neighbors; `z_x = W2 g_x + b2` depends only on
`g_x`). If `v ∉ C` and no neighbor of `v` is in `C`, then `z_v` and every
`z_u, u∈N(v)` are unchanged, so `r_in(v), r_out(v)` and hence `h^l_v` are
unchanged. Contrapositive: a change at `v` requires `v ∈ C` or `N(v)∩C ≠ ∅`,
i.e. `v ∈ C ∪ N(C)`. ∎

### 2.3 Theorem 2 (exact affected set and its bound)

*Let a delta insert edges with endpoint set `P`, changing the Edge2Seq codes
exactly on `S_0 ⊆ P` (§1). Define `S_l = S_{l-1} ∪ N(S_{l-1})` on the
post-insert graph. Then recomputing, at each layer `l`, exactly the nodes in
`R_l = S_{l-1} ∪ N(S_{l-1})` from the current neighbor representations, and
leaving all other nodes unchanged, reproduces the from-scratch MGD view exactly.
Moreover the total touched set `S_L` is contained in the `L`-hop neighborhood of
`P`.*

**Proof.** By induction on `l`. The layer-0 input changes exactly on `S_0`
(§1, Theorem 1). Assume the layer-`(l−1)` output changes exactly on some
`C ⊆ S_{l-1}`. By the locality lemma the layer-`l` output can change only on
`C ∪ N(C) ⊆ S_{l-1} ∪ N(S_{l-1}) = R_l`, so every node outside `R_l` keeps its
stored (correct) value, and every node inside `R_l` is recomputed from the
current `z` of its multiset neighborhood — the same quantity the oracle
computes — hence equals the oracle. The touched set after layer `l` is
`⊆ R_l = S_l`. Unrolling, `S_L = S_0 ∪ N(S_0) ∪ … ∪ N^L(S_0)` is the `L`-hop
closure of `S_0 ⊆ P`. ∎

Consequences: per delta the work is `O( Σ_{v∈S_L} deg(v) · c + |N(S_L)| · c² )`
— proportional to the touched neighborhood, not the graph. On a power-law graph
a delta incident to a hub makes `S_L` large; that is the neighborhood-explosion
case §2.4 addresses.

### 2.4 Pruning (bounded ε, bounded work) — layered on top, added separately

Exact maintenance (ε = 0 in real arithmetic) is the foundation above. For hubs,
a threshold rule stops propagating a change whose magnitude `‖Δh^l_v‖` falls
below `δ`, bounding `|S_l|` at the cost of a controlled error. The resulting
bound `‖Φ̃ − Φ‖_∞ ≤ ε(δ, L, ‖W‖)` and the work/accuracy trade-off will be stated
here with that extension; it does not alter §2.3, which the pruned version falls
back to at `δ = 0`.

### 2.5 Code and test

Implemented in `maintain/incremental_mgd.py` (affected set in
`maintain/affected_set.py`). `tests/test_incremental_mgd.py` streams a graph
through the full incremental stack (§1 Edge2Seq + §2 MGD) and asserts the
maintained representations and scores match `FromScratchMaintainer` within 1e-4
(the gap is floating-point summation order only), on constructed graphs and a
real Elliptic subgraph.
