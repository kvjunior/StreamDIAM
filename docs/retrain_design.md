# Decoupled retraining: design (C4, part 2)

Companion to `drift_design.md`. The drift layer decides *when*; this layer does
the *retraining*, off the critical path. Code:
`src/streamdiam/retrain/{recency_weight,background_trainer,hot_swap}.py`.
Tests: `tests/test_retrain.py`.

## 1. Decoupling, realised in code

When the trigger fires, `BackgroundTrainer.train` fine-tunes a **deep copy** of
the live model and returns a `ModelCheckpoint`; the live model is never mutated.
So the inference view (C3) keeps serving current scores for the whole duration of
the fine-tune. The paper's claim "retraining is off the critical path" is not an
assertion about an architecture diagram -- it is a property of this code, tested
by `test_trainer_does_not_mutate_live_model`.

The finished checkpoint is installed by `HotSwapRegistry.swap`, which builds the
new version beside the old one and publishes it in a single reference
assignment -- atomic under CPython's single writer, so the maintainer never reads
a half-loaded model. One previous version is retained for rollback after a bad
fine-tune.

## 2. The recency curriculum (answers R4 O7)

The old paper's training curriculum was unclear (R4 O7). Here it is one
documented knob: an exponential recency weight

    w_i = exp(-lambda * (t_ref - t_i)),   lambda = ln 2 / half_life,

applied per account by its most recent activity time. The **half-life** -- the
age at which an account's weight halves -- is stated in the dataset's own time
units and swept in the sensitivity study. `lambda = 0` (uniform) is the ablation
baseline. Future-dated samples are clamped to zero age; a retrain never sees the
future.

## 3. Objective: masked, class-weighted, recency-weighted BCE

- **Masked.** Loss is computed over **labelled** accounts only (Elliptic is ~23%
  labelled). Unlabelled accounts are masked out, never pseudo-labelled.
- **Class-weighted.** `pos_weight = n_neg / n_pos` (auto) counters the 2-10%
  illicit imbalance (R4 O6).
- **Recency-weighted.** per §2.

The forward pass is the model's own (`DIAM.forward` over the store snapshot) --
the same path the maintainer and oracle use -- so there is no train/serve skew:
the weights produced are exactly the weights served.

## 4. Reproducibility

Each fine-tune seeds every RNG via `utils/seeding.seed_everything` and records its
config and loss history in the checkpoint. Re-running a fine-tune reproduces it
(R1 O18).

## 5. What is validated on real data

On a real Elliptic subgraph (remapped to a compact labelled universe):
`test_trainer_learns_on_real_elliptic` checks that training-set average precision
rises well above the random baseline -- the blueprint's "reproduce DIAM / sanity
anchor" step, proving the training loop actually learns on real labels. The full
`train -> swap -> rebuild` loop is checked end-to-end: after swapping the
fine-tuned checkpoint into the registry and rebuilding the view, the served
scores' average precision improves.

Honest scope: these tests establish that the machinery learns and installs
correctly. Generalisation (held-out, over-time accuracy) is the eval harness's
job (C5, RQ2/RQ3), not claimed here.
