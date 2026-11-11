# Claim → script map

Every empirical claim in the paper is reproduced by a committed command. Results
land in `results/<rq>/<dataset>/`; figures read only those files. Set
`STREAMDIAM_ELLIPTIC_ROOT` first.

| # | Claim | Reproduce with | Result file / check |
|---|---|---|---|
| C-EXACT | Incremental maintenance equals the from-scratch oracle (ε = float summation order) | `pytest tests/test_incremental_mgd.py tests/test_baselines.py -q` | `test_incremental_equals_from_scratch_every_tick` passes |
| C-RQ1a | Incremental and from-scratch reach **identical** prequential accuracy | `python -m streamdiam.experiments.rq1_maintenance_speed --root $ROOT --per-step 60` | `rq1_.../{from_scratch,incremental}.jsonl` summaries: equal `prequential.ap` |
| C-RQ1b | Incremental update latency ≪ from-scratch | same as C-RQ1a | compare `latency.mean_ms` across the two files |
| C-RQ2 | One harness yields a single F1/AP-over-time curve per policy (no contradiction) | `python -m streamdiam.experiments.rq2_accuracy_over_time --root $ROOT --per-step 60` | `rq2_.../{never,drift,periodic}.jsonl` timelines; `fig_accuracy_over_time` |
| C-RQ3 | Drift-triggering retrains far less often than a fixed schedule, at comparable accuracy | `python -m streamdiam.experiments.rq3_drift_trigger --root $ROOT --per-step 60` | `rq3_.../{drift,periodic}.jsonl`: `n_retrains`, `retrain_times`, `prequential.ap` |
| C-RQ7 | Incremental speedup **grows with graph size**; from-scratch becomes memory-infeasible | `python -m streamdiam.experiments.rq7_scalability --root $ROOT` | `rq7_scalability/elliptic/scaling.jsonl`; `fig_latency_scaling` |
| C-BASE | XGB-Graph is a real, runnable baseline through the same harness | `pytest tests/test_xgb_graph.py -q` | fits, beats base-rate AP, runs prequentially |
| C-NOLEAK | No future leakage: a tick is scored by a model trained strictly before it | `pytest tests/test_eval.py -q` | `test_no_future_leakage_invariant` passes |
| C-NOFAB | No figure can plot fabricated data | `pytest tests/test_no_synthetic_in_results.py -q` | static + functional guard passes |
| C-PARAM | Reported model parameter count is exact | `pytest tests/test_model.py -q` | closed-form parameter-count test passes |
| C-SPLIT | Temporal split has no future leakage | `pytest tests/test_data.py -q` | `temporal_split` leakage test passes |

## Representative committed scaling numbers (Elliptic, pure-Python reference, this artifact's hardware)

From `results/rq7_scalability/elliptic/scaling.jsonl`:

| N (accounts) | from-scratch ms/tick | incremental ms/tick | speedup |
|---|---|---|---|
| 3,247 | 134 | 20 | 6.7× |
| 6,363 | 518 | 37 | 14× |
| 10,638 | 1,821 | 92 | 20× |
| ~32,000 | out of memory | continues | — |

The trend, not any single row, is the claim: the speedup grows with N and
from-scratch stops being viable while the incremental maintainer does not.
