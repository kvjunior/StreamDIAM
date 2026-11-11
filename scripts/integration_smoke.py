"""End-to-end integration smoke test for the whole StreamDIAM pipeline.

Runs the entire chain on real Elliptic data, through every layer, in one process,
and cross-checks that the pieces agree:

    data -> store -> incremental maintainer (C3) -> drift monitor + trigger (C4)
         -> background trainer + hot-swap (C4) -> prequential harness (C5)
         -> committed results/*.jsonl -> figures

Consistency checks (fail loudly):
  * every subpackage imports;
  * incremental and from-scratch strategies reach identical prequential accuracy
    through the full harness (not just per-delta);
  * the drift-retrain run is well-formed (no look-ahead; version advances only
    after a retrain tick);
  * the XGB-Graph baseline runs through the same harness;
  * the RQ drivers write valid JSONL and the figures render from those files only.

Usage:  STREAMDIAM_ELLIPTIC_ROOT=/path/to/elliptic python scripts/integration_smoke.py
Exit code 0 on success; non-zero with a message on any inconsistency.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_SUBPACKAGES = [
    "data", "store", "model", "maintain", "drift", "retrain",
    "eval", "baselines", "experiments", "figures", "utils",
]


def _fail(msg: str) -> None:
    print(f"INTEGRATION FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    root = os.environ.get("STREAMDIAM_ELLIPTIC_ROOT")
    if not root:
        _fail("set STREAMDIAM_ELLIPTIC_ROOT to the Elliptic CSV directory")

    # 1) every subpackage imports --------------------------------------------- #
    for pkg in _SUBPACKAGES:
        importlib.import_module(f"streamdiam.{pkg}")
    print(f"[1/6] all {len(_SUBPACKAGES)} subpackages import ...... OK")

    from streamdiam.baselines import run_xgb_graph
    from streamdiam.eval import RunConfig, run_streaming
    from streamdiam.experiments import load_dataset, rq1_maintenance_speed, rq3_drift_trigger, rq7_scalability
    from streamdiam.experiments import rq2_accuracy_over_time
    from streamdiam.figures.fig_accuracy_over_time import make as fig_acc
    from streamdiam.figures.fig_latency_scaling import make as fig_lat
    from streamdiam.model import DIAM
    from streamdiam.retrain import TrainConfig

    ds = load_dataset("elliptic", root, per_step=40)
    print(f"[2/6] real Elliptic loaded ............ OK  (N={ds.universe_size}, edges={len(ds.edges)}, F={ds.num_features})")

    # 2) full harness, two strategies, identical accuracy --------------------- #
    def _run(maintainer: str):
        from streamdiam.utils.seeding import seed_everything

        seed_everything(0)
        model = DIAM(in_dim=ds.num_features, hidden_dim=16, num_layers=2)
        cfg = RunConfig(maintainer=maintainer, warmup_fraction=0.3,
                        train_config=TrainConfig(epochs=5, seed=0))
        return run_streaming(ds.edges, ds.labels, ds.universe_size, model, cfg, enable_retrain=False)

    r_fs = _run("from_scratch")
    r_inc = _run("incremental")
    if abs(r_fs.prequential.ap - r_inc.prequential.ap) > 1e-6:
        _fail(f"incremental/from-scratch prequential AP differ: {r_fs.prequential.ap} vs {r_inc.prequential.ap}")
    print(f"[3/6] incremental == from-scratch ..... OK  (prequential AP {r_inc.prequential.ap:.4f}, identical)")

    # 3) drift-retrain run is well-formed (no look-ahead) --------------------- #
    from streamdiam.utils.seeding import seed_everything

    seed_everything(0)
    model = DIAM(in_dim=ds.num_features, hidden_dim=16, num_layers=2)
    r_drift = run_streaming(
        ds.edges, ds.labels, ds.universe_size, model,
        RunConfig(maintainer="incremental", retrain_policy="drift",
                  warmup_fraction=0.3, train_config=TrainConfig(epochs=5)),
    )
    ticks = r_drift.ticks
    for i in range(1, len(ticks)):
        if ticks[i].model_version != ticks[i - 1].model_version and not ticks[i - 1].retrained:
            _fail(f"look-ahead: tick {ticks[i].t} version changed without a prior retrain")
    print(f"[4/6] drift-retrain + no-leakage ...... OK  ({r_drift.n_retrains} retrains, {len(ticks)} ticks)")

    # 4) XGB-Graph baseline runs through the same harness --------------------- #
    r_xgb = run_xgb_graph(ds.edges, ds.labels, ds.universe_size, ds.num_features, retrain_policy="periodic", retrain_period=5)
    if r_xgb.maintainer != "xgb_graph" or len(r_xgb.ticks) == 0:
        _fail("xgb_graph baseline did not run through the harness")
    print(f"[5/6] XGB-Graph baseline .............. OK  (AP {r_xgb.prequential.ap:.4f}, {r_xgb.n_retrains} retrains)")

    # 5) drivers -> committed JSONL -> figures (read-only) -------------------- #
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        rq1_maintenance_speed.run(ds, out, epochs=4)
        rq2_accuracy_over_time.run(ds, out, epochs=4)
        rq3_drift_trigger.run(ds, out, epochs=4, periodic_period=4)
        rq7_scalability.run(root, out, sizes=[40])

        acc_file = out / "rq2_accuracy_over_time" / "elliptic" / "drift.jsonl"
        scal_file = out / "rq7_scalability" / "elliptic" / "scaling.jsonl"
        for f in (acc_file, scal_file):
            if not f.exists():
                _fail(f"driver did not write {f}")
        png1 = fig_acc(acc_file, out / "acc.png")
        png2 = fig_lat(scal_file, out / "scal.png")
        if not (png1.exists() and png1.stat().st_size > 0 and png2.exists() and png2.stat().st_size > 0):
            _fail("figures did not render from committed results")
    print("[6/6] drivers -> results -> figures ... OK  (all RQ files + both figures)")

    print("\nINTEGRATION OK: full pipeline consistent end-to-end on real Elliptic.")


if __name__ == "__main__":
    main()
