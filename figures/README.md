# Figures

**Every figure traces to a committed results file.** Figures in this package are
pure consumers: they read numbers from `results/<rq>/<dataset>/*.jsonl` through
`figures._io` and plot them. They contain no model, no random number generation,
and no experiment run.

This is the structural fix for the original submission's fabricated figures. It
is enforced by `tests/test_no_synthetic_in_results.py`, which:

1. scans every figure module for data-generation calls (`np.random`, `random.`,
   `torch.rand*`, `make_classification`, "synthetic", …) and fails if any
   appears;
2. forbids figure modules from importing the run/model machinery (so a figure
   cannot generate its own data); and
3. runs a real experiment, writes its results, and asserts the figure plots back
   exactly the committed numbers.

To regenerate a figure, first produce the results with an experiment driver, then
point the figure at the committed file:

    python -m streamdiam.experiments.rq2_accuracy_over_time --root <elliptic_dir>
    python -m streamdiam.figures.fig_accuracy_over_time \
        results/rq2_accuracy_over_time/elliptic/drift.jsonl --out fig.png
