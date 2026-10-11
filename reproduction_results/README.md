# CIFAR-10 Figure 5 reproduction

This directory contains the point-evaluation histories and plots from the
independent reproduction of LabelBench Figures 5(a) and 5(b).

- `cifar10_eval/bs_1000/`: 32 histories (8 strategies x 4 seeds), with 10
  acquisition points per run.
- `cifar10_eval/bs_200/`: 28 histories (7 strategies x 4 seeds), with 20
  acquisition points per run. Figure 5(b) does not include BAIT.
- `plots/*Test_Accuracy.pdf`: mean test accuracy with one standard error,
  using the cumulative-maximum transformation from the released plotting
  procedure.
- `plots/*Test_Accuracy_raw.pdf`: the same histories and uncertainty
  calculation without cumulative-maximum post-processing.

The histories were exported from the completed Weights & Biases evaluation
runs and intentionally kept separate from the authors' checked-in reference
results under `results/cifar10_eval/`.

Plots are generated with `results/plot_eval.py`. Run it from this directory so
that its relative input and output paths resolve to `cifar10_eval/` and
`plots/`. Use `--smoothing max` for the released plotting procedure or
`--smoothing none --output_suffix _raw` for the raw variant.
