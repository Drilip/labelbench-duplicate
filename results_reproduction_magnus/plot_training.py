# Copy of the authors' results/plot_training.py: no LaTeX, fixed legend, saves a PNG only. Band = ± SE.
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

SMOOTHING = {"none": lambda a: a, "max": np.maximum.accumulate,
             "mov_avg": lambda a: np.convolve(np.concatenate([[a[0]] * 2, a]), np.ones(3) / 3, mode="valid")}

if __name__ == "__main__":
    plt.rcParams.update({"font.size": 16, "figure.autolayout": True})
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--batch_size", type=int, help="Labels collected per round.")
    parser.add_argument("--embed_model", type=str, help="Embedding model name in the result path.")
    parser.add_argument("--classifier_model", type=str, help="Classifier model name in the result path.")
    parser.add_argument("--trainer_config", type=str, help="Trainer name in the result path.")
    parser.add_argument("--metrics", nargs="+", help="Metric names for plotting.")
    parser.add_argument("--strategies", nargs="+", help="Strategy names for plotting.")
    parser.add_argument("--colors", nargs="+", help="Color for plots.")
    parser.add_argument("--smoothing", choices=list(SMOOTHING), default="none")
    parser.add_argument("--y_lim", type=float, nargs="+", default=None)
    args = parser.parse_args()

    name = "%s_%d_%s_%s_%s" % (args.dataset, args.batch_size, args.embed_model, args.classifier_model, args.trainer_config)
    path_name = "./%s/bs_%d/embed_%s/model_%s/trainer_%s" % (
        args.dataset, args.batch_size, args.embed_model, args.classifier_model, args.trainer_config)
    alg_dict = {}
    for file_name in sorted(f for f in os.listdir(path_name) if f.endswith(".pkl")):
        with open(os.path.join(path_name, file_name), "rb") as file:
            alg_dict.setdefault(file_name.split("_")[0], []).append(pickle.load(file)[0])
    num_labeled = alg_dict[args.strategies[0]][0]["Num Labeled"]

    for metric in args.metrics:
        y_lim = args.y_lim
        fig, ax = plt.subplots(1)
        for strategy, color in zip(args.strategies, args.colors):
            perfs = np.array([SMOOTHING[args.smoothing](np.array(run[metric])) for run in alg_dict[strategy]])
            mean, se = perfs.mean(axis=0), perfs.std(ddof=1, axis=0) / np.sqrt(len(perfs))
            ax.plot(num_labeled, mean, linewidth=3, color=color, label=strategy.upper())
            ax.fill_between(num_labeled, mean - se, mean + se, alpha=.3, color=color)
            if y_lim is None and strategy == "random":
                y_lim = [perfs[0][0], None]
        ax.set_xlabel("Number of Labels")
        ax.set_ylabel(metric)
        ax.legend()
        if y_lim:
            ax.set_ylim(*y_lim)
        ax.grid(True, linestyle="--")
        os.makedirs("plots", exist_ok=True)
        fig.savefig("plots/%s_%s.png" % (name, metric.replace(" ", "_")), dpi=300)
        plt.close(fig)
