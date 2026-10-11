# Numbers for the Fig. 1a reproduction and the random-init ablation. Run from this folder.
import pickle

import numpy as np

OURS = "cifar10/bs_1000/embed_none/model_clip_ViTB32_pretrained/trainer_flexmatch"
RANDINIT = "cifar10/bs_1000/embed_none/model_clip_ViTB32_randinit/trainer_flexmatch"
AUTHORS = "../results/cifar10/bs_1000/embed_none/model_clip_ViTB32/trainer_flexmatch"
STRATEGIES = ["random", "confidence", "entropy", "margin", "coreset", "badge"]
SEEDS = [1234, 3334155, 6667077, 9999999]

# Table 6 of the paper, fine-tune column, 10000 labels: (mean, std over 4 seeds) in %.
PAPER = {"random": (97.59, .22), "confidence": (97.84, .07), "entropy": (97.89, .08),
         "margin": (97.97, .12), "coreset": (97.79, .06), "badge": (97.95, .08)}


def acc(folder, strategy, n, seeds=SEEDS):
    # test accuracy in % after n labels, one value per seed
    values = []
    for seed in seeds:
        run, _ = pickle.load(open(f"{folder}/{strategy}_sampling_{seed}.pkl", "rb"))
        i = [abs(k - n) <= 10 for k in run["Num Labeled"]].index(True)  # the authors' BADGE runs label 1001 per round
        values.append(run["Test Accuracy"][i] * 100)
    return np.array(values)


def show(v):
    return f"{v.mean():.2f} ± {v.std(ddof=1) / np.sqrt(len(v)):.2f}"


print("Table I: test accuracy at 10000 labels, mean ± SE over 4 seeds")
for s in STRATEGIES:
    print(f"  {s:<11} ours {show(acc(OURS, s, 10000))}   paper {PAPER[s][0]:.2f} ± {PAPER[s][1] / 2:.2f}")

print("\nTable II: test accuracy at 2000 and 3000 labels, mean ± SE over 4 seeds")
for s in STRATEGIES:
    print(f"  {s:<11} ours 2000: {show(acc(OURS, s, 2000))}   authors 2000: {show(acc(AUTHORS, s, 2000))}"
          f"   ours 3000: {show(acc(OURS, s, 3000))}   authors 3000: {show(acc(AUTHORS, s, 3000))}")

print("\nLabel savings: labels needed to reach Random's accuracy at 10000 labels (interpolated)")
for who, folder in (("ours", OURS), ("authors", AUTHORS)):
    target = acc(folder, "random", 10000).mean()
    print(f"  {who}: Random at 10000 labels = {target:.2f}")
    for s in STRATEGIES[1:]:
        a2, a3 = acc(folder, s, 2000).mean(), acc(folder, s, 3000).mean()
        if a2 >= target:
            print(f"    {s:<11} reached by 2000 labels (saving >= 5.0x)")
        elif a3 < target:
            print(f"    {s:<11} not reached by 3000 labels")
        else:
            n = 2000 + (target - a2) / (a3 - a2) * 1000
            print(f"    {s:<11} {n:.0f} labels, saving {10000 / n:.1f}x")

print("\nTable III: pretrained vs random-init, mean ± SE over 3 seeds, Delta = Confidence - Random per seed")
for setting, folder in (("pretrained", OURS), ("random init", RANDINIT)):
    print(f"  {setting}")
    for n in (2000, 3000, 10000):
        r, c = acc(folder, "random", n, SEEDS[:3]), acc(folder, "confidence", n, SEEDS[:3])
        d = c - r
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        p = 1 - abs(t) / np.sqrt(t * t + 2)  # two-sided p of a paired t-test with 3 seeds (df = 2)
        print(f"    {n:>5} labels: random {show(r)}   confidence {show(c)}   delta {show(d)}   t = {t:.2f}, p = {p:.3f}")
