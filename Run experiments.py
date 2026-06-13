import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pachhaanna import make_permuted_tasks, run_strategy

D, K, T = 20, 5, 5
SIZES = [D, 128, 128, K]
SEEDS = [0, 1, 2]

STRATS = {
    "Naive (no recognition)":      dict(),
    "EWC (protect what mattered)": dict(use_ewc=True, ewc_lambda=1.0),
    "Prototype replay (recall essence)": dict(use_replay=True, replay_frac=1.0),
    "Pachhaanna (EWC + replay)":   dict(use_ewc=True, ewc_lambda=1.0, use_replay=True, replay_frac=1.0),
}

results = {name: {"avg": [], "forget": [], "task1_curve": [], "final": []} for name in STRATS}

for seed in SEEDS:
    tasks = make_permuted_tasks(T, D, K, 2500, 1000, sep=1.0,
                                rng=np.random.default_rng(100 + seed))
    for name, kw in STRATS.items():
        R, net, mem = run_strategy(name, tasks, SIZES, seed=seed, epochs=20, lr=0.05, **kw)
        final = R[-1, :]
        results[name]["avg"].append(final.mean())
        results[name]["forget"].append(np.mean([R[j:, j].max() - R[-1, j] for j in range(T - 1)]))
        results[name]["task1_curve"].append([R[i, 0] for i in range(T)])
        results[name]["final"].append(final)

print(f"{'strategy':<35}{'avg final acc':>16}{'forgetting':>14}")
print("-" * 65)
for name in STRATS:
    a = np.array(results[name]["avg"]); f = np.array(results[name]["forget"])
    print(f"{name:<35}{a.mean():>10.3f}±{a.std():.2f}{f.mean():>10.3f}±{f.std():.2f}")

# ---- Figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colors = ["#b0b0b0", "#e08a3c", "#3c7de0", "#2a9d5c"]

names = list(STRATS)
means = [np.mean(results[n]["avg"]) for n in names]
errs = [np.std(results[n]["avg"]) for n in names]
ax1.bar(range(len(names)), means, yerr=errs, color=colors, capsize=4)
ax1.axhline(1 / K, ls="--", c="k", lw=1, alpha=0.6, label=f"chance ({1/K:.2f})")
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(["Naive", "EWC", "Replay", "EWC+Replay"], fontsize=10)
ax1.set_ylabel("avg accuracy over ALL tasks (after learning all 5)")
ax1.set_title("Memory retained across the whole sequence")
ax1.set_ylim(0, 1); ax1.legend()

for n, c in zip(names, colors):
    curve = np.array(results[n]["task1_curve"]).mean(0)
    ax2.plot(range(1, T + 1), curve, "-o", color=c, label=n.split(" (")[0])
ax2.axhline(1 / K, ls="--", c="k", lw=1, alpha=0.6)
ax2.set_xlabel("after training on task #")
ax2.set_ylabel("accuracy on TASK 1")
ax2.set_title("Forgetting curve: does Task 1 survive?")
ax2.set_xticks(range(1, T + 1)); ax2.set_ylim(0, 1); ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/forgetting.png", dpi=130)
print("\nfigure saved")
