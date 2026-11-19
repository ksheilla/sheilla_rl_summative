from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

DQN_DIR = ROOT / "logs" / "dqn"
PG_DIR = ROOT / "logs" / "pg"
ALGOS = {
    "DQN": (DQN_DIR, "dqn"),
    "REINFORCE": (PG_DIR, "reinforce"),
    "A2C": (PG_DIR, "a2c"),
    "PPO": (PG_DIR, "ppo"),
}
COLORS = {"DQN": "#1f77b4", "REINFORCE": "#d62728", "A2C": "#2ca02c", "PPO": "#9467bd"}

# Reference scores measured earlier with the hand-made and random policies
RANDOM_REWARD = 46.1
HEURISTIC_REWARD = 55.9
THRESHOLD = 65.0  # reward level used to measure convergence speed
WINDOW = 10  # episodes used for smoothing
MAX_ENTROPY = float(np.log(9))
TOTAL_FRAMES = 30_000

results = {}
for algo, (folder, prefix) in ALGOS.items():
    df = pd.read_csv(folder / f"{prefix}_results.csv")
    df["algo"] = algo
    results[algo] = df
all_results = pd.concat(results.values(), ignore_index=True)


def load(algo, name, kind):
    return pd.read_csv(ALGOS[algo][0] / f"{name}_{kind}.csv")


def smooth_curve(algo, name):
    ep = load(algo, name, "episodes")
    ep["smooth"] = ep["episode_reward"].rolling(WINDOW, min_periods=1).mean()
    return ep


def best_run(algo):
    df = results[algo]
    return df.sort_values("eval_mean", ascending=False).iloc[0]["name"]


def baseline_run(algo):
    return f"{ALGOS[algo][1]}_01_baseline"


# ---------- Figure 1 & 2: comparative training rewards ----------
for fig_name, picker, title in [
    ("fig1_reward_baselines.png", baseline_run, "Training reward: baseline run of each algorithm"),
    ("fig2_reward_best.png", best_run, "Training reward: best run (by evaluation) of each algorithm"),
]:
    plt.figure(figsize=(9, 5))
    for algo in ALGOS:
        name = picker(algo)
        ep = smooth_curve(algo, name)
        plt.plot(ep["frames"], ep["smooth"], label=f"{algo} ({name})", color=COLORS[algo])
    plt.axhline(RANDOM_REWARD, color="gray", ls=":", label="random policy")
    plt.axhline(HEURISTIC_REWARD, color="gray", ls="--", label="hand-made heuristic")
    plt.xlabel("Training frames (1 frame = 1 simulated day)")
    plt.ylabel(f"Episode reward ({WINDOW}-episode average)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS / fig_name, dpi=150)
    plt.close()

# ---------- Figure 3: DQN loss ----------
plt.figure(figsize=(9, 5))
for name in results["DQN"]["name"]:
    lo = load("DQN", name, "loss")
    is_base = name == baseline_run("DQN")
    is_tau = "tau_0.1" in name
    plt.plot(
        lo["frames"],
        lo["loss"],
        lw=2.5 if (is_base or is_tau) else 1,
        alpha=1 if (is_base or is_tau) else 0.35,
        color="black" if is_base else ("red" if is_tau else "#1f77b4"),
        label="baseline" if is_base else ("tau=0.1 (unstable)" if is_tau else None),
    )
plt.yscale("log")
plt.xlabel("Training frames")
plt.ylabel("DQN loss (smooth L1, log scale)")
plt.title("DQN objective: loss over training, all 12 runs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS / "fig3_dqn_loss.png", dpi=150)
plt.close()

# ---------- Figure 4: policy-gradient entropy ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, algo in zip(axes, ["REINFORCE", "A2C", "PPO"]):
    for name in results[algo]["name"]:
        lo = load(algo, name, "loss")
        is_base = name == baseline_run(algo)
        ax.plot(
            lo["frames"],
            lo["entropy"],
            lw=2.5 if is_base else 1,
            alpha=1 if is_base else 0.35,
            color="black" if is_base else COLORS[algo],
            label="baseline" if is_base else None,
        )
    ax.axhline(MAX_ENTROPY, color="gray", ls=":", label="max entropy (uniform)")
    ax.set_title(f"{algo}: policy entropy")
    ax.set_xlabel("Training frames")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Policy entropy")
plt.tight_layout()
plt.savefig(ASSETS / "fig4_pg_entropy.png", dpi=150)
plt.close()

# ---------- Figure 5: convergence speed ----------
rows = []
plt.figure(figsize=(9, 5))
for i, algo in enumerate(ALGOS):
    for name in results[algo]["name"]:
        ep = smooth_curve(algo, name)
        hit = ep[ep["smooth"] >= THRESHOLD]
        frames = float(hit["frames"].iloc[0]) if len(hit) else np.nan
        rows.append({"algo": algo, "name": name, "frames_to_threshold": frames})
        x = i + np.random.uniform(-0.18, 0.18)
        if np.isnan(frames):
            plt.scatter(x, TOTAL_FRAMES * 1.06, marker="x", color="gray")
        else:
            plt.scatter(x, frames, color=COLORS[algo], alpha=0.8)
conv = pd.DataFrame(rows)
plt.xticks(range(4), list(ALGOS))
plt.ylim(0, TOTAL_FRAMES * 1.12)
plt.ylabel(f"Frames until {WINDOW}-episode average reward >= {THRESHOLD:.0f}")
plt.title("Convergence speed (gray x = never reached)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS / "fig5_convergence.png", dpi=150)
plt.close()

# ---------- Figure 6: evaluation on unseen episodes ----------
from matplotlib.lines import Line2D

ordered = pd.concat(
    [results[a].sort_values("eval_mean", ascending=False) for a in ALGOS], ignore_index=True
)
plt.figure(figsize=(12, 5))
plt.bar(
    range(len(ordered)),
    ordered["eval_mean"],
    yerr=ordered["eval_std"],
    color=[COLORS[a] for a in ordered["algo"]],
    capsize=2,
)
plt.axhline(RANDOM_REWARD, color="gray", ls=":")
plt.axhline(HEURISTIC_REWARD, color="gray", ls="--")
plt.xticks([])
plt.ylim(30, 85)
plt.ylabel("Mean reward on 20 unseen episodes (greedy policy)")
plt.title("Evaluation of every run, grouped by algorithm")
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[a]) for a in ALGOS]
handles += [Line2D([0], [0], color="gray", ls=":"), Line2D([0], [0], color="gray", ls="--")]
plt.legend(handles, list(ALGOS) + ["random policy", "hand-made heuristic"], ncol=3, fontsize=8)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(ASSETS / "fig6_evaluation.png", dpi=150)
plt.close()

# ---------- Summary table ----------
summary = []
for algo in ALGOS:
    df = results[algo]
    b = df.sort_values("eval_mean", ascending=False).iloc[0]
    reached = conv[(conv["algo"] == algo) & conv["frames_to_threshold"].notna()]
    summary.append(
        {
            "algo": algo,
            "runs": len(df),
            "best_run": b["name"],
            "best_eval": round(b["eval_mean"], 1),
            "median_eval": round(df["eval_mean"].median(), 1),
            "worst_eval": round(df["eval_mean"].min(), 1),
            "runs_reaching_threshold": f"{len(reached)}/{len(df)}",
            "median_frames_to_threshold": reached["frames_to_threshold"].median(),
        }
    )
summary = pd.DataFrame(summary)
summary.to_csv(ASSETS / "summary_best_per_algo.csv", index=False)
all_results[["algo", "name", "eval_mean", "eval_std", "final_train_reward"]].to_csv(
    ASSETS / "all_runs_table.csv", index=False
)
print(summary.to_string(index=False))
print("\nSaved figures to assets/:")
for f in sorted(ASSETS.glob("fig*.png")):
    print(" -", f.name)