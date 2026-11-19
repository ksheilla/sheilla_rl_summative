import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.optim import Adam
from torchrl.collectors import Collector as SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymWrapper, RewardSum, TransformedEnv
from torchrl.modules import EGreedyModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

from environment import ReadCoachEnv

LOG_DIR = ROOT / "logs" / "dqn"
MODEL_DIR = ROOT / "models" / "dqn"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Every experiment uses the same total_frames so the comparison is fair.
DEFAULT_CFG = {
    "seed": 0,
    "total_frames": 30_000,
    "frames_per_batch": 200,
    "updates_per_batch": 50,
    "learning_starts": 1_000,
    "batch_size": 64,
    "buffer_size": 20_000,
    "lr": 1e-3,
    "gamma": 0.99,
    "hidden": [128, 128],
    "eps_init": 1.0,
    "eps_end": 0.05,
    "eps_decay_frames": 15_000,
    "tau": 0.01,  # soft target update rate (target = (1-tau)*target + tau*online)
}


def make_env(seed):
    inner = ReadCoachEnv()
    inner.reset(seed=seed)  # seeds the environment's random generator
    env = GymWrapper(inner, categorical_action_encoding=True)
    return TransformedEnv(env, RewardSum())


def evaluate(value_net, episodes=20, seed=1000):
    """Greedy evaluation on fresh episodes (no exploration)."""
    env = ReadCoachEnv()
    totals = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            td = TensorDict(
                {"observation": torch.tensor(obs, dtype=torch.float32).unsqueeze(0)},
                batch_size=[1],
            )
            with torch.no_grad():
                action = value_net(td)["action_value"].argmax(-1).item()
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        totals.append(total)
    return float(np.mean(totals)), float(np.std(totals))


def run_experiment(name, **overrides):
    cfg = {**DEFAULT_CFG, **overrides}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    start_time = time.time()

    env = make_env(cfg["seed"])
    n_actions = env.action_spec.space.n

    # Q-network: 9 observations -> Q-value for each of the 9 actions
    mlp = MLP(in_features=12, out_features=n_actions, num_cells=list(cfg["hidden"]))
    value_net = TensorDictModule(mlp, in_keys=["observation"], out_keys=["action_value"])
    policy = TensorDictSequential(value_net, QValueModule(spec=env.action_spec))

    explore = EGreedyModule(
        spec=env.action_spec,
        eps_init=cfg["eps_init"],
        eps_end=cfg["eps_end"],
        annealing_num_steps=cfg["eps_decay_frames"],
    )
    policy_explore = TensorDictSequential(policy, explore)

    collector = SyncDataCollector(
        env,
        policy_explore,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=cfg["total_frames"],
        init_random_frames=cfg["learning_starts"],
                auto_register_policy_transforms=True,
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]))

    loss_module = DQNLoss(
        value_network=policy, action_space=env.action_spec, loss_function="smooth_l1"
    )
    loss_module.make_value_estimator(gamma=cfg["gamma"])
    optim = Adam(loss_module.parameters(), lr=cfg["lr"])
    updater = SoftUpdate(loss_module, eps=1.0 - cfg["tau"])

    episode_log, loss_log = [], []
    frames = 0

    for i, data in enumerate(collector):
        frames += data.numel()
        rb.extend(data)

        # Record finished episodes (total reward of each 60-day episode)
        done_mask = data["next", "done"].squeeze(-1)
        if done_mask.any():
            for r in data["next", "episode_reward"][done_mask].flatten().tolist():
                episode_log.append({"frames": frames, "episode_reward": r})

        batch_losses = []
        if len(rb) >= cfg["learning_starts"]:
            for _ in range(cfg["updates_per_batch"]):
                sample = rb.sample(cfg["batch_size"])
                loss_vals = loss_module(sample)
                loss_vals["loss"].backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 10.0)
                optim.step()
                optim.zero_grad()
                updater.step()
                batch_losses.append(loss_vals["loss"].item())
        explore.step(data.numel())

        if batch_losses:
            loss_log.append({"frames": frames, "loss": float(np.mean(batch_losses))})

        if i % 10 == 0:
            recent = [e["episode_reward"] for e in episode_log[-10:]]
            mean_r = np.mean(recent) if recent else float("nan")
            last_loss = loss_log[-1]["loss"] if loss_log else float("nan")
            print(
                f"[{name}] frames={frames:6d}  eps={explore.eps.item():.2f}  "
                f"recent_reward={mean_r:6.1f}  loss={last_loss:.4f}"
            )

    collector.shutdown()

    # ---- Save model + logs ----
    torch.save(value_net.state_dict(), MODEL_DIR / f"{name}.pt")
    pd.DataFrame(episode_log).to_csv(LOG_DIR / f"{name}_episodes.csv", index=False)
    pd.DataFrame(loss_log).to_csv(LOG_DIR / f"{name}_loss.csv", index=False)

    eval_mean, eval_std = evaluate(value_net)
    last_train = [e["episode_reward"] for e in episode_log[-20:]]
    summary = {
        "name": name,
        "final_train_reward": float(np.mean(last_train)) if last_train else None,
        "eval_mean": eval_mean,
        "eval_std": eval_std,
        "minutes": round((time.time() - start_time) / 60, 2),
        **{k: v for k, v in cfg.items()},
    }
    with open(LOG_DIR / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{name}] DONE  eval_mean={eval_mean:.1f} +/- {eval_std:.1f}  "
          f"({summary['minutes']} min)\n")
    return summary


EXPERIMENTS = [
    ("dqn_01_baseline", {}),
    ("dqn_02_lr_1e-4", {"lr": 1e-4}),
    ("dqn_03_lr_5e-3", {"lr": 5e-3}),
    ("dqn_04_gamma_0.90", {"gamma": 0.90}),
    ("dqn_05_gamma_0.999", {"gamma": 0.999}),
    ("dqn_06_batch_256", {"batch_size": 256}),
    ("dqn_07_buffer_2k", {"buffer_size": 2_000}),
    ("dqn_08_eps_fast_decay", {"eps_decay_frames": 5_000}),
    ("dqn_09_eps_slow_high_floor", {"eps_decay_frames": 28_000, "eps_end": 0.20}),
    ("dqn_10_tau_0.001", {"tau": 0.001}),
    ("dqn_11_tau_0.1", {"tau": 0.1}),
    ("dqn_12_small_net", {"hidden": [32, 32]}),
]


def run_all():
    rows = []
    for name, overrides in EXPERIMENTS:
        summary_path = LOG_DIR / f"{name}_summary.json"
        if summary_path.exists():  # lets you resume if you stop midway
            print(f"Skipping {name} (already done)")
            with open(summary_path) as f:
                rows.append(json.load(f))
        else:
            rows.append(run_experiment(name, **overrides))
        pd.DataFrame(rows).to_csv(LOG_DIR / "dqn_results.csv", index=False)

    df = pd.DataFrame(rows)[["name", "eval_mean", "eval_std", "final_train_reward"]]
    print(df.sort_values("eval_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        run_all()
    else:
        run_experiment("dqn_test")