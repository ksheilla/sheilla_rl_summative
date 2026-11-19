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
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torch.optim import Adam
from torchrl.collectors import Collector as SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import GymWrapper, RewardSum, TransformedEnv
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

from environment import ReadCoachEnv

LOG_DIR = ROOT / "logs" / "pg"
MODEL_DIR = ROOT / "models" / "pg"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Settings shared by all three algorithms (same total_frames as DQN = fair comparison)
BASE_CFG = {
    "seed": 0,
    "total_frames": 30_000,
    "gamma": 0.99,
    "hidden": [128, 128],
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
}

# Algorithm-specific defaults
ALGO_DEFAULTS = {
    "reinforce": {"frames_per_batch": 240, "lr": 3e-3, "normalize_returns": True},
    "a2c": {
        "frames_per_batch": 240,
        "lr": 1e-3,
        "gae_lambda": 0.95,
        "critic_coef": 0.5,
    },
    "ppo": {
        "frames_per_batch": 1000,
        "lr": 1e-3,
        "gae_lambda": 0.95,
        "critic_coef": 0.5,
        "clip_epsilon": 0.2,
        "ppo_epochs": 8,
        "minibatch_size": 125,
    },
}


def make_env(seed):
    inner = ReadCoachEnv()
    inner.reset(seed=seed)
    env = GymWrapper(inner, categorical_action_encoding=True)
    return TransformedEnv(env, RewardSum())


def evaluate(actor_module, episodes=20, seed=1000):
    """Greedy evaluation (most likely action), same episodes as the DQN evaluation."""
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
                action = actor_module(td)["logits"].argmax(-1).item()
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        totals.append(total)
    return float(np.mean(totals)), float(np.std(totals))


def reward_to_go(rewards, dones, gamma):
    out = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running = 0.0
        running = rewards[t].item() + gamma * running
        out[t] = running
    return out


def total_loss(loss_vals):
    """Sum every loss term TorchRL returned (entropy is absent when the coefficient is 0)."""
    return sum(v for k, v in loss_vals.items() if k.startswith("loss_"))


def get_entropy(loss_vals, actor, td):
    if "entropy" in loss_vals.keys():
        return loss_vals["entropy"].item()
    with torch.no_grad():
        return actor.get_dist(td.select("observation")).entropy().mean().item()


def run_experiment(name, algo, **overrides):
    cfg = {**BASE_CFG, **ALGO_DEFAULTS[algo], **overrides}
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    start_time = time.time()

    env = make_env(cfg["seed"])
    n_actions = env.action_spec.space.n

    # Policy network: 12 observations -> logits for 9 actions
    actor_net = MLP(in_features=12, out_features=n_actions, num_cells=list(cfg["hidden"]))
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(
        actor_module,
        spec=env.action_spec,
        in_keys=["logits"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    critic = None
    if algo in ("a2c", "ppo"):
        critic_net = MLP(in_features=12, out_features=1, num_cells=list(cfg["hidden"]))
        critic = ValueOperator(critic_net, in_keys=["observation"])

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=cfg["frames_per_batch"],
        total_frames=cfg["total_frames"],
        auto_register_policy_transforms=True,
    )

    if algo == "reinforce":
        loss_module = None
        optim = Adam(actor.parameters(), lr=cfg["lr"])
    else:
        advantage = GAE(
            gamma=cfg["gamma"],
            lmbda=cfg["gae_lambda"],
            value_network=critic,
            average_gae=True,
        )
        if algo == "ppo":
            loss_module = ClipPPOLoss(
                actor_network=actor,
                critic_network=critic,
                clip_epsilon=cfg["clip_epsilon"],
                entropy_bonus=True,
                entropy_coeff=cfg["entropy_coef"],
                critic_coeff=cfg["critic_coef"],
                loss_critic_type="smooth_l1",
            )
            rb = ReplayBuffer(
                storage=LazyTensorStorage(cfg["frames_per_batch"]),
                sampler=SamplerWithoutReplacement(),
            )
        else:
            loss_module = A2CLoss(
                actor_network=actor,
                critic_network=critic,
                entropy_bonus=True,
                entropy_coeff=cfg["entropy_coef"],
                critic_coeff=cfg["critic_coef"],
            )
        optim = Adam(loss_module.parameters(), lr=cfg["lr"])

    episode_log, loss_log = [], []
    frames, next_print = 0, 0

    for data in collector:
        frames += data.numel()

        # Record finished episodes
        done_mask = data["next", "done"].squeeze(-1)
        if done_mask.any():
            for r in data["next", "episode_reward"][done_mask].flatten().tolist():
                episode_log.append({"frames": frames, "episode_reward": r})

        losses, entropies = [], []

        if algo == "reinforce":
            rewards = data["next", "reward"].squeeze(-1)
            dones = data["next", "done"].squeeze(-1)
            returns = reward_to_go(rewards, dones, cfg["gamma"])
            if cfg["normalize_returns"]:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            dist = actor.get_dist(data.select("observation"))
            log_prob = dist.log_prob(data["action"])
            entropy = dist.entropy().mean()
            loss = -(log_prob * returns).mean() - cfg["entropy_coef"] * entropy
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"])
            optim.step()
            losses.append(loss.item())
            entropies.append(entropy.item())

        elif algo == "a2c":
            with torch.no_grad():
                advantage(data)
            loss_vals = loss_module(data)
            loss = total_loss(loss_vals)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg["max_grad_norm"])
            optim.step()
            losses.append(loss.item())
            entropies.append(get_entropy(loss_vals, actor, data))

        else:  # ppo
            with torch.no_grad():
                advantage(data)
            rb.extend(data.reshape(-1))
            n_minibatches = cfg["frames_per_batch"] // cfg["minibatch_size"]
            for _ in range(cfg["ppo_epochs"]):
                for _ in range(n_minibatches):
                    sub = rb.sample(cfg["minibatch_size"])
                    loss_vals = loss_module(sub)
                    loss = total_loss(loss_vals)
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), cfg["max_grad_norm"]
                    )
                    optim.step()
                    losses.append(loss.item())
                    entropies.append(get_entropy(loss_vals, actor, sub))

        loss_log.append(
            {"frames": frames, "loss": float(np.mean(losses)), "entropy": float(np.mean(entropies))}
        )

        if frames >= next_print:
            recent = [e["episode_reward"] for e in episode_log[-10:]]
            mean_r = np.mean(recent) if recent else float("nan")
            print(
                f"[{name}] frames={frames:6d}  recent_reward={mean_r:6.1f}  "
                f"entropy={loss_log[-1]['entropy']:.3f}  loss={loss_log[-1]['loss']:.4f}"
            )
            next_print += 2_000

    collector.shutdown()

    torch.save(actor_module.state_dict(), MODEL_DIR / f"{name}.pt")
    pd.DataFrame(episode_log).to_csv(LOG_DIR / f"{name}_episodes.csv", index=False)
    pd.DataFrame(loss_log).to_csv(LOG_DIR / f"{name}_loss.csv", index=False)

    eval_mean, eval_std = evaluate(actor_module)
    last_train = [e["episode_reward"] for e in episode_log[-20:]]
    summary = {
        "name": name,
        "algo": algo,
        "final_train_reward": float(np.mean(last_train)) if last_train else None,
        "eval_mean": eval_mean,
        "eval_std": eval_std,
        "minutes": round((time.time() - start_time) / 60, 2),
        **cfg,
    }
    with open(LOG_DIR / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{name}] DONE  eval_mean={eval_mean:.1f} +/- {eval_std:.1f}  "
          f"({summary['minutes']} min)\n")
    return summary


EXPERIMENTS = {
    "reinforce": [
        ("reinforce_01_baseline", {}),
        ("reinforce_02_lr_1e-3", {"lr": 1e-3}),
        ("reinforce_03_lr_1e-2", {"lr": 1e-2}),
        ("reinforce_04_gamma_0.90", {"gamma": 0.90}),
        ("reinforce_05_gamma_0.999", {"gamma": 0.999}),
        ("reinforce_06_no_normalization", {"normalize_returns": False}),
        ("reinforce_07_entropy_0", {"entropy_coef": 0.0}),
        ("reinforce_08_entropy_0.05", {"entropy_coef": 0.05}),
        ("reinforce_09_rollout_120", {"frames_per_batch": 120}),
        ("reinforce_10_rollout_600", {"frames_per_batch": 600}),
        ("reinforce_11_small_net", {"hidden": [32, 32]}),
    ],
    "a2c": [
        ("a2c_01_baseline", {}),
        ("a2c_02_lr_3e-4", {"lr": 3e-4}),
        ("a2c_03_lr_3e-3", {"lr": 3e-3}),
        ("a2c_04_gamma_0.90", {"gamma": 0.90}),
        ("a2c_05_gamma_0.999", {"gamma": 0.999}),
        ("a2c_06_gae_lambda_0.80", {"gae_lambda": 0.80}),
        ("a2c_07_gae_lambda_1.0", {"gae_lambda": 1.0}),
        ("a2c_08_entropy_0", {"entropy_coef": 0.0}),
        ("a2c_09_entropy_0.05", {"entropy_coef": 0.05}),
        ("a2c_10_critic_coef_1.0", {"critic_coef": 1.0}),
        ("a2c_11_rollout_120", {"frames_per_batch": 120}),
        ("a2c_12_rollout_600", {"frames_per_batch": 600}),
    ],
    "ppo": [
        ("ppo_01_baseline", {}),
        ("ppo_02_lr_3e-4", {"lr": 3e-4}),
        ("ppo_03_lr_3e-3", {"lr": 3e-3}),
        ("ppo_04_clip_0.1", {"clip_epsilon": 0.1}),
        ("ppo_05_clip_0.3", {"clip_epsilon": 0.3}),
        ("ppo_06_epochs_3", {"ppo_epochs": 3}),
        ("ppo_07_epochs_15", {"ppo_epochs": 15}),
        ("ppo_08_gae_lambda_0.80", {"gae_lambda": 0.80}),
        ("ppo_09_entropy_0", {"entropy_coef": 0.0}),
        ("ppo_10_entropy_0.05", {"entropy_coef": 0.05}),
        ("ppo_11_rollout_500", {"frames_per_batch": 500}),
        ("ppo_12_minibatch_250", {"minibatch_size": 250}),
    ],
}


def run_all(algo):
    rows = []
    for name, overrides in EXPERIMENTS[algo]:
        summary_path = LOG_DIR / f"{name}_summary.json"
        if summary_path.exists():  # lets you resume if you stop midway
            print(f"Skipping {name} (already done)")
            with open(summary_path) as f:
                rows.append(json.load(f))
        else:
            rows.append(run_experiment(name, algo, **overrides))
        pd.DataFrame(rows).to_csv(LOG_DIR / f"{algo}_results.csv", index=False)

    df = pd.DataFrame(rows)[["name", "eval_mean", "eval_std", "final_train_reward"]]
    print(df.sort_values("eval_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "test"
    if arg == "test":
        for a in ["reinforce", "a2c", "ppo"]:
            run_experiment(f"smoke_{a}", a, total_frames=6_000)
    elif arg == "all":
        for a in ["reinforce", "a2c", "ppo"]:
            run_all(a)
    else:
        run_all(arg)