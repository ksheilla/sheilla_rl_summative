import argparse
import json
from collections import Counter
from pathlib import Path

import pygame
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP

from environment import ACTION_NAMES, ReadCoachEnv

ROOT = Path(__file__).resolve().parent

# algorithm -> models/logs subfolder
FOLDERS = {"dqn": "dqn", "reinforce": "pg", "a2c": "pg", "ppo": "pg"}

# Best run of each algorithm (highest evaluation reward)
BEST_RUNS = {
    "dqn": "dqn_08_eps_fast_decay",
    "reinforce": "reinforce_05_gamma_0.999",
    "a2c": "a2c_11_rollout_120",
    "ppo": "ppo_10_entropy_0.05",
}

OBS_NAMES = [
    "reading_level",
    "interest_comic",
    "interest_thriller",
    "interest_ya",
    "motivation",
    "streak",
    "boredom",
    "nudge_fatigue",
    "day",
    "last_genre_comic",
    "last_genre_thriller",
    "last_genre_ya",
]


def load_agent(algo, run):
    """Load a trained network (Q-values for DQN, action logits for policy gradient)."""
    folder = FOLDERS[algo]
    hidden = [128, 128]
    summary = ROOT / "logs" / folder / f"{run}_summary.json"
    if summary.exists():
        hidden = json.loads(summary.read_text())["hidden"]
    net = TensorDictModule(
        MLP(in_features=12, out_features=9, num_cells=list(hidden)),
        in_keys=["observation"],
        out_keys=["scores"],
    )
    state = torch.load(ROOT / "models" / folder / f"{run}.pt", map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    return net


def choose_action(net, obs):
    """Greedy action: the one with the highest Q-value / logit."""
    td = TensorDict(
        {"observation": torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)},
        batch_size=[1],
    )
    with torch.no_grad():
        return int(net(td)["scores"].argmax(-1).item())


def predict_json(net, payload):
    """Integration path: JSON in (student state) -> JSON out (recommended action).
    A web or mobile backend could call this for each student."""
    state = json.loads(payload)
    obs = [float(state[name]) for name in OBS_NAMES]
    action = choose_action(net, obs)
    return json.dumps({"action_id": action, "action": ACTION_NAMES[action]})


def play(net, episodes, seed, render, fps, verbose):
    env = ReadCoachEnv(render_mode="human" if render else None)
    env.render_fps = fps
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done, counts = 0.0, False, Counter()
        print(f"\n=== Student {ep + 1} of {episodes} (seed {seed + ep}) ===")
        if render:
            env.render()  # show the starting room before the first move
        while not done:
            action = choose_action(net, obs)
            counts[ACTION_NAMES[action]] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
            if verbose:
                print(
                    f"Day {env.day:2d} | {ACTION_NAMES[action]:<19} | "
                    f"read {info['minutes']:5.1f} min | reward {reward:5.2f} | "
                    f"motivation {env.motivation:.2f} | level {env.level:.2f} | "
                    f"streak {env.streak}"
                )
            if render:
                env.render()
                if any(e.type == pygame.QUIT for e in pygame.event.get()):
                    done = True
        print(
            f"\nStudent {ep + 1}: total reward={total:.1f}  days={env.day}  "
            f"final reading level={info['level']:.2f}  dropped out={info['dropped_out']}"
        )
        print("Actions used:", dict(counts.most_common()))
        if render:
            pygame.time.wait(1500)  # hold the last frame before the next student
    env.close()


def main():
    parser = argparse.ArgumentParser(description="READit Coach: run a trained RL agent")
    parser.add_argument("--algo", choices=list(FOLDERS), default="dqn")
    parser.add_argument("--run", default=None, help="run name (default: best run of the algorithm)")
    parser.add_argument("--episodes", type=int, default=1, help="number of students to simulate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=5, help="days per second in the window")
    parser.add_argument("--verbose", action="store_true", help="print every day's decision")
    parser.add_argument("--no-render", action="store_true", help="skip the visualization window")
    parser.add_argument("--api-demo", action="store_true", help="show the JSON prediction example")
    args = parser.parse_args()

    run = args.run or BEST_RUNS[args.algo]
    print(f"Loading {args.algo.upper()} agent: {run}")
    net = load_agent(args.algo, run)

    if args.api_demo:
        env = ReadCoachEnv()
        obs, _ = env.reset(seed=args.seed)
        request = json.dumps({n: round(float(v), 3) for n, v in zip(OBS_NAMES, obs)})
        print("Request :", request)
        print("Response:", predict_json(net, request))
        return

    play(net, args.episodes, args.seed, render=not args.no_render, fps=args.fps, verbose=args.verbose)


if __name__ == "__main__":
    main()