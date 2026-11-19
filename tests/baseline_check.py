import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from environment import ReadCoachEnv


def run(policy, episodes=50):
    totals, levels, dropouts = [], [], 0
    for ep in range(episodes):
        env = ReadCoachEnv()
        obs, _ = env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, info = env.step(policy(env))
            total += r
            done = term or trunc
        totals.append(total)
        levels.append(info["level"])
        dropouts += info["dropped_out"]
    return round(np.mean(totals), 1), round(np.mean(levels), 2), dropouts


policies = {
    "random": lambda e: e.action_space.sample(),
    "always_stretch_comic": lambda e: 3,
    "always_rest": lambda e: 8,
    "always_peer_nudge": lambda e: 6,
    "always_recommend_comic": lambda e: 0,
}

for name, p in policies.items():
    reward, level, drops = run(p)
    print(f"{name:25s} reward={reward:7.1f}  level={level}  dropouts={drops}/50")