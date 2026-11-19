import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from environment import ReadCoachEnv


def smart_policy(env, state={"i": 0}):
    # Rotate genres to avoid boredom
    genre = env.day % 3
    # Nudge when motivation is low and the student is not nudge-fatigued
    if env.motivation < 0.35 and env.fatigue < 0.3:
        return 6
    # Stretch when motivated and not bored
    if env.motivation > 0.6 and env.boredom < 0.3:
        return 3 + genre
    # Otherwise recommend at the current level
    return genre


def run(policy, episodes=50):
    totals, levels, dropouts = [], [], 0
    for ep in range(episodes):
        env = ReadCoachEnv()
        env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, info = env.step(policy(env))
            total += r
            done = term or trunc
        totals.append(total)
        levels.append(info["level"])
        dropouts += info["dropped_out"]
    return round(np.mean(totals), 1), round(np.mean(levels), 2), dropouts


reward, level, drops = run(smart_policy)
print(f"smart_policy  reward={reward}  level={level}  dropouts={drops}/50")
rand = run(lambda e: e.action_space.sample())
print(f"random        reward={rand[0]}  level={rand[1]}  dropouts={rand[2]}/50")