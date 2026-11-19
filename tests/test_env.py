import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment import ReadCoachEnv

env = ReadCoachEnv()
obs, _ = env.reset(seed=0)
print("start obs:", obs)

total, done = 0.0, False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total += reward
    done = terminated or truncated

print("days played:", env.day)
print("total reward:", round(total, 2))
print("dropped out:", info["dropped_out"])
print("final level:", round(info["level"], 3))