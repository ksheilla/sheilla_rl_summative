import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame
from environment import ReadCoachEnv


def smart_policy(env):
    genre = env.day % 3
    if env.motivation < 0.35 and env.fatigue < 0.3:
        return 6
    if env.motivation > 0.6 and env.boredom < 0.3:
        return 3 + genre
    return genre


env = ReadCoachEnv(render_mode="human")
env.reset(seed=3)
done = False
while not done:
    _, _, term, trunc, _ = env.step(smart_policy(env))
    env.render()
    if env.day == 30:
        pygame.image.save(env._renderer.screen, "assets/env_visual.png")
    done = term or trunc
env.close()
print("Saved screenshot to assets/env_visual.png")