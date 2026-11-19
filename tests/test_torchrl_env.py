import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchrl.envs import GymWrapper
from torchrl.envs.utils import check_env_specs
from environment import ReadCoachEnv

env = GymWrapper(ReadCoachEnv(), categorical_action_encoding=True)
check_env_specs(env)

print("observation spec:", env.observation_spec)
print("action spec:", env.action_spec)

rollout = env.rollout(max_steps=5)
print(rollout)