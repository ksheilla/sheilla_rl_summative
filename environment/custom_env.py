import numpy as np
import gymnasium as gym
from gymnasium import spaces

GENRES = ["comic", "thriller", "young_adult"]

ACTION_NAMES = [
    "recommend_comic",
    "recommend_thriller",
    "recommend_ya",
    "stretch_comic",
    "stretch_thriller",
    "stretch_ya",
    "peer_nudge",
    "badge_challenge",
    "rest",
]


class ReadCoachEnv(gym.Env):
    """
    READit Coach: an agent guides one student's leisure reading over 60 days.
    1 step = 1 day.

    Observation (9 values, all in [0, 1]):
      0 reading_level, 1-3 genre interests (comic, thriller, ya),
      4 motivation, 5 streak (min(streak,30)/30), 6 boredom,
      7 nudge_fatigue, 8 day/max_days

    Actions (Discrete 9):
      0-2 recommend comic / thriller / ya at the student's level
      3-5 recommend the same genres as a harder "stretch" book
      6 peer nudge, 7 badge challenge, 8 rest
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, max_days=60, render_mode=None):
        super().__init__()
        self.max_days = max_days
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self._renderer = None
        self.last_action = None
        self.last_minutes = 0.0
        self.reset()

    # ------------------------------------------------------------------ #
    def _get_obs(self):
        obs = np.array(
            [
                self.level,
                *self.interest,
                self.motivation,
                min(self.streak, 30) / 30.0,
                self.boredom,
                self.fatigue,
                self.day / self.max_days,
                *[1.0 if self.last_genre == g else 0.0 for g in range(3)],
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Random student profile (start state)
        self.level = float(rng.uniform(0.1, 0.5))
        self.interest = rng.uniform(0.2, 0.9, size=3).astype(np.float64)
        self.motivation = float(rng.uniform(0.4, 0.7))
        self.streak = 0
        self.boredom = 0.0
        self.fatigue = 0.0
        self.day = 0
        self.last_genre = -1
        self.low_motivation_days = 0
        self.last_action = None
        self.last_minutes = 0.0

        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    def step(self, action):
        action = int(action)
        rng = self.np_random
        self.last_action = action

        old_level = self.level
        is_book = action <= 5
        is_stretch = 3 <= action <= 5
        genre = action % 3 if is_book else None

        # ---- 1. Engagement for today (how much the student reads) ----
        engagement = self.motivation * (1.0 - 0.7 * self.boredom)
        if is_book:
            engagement *= 0.5 + 0.5 * self.interest[genre]
            if is_stretch:
                engagement *= 0.8  # harder book, less comfortable today
        else:
            engagement *= 0.4  # student continues an old book 

        max_minutes = 30.0 + 20.0 * self.level  # higher level reads more
        minutes = max(0.0, max_minutes * engagement * (1.0 + rng.normal(0, 0.1)))
        read_today = minutes > 5.0
        self.last_minutes = minutes

        # ---- 2. Action-specific effects on the student state ----
        if is_book:
            repeated = genre == self.last_genre
            self.boredom += 0.25 if repeated else -0.10
            self.interest[genre] += 0.04
            for g in range(3):
                if g != genre:
                    self.interest[g] -= 0.01

            if is_stretch:
                self.motivation -= 0.05  # short-term cost
                if read_today:
                    self.level += 0.02 * self.motivation  # long-term gain
            else:
                self.motivation += 0.05 * self.interest[genre] - 0.05 * self.boredom
                if read_today:
                    self.level += 0.005
            self.last_genre = genre
            self.fatigue -= 0.05

        elif action == 6:  # peer nudge
            self.motivation += 0.10 * (1.0 - self.fatigue)
            self.fatigue += 0.20
        elif action == 7:  # badge / streak challenge
            self.motivation += 0.08 * (1.0 - self.fatigue) + 0.05 * (self.streak > 0)
            self.fatigue += 0.15
        else:  # rest
            self.motivation += 0.01
            self.boredom -= 0.10
            self.fatigue -= 0.10

        # ---- 3. Daily dynamics ----
        self.motivation += 0.03 if read_today else -0.05
        self.streak = self.streak + 1 if read_today else 0

        self.level = float(np.clip(self.level, 0.0, 1.0))
        self.interest = np.clip(self.interest, 0.0, 1.0)
        self.motivation = float(np.clip(self.motivation, 0.0, 1.0))
        self.boredom = float(np.clip(self.boredom, 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue, 0.0, 1.0))

        # ---- 4. Reward ----
        reward = minutes / 50.0                      # reading time
        reward += 0.02 * min(self.streak, 30)        # streak bonus
        reward += 10.0 * (self.level - old_level)    # level growth bonus

        # ---- 5. Termination ----
        self.low_motivation_days = (
            self.low_motivation_days + 1 if self.motivation < 0.05 else 0
        )
        dropped_out = self.low_motivation_days >= 3
        if dropped_out:
            reward -= 5.0  # penalty for losing the reader

        self.day += 1
        terminated = dropped_out
        truncated = self.day >= self.max_days

        info = {
            "minutes": minutes,
            "action_name": ACTION_NAMES[action],
            "dropped_out": dropped_out,
            "level": self.level,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .rendering import Renderer

            self._renderer = Renderer(self.render_mode)
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
