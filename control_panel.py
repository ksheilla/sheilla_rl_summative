import time
from collections import Counter

import pygame

from environment import ACTION_NAMES, ReadCoachEnv
from environment.rendering import HAND, SANS, SAY, H, W, Renderer

PW = 260  # width of the control board on the right
ORDER = ["dqn", "ppo", "a2c", "reinforce"]
SPEEDS = [1, 2, 3, 5, 10]  # days per second

WOOD_DARK = (66, 45, 31)
WOOD = (110, 70, 43)
WOOD_LIGHT = (150, 102, 63)
BRASS = (226, 188, 100)
CREAM = (246, 240, 226)
MUTED = (206, 184, 150)
INK = (60, 42, 28)
GREEN = (74, 138, 92)


def shade(color, amount):
    return tuple(max(0, min(255, c + amount)) for c in color)


class ControlPanel:
    def __init__(self, load_agent, choose_action, best_runs, seed=7, fps=2):
        pygame.init()
        self.screen = pygame.display.set_mode((W + PW, H))
        pygame.display.set_caption("READit Coach")
        self.renderer = Renderer("rgb_array")
        self.env = ReadCoachEnv()
        self.load_agent = load_agent
        self.choose_action = choose_action
        self.best_runs = best_runs
        self.base_seed = seed
        self.student = 1
        self.speed_idx = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - fps))
        self.fonts = {}
        self.nets = {}
        self.auto = False
        self.acc = 0.0
        self.finished_at = None
        self.algo, self.net, self.run_name = None, None, ""
        self.obs, self.total, self.last_reward = None, 0.0, 0.0
        self.done, self.summary, self.counts = False, None, Counter()

        x0 = W + 14
        self.model_rects = {
            a: pygame.Rect(x0 + (i % 2) * 120, 88 + (i // 2) * 42, 112, 34)
            for i, a in enumerate(ORDER)
        }
        self.rect_auto = pygame.Rect(x0, 206, 232, 46)
        self.rect_step = pygame.Rect(x0, 262, 112, 36)
        self.rect_reset = pygame.Rect(x0 + 120, 262, 112, 36)
        self.rect_speed = pygame.Rect(x0, 308, 232, 34)

        self.select_model("dqn")

    # ---------------- state ----------------
    def select_model(self, algo):
        self.run_name = self.best_runs[algo]
        if algo not in self.nets:
            self.nets[algo] = self.load_agent(algo, self.run_name)
        self.net, self.algo = self.nets[algo], algo
        print(f"\nModel selected: {algo.upper()} ({self.run_name})", flush=True)
        self.reset(new_student=False)  # same student, so models can be compared

    def reset(self, new_student=True):
        if new_student:
            self.student += 1
        seed = self.base_seed + self.student - 1
        self.obs, _ = self.env.reset(seed=seed)
        self.total, self.last_reward = 0.0, 0.0
        self.done, self.summary, self.counts = False, None, Counter()
        self.finished_at, self.acc = None, 0.0
        print(f"=== Student {self.student} (seed {seed}) | model {self.algo.upper()} ===", flush=True)
        self.renderer.draw(self.env)

    def step(self):
        if self.done:
            return
        a = self.choose_action(self.net, self.obs)
        self.obs, r, term, trunc, info = self.env.step(a)
        self.total += r
        self.last_reward = r
        self.counts[ACTION_NAMES[a]] += 1
        self.done = term or trunc
        print(
            f"Day {self.env.day:2d} | {ACTION_NAMES[a]:<19} | read {info['minutes']:5.1f} min | "
            f"reward {r:5.2f} | total {self.total:6.1f} | motivation {self.env.motivation:.2f} | "
            f"level {self.env.level:.2f} | streak {self.env.streak}",
            flush=True,
        )
        if self.done:
            self.summary = (info["level"], info["dropped_out"])
            self.finished_at = time.time()
            print(
                f"Student {self.student}: total reward={self.total:.1f}  days={self.env.day}  "
                f"final reading level={info['level']:.2f}  dropped out={info['dropped_out']}",
                flush=True,
            )
            print("Actions used:", dict(self.counts.most_common()), flush=True)
        self.renderer.draw(self.env)

    # ---------------- input ----------------
    def click(self, pos):
        for algo, rect in self.model_rects.items():
            if rect.collidepoint(pos) and algo != self.algo:
                self.select_model(algo)
        if self.rect_auto.collidepoint(pos):
            self.auto = not self.auto
        elif self.rect_step.collidepoint(pos) and not self.auto:
            self.step()
        elif self.rect_reset.collidepoint(pos):
            self.reset(new_student=True)
        elif self.rect_speed.collidepoint(pos):
            self.speed_idx = (self.speed_idx + 1) % len(SPEEDS)

    def key(self, k):
        if k == pygame.K_SPACE:
            self.auto = not self.auto
        elif k == pygame.K_n and not self.auto:
            self.step()
        elif k == pygame.K_r:
            self.reset(new_student=True)
        elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            algo = ORDER[k - pygame.K_1]
            if algo != self.algo:
                self.select_model(algo)
        elif k == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    # ---------------- drawing ----------------
    def font(self, kind, size):
        key = (kind, size)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont(HAND if kind == "hand" else SANS, size)
        return self.fonts[key]

    def text(self, s, x, y, size, color, kind="hand", anchor="tl"):
        img = self.font(kind, size).render(s, True, color)
        if anchor == "mc":
            x, y = x - img.get_width() / 2, y - img.get_height() / 2
        self.screen.blit(img, (int(x), int(y)))
        return img.get_width()

    def button(self, rect, label, fill, fg, size=17):
        hover = rect.collidepoint(pygame.mouse.get_pos())
        fill = shade(fill, 14) if hover else fill
        pygame.draw.rect(self.screen, shade(fill, -45), rect.move(0, 3), border_radius=8)
        pygame.draw.rect(self.screen, fill, rect, border_radius=8)
        self.text(label, rect.centerx, rect.centery, size, fg, anchor="mc")

    def wrap(self, s, size, width):
        words, lines, cur = s.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if self.font("hand", size).size(trial)[0] <= width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def draw_panel(self):
        s = self.screen
        s.blit(self.renderer.screen, (0, 0))
        pygame.draw.rect(s, WOOD_DARK, (W, 0, PW, H))
        pygame.draw.rect(s, WOOD, (W, 0, 6, H))
        x0 = W + 14

        self.text("Coach controls", x0, 18, 28, CREAM)
        self.text("Model", x0, 62, 16, MUTED)
        for algo, rect in self.model_rects.items():
            on = algo == self.algo
            self.button(rect, algo.upper(), BRASS if on else WOOD_LIGHT, INK if on else CREAM, 16)
        self.text(f"Loaded: {self.run_name}", x0, 176, 12, MUTED, kind="sans")

        self.button(
            self.rect_auto,
            "Pause" if self.auto else "Auto run",
            BRASS if self.auto else GREEN,
            INK if self.auto else CREAM,
            22,
        )
        self.button(self.rect_step, "Single step", WOOD_LIGHT, CREAM, 16)
        self.button(self.rect_reset, "New student", WOOD_LIGHT, CREAM, 16)
        self.button(self.rect_speed, f"Speed: {SPEEDS[self.speed_idx]} days/s", WOOD_LIGHT, CREAM, 16)

        # status card
        card = pygame.Rect(x0, 356, 232, 200)
        pygame.draw.rect(s, CREAM, card, border_radius=6)
        self.text(f"Student {self.student}", card.x + 14, card.y + 10, 20, INK)
        self.text(f"Day {self.env.day} of {self.env.max_days}", card.x + 14, card.y + 38, 16, INK)
        self.text("Total reward", card.x + 14, card.y + 66, 14, (120, 96, 70))
        self.text(f"{self.total:.1f}", card.x + 14, card.y + 80, 40, (46, 110, 70))
        self.text(f"Today: {self.last_reward:+.2f}", card.x + 14, card.y + 130, 16, INK)
        if self.done:
            level, dropped = self.summary
            msg = "Dropped out." if dropped else "Finished, no dropout."
            self.text(msg, card.x + 14, card.y + 154, 15, (160, 60, 50) if dropped else (46, 110, 70))
            self.text(f"Reading level {level:.2f}", card.x + 14, card.y + 172, 15, INK)
        else:
            a = self.env.last_action
            msg = SAY[a] if a is not None else "Press Auto run"
            for i, line in enumerate(self.wrap(msg, 15, 204)[:2]):
                self.text(line, card.x + 14, card.y + 154 + i * 18, 15, INK)

        self.text("Space: auto / pause    N: single step", x0, 572, 12, MUTED, kind="sans")
        self.text("R: new student    1-4: pick model", x0, 590, 12, MUTED, kind="sans")

    # ---------------- main loop ----------------
    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(30) / 1000.0
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.click(e.pos)
                elif e.type == pygame.KEYDOWN:
                    self.key(e.key)
            if self.auto:
                if not self.done:
                    self.acc += dt
                    if self.acc >= 1.0 / SPEEDS[self.speed_idx]:
                        self.acc = 0.0
                        self.step()
                elif self.finished_at is None or time.time() - self.finished_at > 1.5:
                    self.reset(new_student=True)  # keep going with the next student
            self.draw_panel()
            pygame.display.flip()
        pygame.quit()


def run_panel(load_agent, choose_action, best_runs, seed=7, fps=2):
    ControlPanel(load_agent, choose_action, best_runs, seed, fps).run()