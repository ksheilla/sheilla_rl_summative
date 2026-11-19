import math
import random

import numpy as np
import pygame

W, H = 1000, 620
K = 2  # everything is drawn at 2x and scaled down, which smooths the edges

HAND = "inkfree,segoeprint,segoescript,bradleyhanditc,comicsansms,arial"
SANS = "segoeui,calibri,arial"

WALL = (164, 199, 186)
WALL_SHADE = (148, 185, 172)
PANEL_TOP = (96, 64, 42)
PANEL = (124, 84, 55)
FLOOR = (178, 122, 72)
FLOOR_LINE = (154, 100, 58)
WOOD = (110, 70, 43)
WOOD_LIGHT = (150, 102, 63)
SLATE = (40, 55, 48)
CHALK = (238, 238, 228)
CHALK_DIM = (160, 172, 164)
PAPER = (251, 247, 237)

# book spine colours for each genre: comics, thrillers, young adult
BOOK_COLORS = [
    [(232, 111, 53), (244, 175, 52), (205, 70, 60), (240, 205, 90)],
    [(34, 45, 66), (120, 32, 44), (62, 70, 82), (150, 60, 50)],
    [(70, 150, 170), (120, 100, 175), (90, 170, 140), (60, 110, 160)],
]
GENRE_NAMES = ["Comics", "Thrillers", "Young adult"]
CHALK_COLORS = [(160, 215, 160), (150, 195, 238), (244, 196, 128), (226, 160, 206)]

SAY = [
    "Try a comic",
    "Try a thriller",
    "Try a young adult book",
    "Stretch: a harder comic",
    "Stretch: a harder thriller",
    "Stretch: a harder young adult book",
    "A friend shares a book",
    "Badge challenge: keep your streak",
    "Rest day: no new book",
]

SKY_KEYS = [  # (progress, top colour, bottom colour)
    (0.00, (244, 184, 156), (255, 226, 176)),
    (0.30, (118, 178, 230), (190, 225, 245)),
    (0.70, (108, 168, 226), (198, 226, 240)),
    (1.00, (232, 134, 108), (250, 188, 120)),
]


def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def sky_colors(p):
    for (p0, t0, b0), (p1, t1, b1) in zip(SKY_KEYS, SKY_KEYS[1:]):
        if p <= p1:
            f = (p - p0) / (p1 - p0)
            return lerp(t0, t1, f), lerp(b0, b1, f)
    return SKY_KEYS[-1][1], SKY_KEYS[-1][2]


def make_books():
    shelves = []
    for g in range(3):
        rng = random.Random(40 + g)
        row = []
        for _ in range(12):
            row.append(
                {
                    "w": rng.randint(13, 22),
                    "h": rng.randint(66, 104),
                    "color": rng.choice(BOOK_COLORS[g]),
                    "band": rng.random() < 0.5,
                }
            )
        shelves.append(row)
    return shelves


class Renderer:
    def __init__(self, render_mode="human"):
        pygame.init()
        self.render_mode = render_mode
        if render_mode == "human":
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("READit Coach")
        else:
            self.screen = pygame.Surface((W, H))
        self.s = pygame.Surface((W * K, H * K))
        self.clock = pygame.time.Clock()
        self.fonts = {}
        self.books = make_books()
        self.history = []

    # ---------------- drawing helpers (logical coordinates) ----------------
    def font(self, kind, size):
        key = (kind, size)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont(HAND if kind == "hand" else SANS, int(size * K))
        return self.fonts[key]

    def text(self, s, x, y, size, color, kind="hand", anchor="tl", surf=None):
        img = self.font(kind, size).render(s, True, color)
        w, h = img.get_width() / K, img.get_height() / K
        if anchor == "tr":
            x -= w
        elif anchor == "tc":
            x -= w / 2
        elif anchor == "mc":
            x -= w / 2
            y -= h / 2
        (surf or self.s).blit(img, (int(x * K), int(y * K)))
        return w

    def R(self, color, x, y, w, h, r=0):
        pygame.draw.rect(self.s, color, (x * K, y * K, w * K, h * K), border_radius=int(r * K))

    def C(self, color, x, y, r):
        pygame.draw.circle(self.s, color, (int(x * K), int(y * K)), int(r * K))

    def E(self, color, x, y, w, h):
        pygame.draw.ellipse(self.s, color, (x * K, y * K, w * K, h * K))

    def P(self, color, pts):
        pygame.draw.polygon(self.s, color, [(int(a * K), int(b * K)) for a, b in pts])

    def L(self, color, a, b, w=2):
        pygame.draw.line(
            self.s, color, (int(a[0] * K), int(a[1] * K)), (int(b[0] * K), int(b[1] * K)), max(1, int(w * K))
        )

    def wrap(self, s, size, width):
        words, lines, cur = s.split(), [], ""
        f = self.font("hand", size)
        for w in words:
            trial = (cur + " " + w).strip()
            if f.size(trial)[0] / K <= width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    # ---------------- scene ----------------
    def room(self):
        self.R(WALL, 0, 0, W, 470)
        for x in range(0, W, 62):  # faint panel seams on the wall
            self.R(WALL_SHADE, x, 0, 2, 395)
        self.R(PANEL_TOP, 0, 392, W, 10)
        self.R(PANEL, 0, 402, W, 68)
        for x in range(0, W, 110):
            self.R(PANEL_TOP, x, 412, 3, 48)
        self.R(FLOOR, 0, 470, W, 150)
        for y in (500, 536, 578):
            self.R(FLOOR_LINE, 0, y, W, 2)
        self.R((98, 62, 38), 0, 468, W, 6)
        self.E((150, 100, 58), 383, 466, 224, 26)  # soft shadow under the chair

    def window(self, p):
        x0, y0, w, h = 40, 44, 290, 160
        top, bot = sky_colors(p)
        steps = 32
        for i in range(steps):
            self.R(lerp(top, bot, i / (steps - 1)), x0, y0 + i * h / steps, w, h / steps + 1)
        sx = x0 + 28 + p * (w - 56)
        sy = y0 + h - 34 - math.sin(math.pi * p) * (h - 78)
        self.C(lerp((255, 236, 170), (255, 250, 214), 0.5), sx, sy, 24)
        self.C((255, 224, 120), sx, sy, 17)
        yb = y0 + h
        self.P((120, 160, 112), [(x0, yb), (x0, yb - 38), (x0 + 70, yb - 54), (x0 + 150, yb - 36), (x0 + 230, yb - 50), (x0 + w, yb - 34), (x0 + w, yb)])
        self.P((84, 128, 88), [(x0, yb), (x0, yb - 20), (x0 + 90, yb - 30), (x0 + 190, yb - 18), (x0 + w, yb - 28), (x0 + w, yb)])
        # acacia tree
        tx = x0 + 214
        self.P((70, 54, 40), [(tx - 3, yb - 22), (tx + 3, yb - 22), (tx + 2, yb - 62), (tx - 2, yb - 62)])
        self.E((64, 104, 72), tx - 34, yb - 80, 70, 22)
        self.E((74, 116, 80), tx - 22, yb - 88, 44, 16)
        # frame, cross bars, sill
        for rect in [(x0 - 8, y0 - 8, w + 16, 8), (x0 - 8, y0 + h, w + 16, 8), (x0 - 8, y0, 8, h), (x0 + w, y0, 8, h)]:
            self.R(WOOD, *rect)
        self.R(WOOD, x0 + w / 2 - 3, y0, 6, h)
        self.R(WOOD, x0, y0 + 70, w, 5)
        self.R(WOOD_LIGHT, x0 - 14, y0 + h + 8, w + 28, 9, r=2)

    def chalkboard(self, env):
        self.R((60, 40, 26), 32, 240, 306, 222, r=4)
        self.R(SLATE, 40, 248, 290, 206)
        self.R((50, 66, 58), 40, 248, 290, 3)
        self.R(WOOD_LIGHT, 32, 458, 306, 8, r=2)  # chalk ledge
        self.R(CHALK, 62, 454, 18, 4, r=2)
        self.text(f"Day {env.day} of {env.max_days}", 54, 254, 26, CHALK)
        self.text(f"Read today: {env.last_minutes:.0f} min", 54, 288, 17, CHALK_DIM)
        rows = [
            ("Motivation", env.motivation),
            ("Reading level", env.level),
            ("Boredom", env.boredom),
            ("Nudge fatigue", env.fatigue),
        ]
        for i, (label, v) in enumerate(rows):
            y = 322 + i * 28
            self.text(label, 54, y - 3, 16, CHALK)
            self.R(CHALK_DIM, 176, y + 2, 140, 12, r=6)
            self.R(SLATE, 178, y + 4, 136, 8, r=4)
            self.R(CHALK_COLORS[i], 178, y + 4, max(6, 136 * float(v)), 8, r=4)
        self.text(f"Streak: {env.streak} days", 54, 428, 18, (244, 206, 120))

    def shelf(self, env):
        x0, y0, w, h = 656, 36, 328, 438
        self.R(WOOD, x0, y0, w, h, r=4)
        self.R((84, 54, 33), x0 + 12, y0 + 12, w - 24, h - 24)
        planks = [180, 318, 456]
        last = env.last_genre
        for g in range(3):
            n = int(max(0, min(12, round(float(env.interest[g]) * 12))))
            x = x0 + 22
            for i in range(n):
                b = self.books[g][i]
                lift = 14 if (g == last and i == n - 1) else 0
                self.R(b["color"], x, planks[g] - b["h"] - lift, b["w"], b["h"], r=2)
                if b["band"]:
                    self.R(lerp(b["color"], (255, 255, 255), 0.45), x, planks[g] - b["h"] - lift + 12, b["w"], 4)
                if lift:
                    self.R((212, 170, 60), x + b["w"] // 2 - 2, planks[g] - b["h"] - lift - 8, 4, 14)
                x += b["w"] + 2
            self.R(WOOD_LIGHT, x0 + 6, planks[g], w - 12, 16, r=2)
            lit = g == last
            self.R((226, 188, 100) if lit else (190, 152, 84), x0 + 22, planks[g] + 2, 86, 12, r=2)
            self.text(GENRE_NAMES[g], x0 + 28, planks[g] + 1, 11, (60, 40, 20), kind="sans")

    def reading_log(self, env):
        # keep a record of which days the student actually read
        if env.day < len(self.history):
            self.history = []
        while len(self.history) < env.day:
            self.history.append(env.last_minutes)
        px, py = 360, 36
        self.R((222, 214, 198), px + 3, py + 4, 210, 162)  # soft shadow
        self.R(PAPER, px, py, 210, 162)
        self.R((226, 196, 120), px + 8, py - 6, 34, 12)  # tape
        self.R((226, 196, 120), px + 168, py - 6, 34, 12)
        self.text("Reading log", px + 105, py + 8, 18, (70, 60, 50), anchor="tc")
        for i in range(env.max_days):
            cx, cy = px + 8 + (i % 10) * 20, py + 36 + (i // 10) * 20
            if i < len(self.history):
                m = self.history[i]
                if m < 5:
                    self.R((226, 218, 206), cx, cy, 16, 16, r=3)
                    self.L((190, 120, 100), (cx + 4, cy + 4), (cx + 12, cy + 12), 1.5)
                    self.L((190, 120, 100), (cx + 12, cy + 4), (cx + 4, cy + 12), 1.5)
                else:
                    self.R(lerp((170, 214, 170), (64, 140, 90), (m - 5) / 30), cx, cy, 16, 16, r=3)
            else:
                self.R((236, 231, 220), cx, cy, 16, 16, r=3)
        if env.day > 0:
            i = env.day - 1
            cx, cy = px + 8 + (i % 10) * 20, py + 36 + (i // 10) * 20
            pygame.draw.rect(self.s, (60, 50, 44), (cx * K - 2 * K, cy * K - 2 * K, 20 * K, 20 * K), 2 * K, border_radius=4 * K)

    def glow(self, env):
        surf = pygame.Surface((300 * K, 300 * K), pygame.SRCALPHA)
        a = int(3 + 9 * float(env.motivation))
        for r in range(140, 20, -4):
            pygame.draw.circle(surf, (255, 226, 150, a), (150 * K, 150 * K), r * K)
        self.s.blit(surf, ((372 - 150) * K, (352 - 150) * K))

    def lamp(self, env):
        self.R((78, 66, 58), 370, 366, 5, 104)
        self.R((78, 66, 58), 356, 466, 34, 8, r=4)
        lit = lerp((222, 190, 120), (255, 232, 160), float(env.motivation))
        self.P(lit, [(352, 330), (392, 330), (402, 368), (342, 368)])
        self.R((200, 160, 90), 342, 366, 60, 4, r=2)

    def rug(self):
        self.P((40, 50, 86), [(388, 478), (602, 478), (630, 584), (360, 584)])
        self.P((58, 72, 112), [(392, 480), (598, 480), (624, 580), (366, 580)])
        self.P((226, 178, 70), [(386, 500), (604, 500), (610, 512), (380, 512)])
        self.P((240, 232, 214), [(378, 520), (610, 520), (614, 528), (374, 528)])
        for i in range(11):  # zigzag band
            x = 376 + i * 22.6
            self.P((196, 84, 62), [(x, 556), (x + 11, 540), (x + 22, 556)])
        for i in range(26):  # fringe
            x = 362 + i * 10.3
            self.L((226, 214, 190), (x, 584), (x - 1.5, 592), 1.5)

    def student(self, env, cx=495):
        chair, chair_d = (206, 152, 54), (168, 118, 38)
        self.R(chair_d, cx - 82, 286, 164, 168, r=28)
        self.R(chair, cx - 72, 296, 144, 150, r=22)
        self.R(chair_d, cx - 100, 394, 30, 72, r=12)
        self.R(chair_d, cx + 70, 394, 30, 72, r=12)
        self.R(chair, cx - 86, 430, 172, 34, r=12)
        self.R((120, 84, 34), cx - 82, 464, 12, 16)
        self.R((120, 84, 34), cx + 70, 464, 12, 16)

        shirt, pants, skin, hair = (64, 98, 140), (52, 60, 82), (122, 78, 54), (28, 22, 20)
        self.R(pants, cx - 46, 414, 92, 28, r=10)
        self.R(pants, cx - 42, 436, 24, 48, r=8)
        self.R(pants, cx + 18, 436, 24, 48, r=8)
        self.R((236, 232, 222), cx - 46, 484, 34, 10, r=5)
        self.R((236, 232, 222), cx + 12, 484, 34, 10, r=5)
        self.R(shirt, cx - 34, 330, 68, 92, r=24)
        self.R(skin, cx - 8, 322, 16, 14, r=4)

        # head: hair puff, face, features
        m = float(env.motivation)
        self.C(hair, cx, 294, 33)
        self.C(skin, cx - 26, 304, 5)
        self.C(skin, cx + 26, 304, 5)
        self.C(skin, cx, 302, 26)
        self.C(hair, cx, 284, 27)
        self.E(skin, cx - 25, 286, 50, 40)
        self.C((30, 22, 20), cx - 9, 304, 3)
        self.C((30, 22, 20), cx + 9, 304, 3)
        if m > 0.55:
            pts = [(cx + t * 9, 312 + 5 * (1 - t * t)) for t in np.linspace(-1, 1, 9)]
        elif m < 0.35:
            pts = [(cx + t * 8, 319 - 4 * (1 - t * t)) for t in np.linspace(-1, 1, 9)]
        else:
            pts = [(cx - 7, 316), (cx + 7, 316)]
        for a, b in zip(pts, pts[1:]):
            self.L((40, 24, 20), a, b, 1.6)

        # book (back cover faces us) and arms
        g = env.last_genre
        if g is not None and g >= 0:
            self.R(BOOK_COLORS[g][0], cx - 34, 372, 68, 56, r=3)
            self.R(lerp(BOOK_COLORS[g][0], (0, 0, 0), 0.3), cx - 34, 372, 7, 56, r=2)
            self.R((246, 240, 226), cx - 30, 368, 62, 6, r=2)
            lh, rh = (cx - 40, 410), (cx + 40, 410)
        else:
            lh, rh = (cx - 12, 412), (cx + 12, 412)
        self.L(shirt, (cx - 28, 348), lh, 19)
        self.L(shirt, (cx + 28, 348), rh, 19)
        self.C(skin, lh[0], lh[1], 9)
        self.C(skin, rh[0], rh[1], 9)

        # sleepy z's when bored
        if env.boredom > 0.45:
            n = 1 + int(float(env.boredom) * 3)
            for i in range(min(n, 3)):
                self.text("z", cx + 36 + i * 12, 270 - i * 18, 16 + i * 5, (84, 108, 168), kind="sans")

    def note(self, env):
        a = env.last_action
        msg = SAY[a] if a is not None else "Waiting for the first move"
        size = 17 if len(msg) <= 24 else 14
        lines = self.wrap(msg, size, 100)
        nw, nh = 124, 92
        surf = pygame.Surface((nw * K, nh * K), pygame.SRCALPHA)
        pygame.draw.rect(surf, (150, 120, 40, 70), (4 * K, 5 * K, (nw - 4) * K, (nh - 5) * K))
        pygame.draw.rect(surf, (255, 228, 112), (0, 0, (nw - 4) * K, (nh - 5) * K))
        pygame.draw.rect(surf, (240, 206, 84), (0, 0, (nw - 4) * K, 10 * K))
        for i, line in enumerate(lines[:4]):
            self.text(line, 10, 18 + i * (size + 1), size, (60, 48, 30), surf=surf)
        surf = pygame.transform.rotate(surf, 3)
        self.s.blit(surf, (528 * K, 132 * K))

    # ---------------- main entry ----------------
    def draw(self, env):
        pygame.event.pump()
        p = env.day / env.max_days
        self.room()
        self.glow(env)
        self.window(p)
        self.chalkboard(env)
        self.shelf(env)
        self.reading_log(env)
        self.lamp(env)
        self.rug()
        self.student(env)
        self.note(env)

        small = pygame.transform.smoothscale(self.s, (W, H))
        self.screen.blit(small, (0, 0))
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(getattr(env, "render_fps", 5))
            return None
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        pygame.quit()