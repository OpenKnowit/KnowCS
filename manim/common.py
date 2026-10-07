"""Shared palette and helpers for the KnowCS Manim explainers.

Colours follow the site's slate/blue Tailwind palette so the videos sit
naturally next to the interactive modules.
"""

from manim import (
    DOWN,
    LEFT,
    UL,
    UP,
    FadeIn,
    FadeOut,
    MathTex,
    Scene,
    Tex,
    Text,
    VGroup,
    config,
)

BG = "#0f172a"  # slate-900
FG = "#e2e8f0"  # slate-200
MUTED = "#94a3b8"  # slate-400
GRID = "#334155"  # slate-700
BLUE = "#60a5fa"  # blue-400
ORANGE = "#fb923c"  # orange-400
ROSE = "#fb7185"  # rose-400
GREEN = "#4ade80"  # green-400
YELLOW = "#facc15"  # yellow-400
PURPLE = "#c084fc"  # purple-400
TEAL = "#2dd4bf"  # teal-400

FONT = "Helvetica Neue"
MONO = "Menlo"

config.background_color = BG

MathTex.set_default(color=FG)
Tex.set_default(color=FG)
Text.set_default(color=FG, font=FONT)


def T(text, size=32, color=FG, **kw):
    return Text(text, font_size=size, color=color, **kw)


def code(text, size=24, color=FG, **kw):
    return Text(text, font=MONO, font_size=size, color=color, **kw)


class Explainer(Scene):
    """Scene with a title card and a corner heading helper."""

    def title_card(self, title, subtitle):
        t = T(title, 48, weight="BOLD")
        s = T(subtitle, 26, MUTED).next_to(t, DOWN, buff=0.35)
        g = VGroup(t, s)
        self.play(FadeIn(g, shift=UP * 0.2))
        self.wait(1.4)
        self.play(FadeOut(g))

    def heading(self, text, old=None):
        h = T(text, 30, MUTED).to_corner(UL, buff=0.4)
        if old is None:
            self.play(FadeIn(h, shift=DOWN * 0.1))
        else:
            self.play(FadeOut(old, shift=UP * 0.1), FadeIn(h, shift=UP * 0.1))
        return h


def caption(text, size=28, color=FG):
    return T(text, size, color).to_edge(DOWN, buff=0.45)

