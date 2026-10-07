"""Topic 1 — Why XOR needs a hidden layer.

Render:  manim -qh s1_xor.py XorHiddenLayer
"""

import numpy as np
from manim import *

from common import *

# Hidden layer used throughout (non-singular, so the plane stretches instead
# of collapsing onto a line):
#   h1 = sigma(3 * (1.5 x1 + 1.0 x2 - 0.5))   ~ OR
#   h2 = sigma(3 * (1.0 x1 + 1.5 x2 - 2.0))   ~ AND
#   y  = step(h1 - h2 - 0.4)                  OR and not AND = XOR
W = np.array([[1.5, 1.0], [1.0, 1.5]])
C = np.array([-0.5, -2.0])
K = 3.0
W_INV = np.linalg.inv(W)

XOR_PTS = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
CLS_COLOR = {0: BLUE, 1: ORANGE}

# Display transforms (scene = O + S * coords) for each stage.
STAGE_IN = (np.array([-3.7, -1.5]), 2.4)  # input space [-0.5, 1.5]^2
STAGE_LIN = (np.array([-3.2, 0.3]), 1.2)  # after W x + c
STAGE_HID = (np.array([-4.4, -2.4]), 4.2)  # hidden space [0, 1]^2


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def pipeline(x, a, b):
    """Map input-space points x (…, 2) to scene points.

    a in [0, 1]: progress of the linear map, b in [0, 1]: progress of the squash.
    """
    z = (1 - a) * x + a * (x @ W.T + C)
    h = (1 - b) * z + b * sigmoid(K * z)
    if b > 0:
        o0, s0 = STAGE_LIN
        o1, s1 = STAGE_HID
        o, s = (1 - b) * o0 + b * o1, (1 - b) * s0 + b * s1
    else:
        o0, s0 = STAGE_IN
        o1, s1 = STAGE_LIN
        o, s = (1 - a) * o0 + a * o1, (1 - a) * s0 + a * s1
    p = o + s * h
    return np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], axis=-1)


class Warpable(VGroup):
    """A set of curves and dots defined by input-space coordinates.

    ``fn(x, *state)`` maps input coords (..., 2) to scene points.
    """

    def __init__(self, fn=pipeline, state=(0.0, 0.0)):
        super().__init__()
        self.fn = fn
        self.curves = []  # (VMobject, (n, 2) input coords)
        self.dots = []  # (Mobject, (2,) input coord)
        self.state = tuple(state)

    def add_curve(self, xs, **style):
        m = VMobject(**style)
        m.set_points_as_corners(self.fn(np.asarray(xs, float), *self.state))
        self.curves.append((m, np.asarray(xs, float)))
        self.add(m)
        return m

    def add_dot(self, x, mob):
        mob.move_to(self.fn(np.asarray(x, float), *self.state))
        self.dots.append((mob, np.asarray(x, float)))
        self.add(mob)
        return mob

    def set_state(self, *state):
        self.state = tuple(state)
        for m, xs in self.curves:
            m.set_points_as_corners(self.fn(xs, *state))
        for m, x in self.dots:
            m.move_to(self.fn(x, *state))


def warp_to(w, *target, **kw):
    s0 = np.array(w.state, float)
    s1 = np.array(target, float)

    def upd(mob, t):
        mob.set_state(*(s0 + (s1 - s0) * smooth(t)))

    return UpdateFromAlphaFunc(w, upd, **kw)


# Coda: (x1, x2) -> (x1^2, x2).  Shared display for input and feature space.
FOLD_O, FOLD_S = np.array([-4.1, -1.4]), 1.1


def fold(x, a):
    u = (1 - a) * x[..., 0] + a * x[..., 0] ** 2
    p = FOLD_O + FOLD_S * np.stack([u, x[..., 1]], -1)
    return np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], axis=-1)


PARA_PTS = [
    ((0, 0.5), 1), ((0.5, 1.0), 1), ((-0.5, 1.1), 1), ((-1, 1.5), 1), ((1, 1.2), 1), ((0.2, 2.2), 1),
    ((-1.5, 0.8), 0), ((1.5, 1.5), 0), ((-1, 0.2), 0), ((1, 0), 0), ((0.5, -0.5), 0), ((-0.4, -0.6), 0),
    ((1.7, 2.3), 0), ((-1.8, 2.0), 0),
]


def make_grid(w, lo=-0.5, hi=1.5, step=0.25, n=120):
    ts = np.linspace(lo, hi, n)
    for v in np.arange(lo, hi + 1e-9, step):
        major = abs(v - round(v)) < 1e-9
        style = dict(stroke_color=MUTED if major else GRID, stroke_width=2 if major else 1.2)
        w.add_curve(np.stack([np.full(n, v), ts], 1), **style)
        w.add_curve(np.stack([ts, np.full(n, v)], 1), **style)


def xor_dot(cls):
    if cls == 0:
        return Dot(radius=0.13, color=BLUE).set_stroke(FG, 2)
    return Square(0.24, color=ORANGE, fill_opacity=1).rotate(PI / 4).set_stroke(FG, 2)


class XorHiddenLayer(Explainer):
    def construct(self):
        self.title_card("Why XOR needs a hidden layer", "COMP2211 · MLP · appears in 6 exams")
        head = self.heading("1 · Four points, one line?")

        w = Warpable()
        make_grid(w)
        dots = {}
        for (x, cls) in XOR_PTS:
            dots[x] = w.add_dot(x, xor_dot(cls))
        o, s = STAGE_IN
        axis_lbls = VGroup(
            MathTex("x_1", font_size=34, color=MUTED).move_to([o[0] + s * 1.5 + 0.3, o[1] - 0.25, 0]),
            MathTex("x_2", font_size=34, color=MUTED).move_to([o[0] - 0.3, o[1] + s * 1.5 + 0.1, 0]),
        )
        coord_lbls = VGroup(
            *[
                MathTex(f"({x[0]},{x[1]})", font_size=26, color=MUTED).next_to(
                    pipeline(np.array(x, float), 0, 0), DL if x[0] == 0 else DR, buff=0.12
                )
                for x, _ in XOR_PTS
            ]
        )

        table = MathTex(
            r"\begin{array}{cc|c} x_1 & x_2 & y\\ \hline 0&0&0\\ 0&1&1\\ 1&0&1\\ 1&1&0 \end{array}",
            font_size=40,
        ).move_to([3.6, 1.2, 0])
        legend = VGroup(
            VGroup(xor_dot(0).scale(0.8), T("y = 0", 26)).arrange(RIGHT, buff=0.2),
            VGroup(xor_dot(1).scale(0.8), T("y = 1", 26)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(table, DOWN, buff=0.5)

        self.play(Create(VGroup(*[m for m, _ in w.curves]), lag_ratio=0.02, run_time=1.6), FadeIn(axis_lbls))
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in dots.values()], lag_ratio=0.2),
            FadeIn(coord_lbls),
            Write(table),
        )
        self.play(FadeIn(legend))
        self.wait()

        # --- try every line -------------------------------------------------
        theta = ValueTracker(0.3)
        offset = ValueTracker(0.0)
        centre = np.array([0.5, 0.5])

        def line_errors():
            n = np.array([np.cos(theta.get_value()), np.sin(theta.get_value())])
            side = [(np.dot(np.array(x) - centre, n) - offset.get_value()) > 0 for x, _ in XOR_PTS]
            labels = [c for _, c in XOR_PTS]
            wrong_a = [s != bool(c) for s, c in zip(side, labels)]
            wrong_b = [not v for v in wrong_a]
            return wrong_a if sum(wrong_a) <= sum(wrong_b) else wrong_b

        def make_line():
            n = np.array([np.cos(theta.get_value()), np.sin(theta.get_value())])
            d = np.array([-n[1], n[0]])
            p0 = centre + n * offset.get_value()
            xs = np.array([p0 - 3 * d, p0 + 3 * d])
            seg = Line(pipeline(xs[0], 0, 0), pipeline(xs[1], 0, 0), color=YELLOW, stroke_width=5)
            return seg

        clip = Rectangle(width=4.8, height=4.8).move_to(pipeline(centre, 0, 0))

        def clipped_line():
            seg = make_line()
            # keep the line inside the plot window
            a, b = seg.get_start(), seg.get_end()
            ts = np.linspace(0, 1, 400)
            pts = [a + t * (b - a) for t in ts]
            inside = [p for p in pts if abs(p[0] - clip.get_x()) <= 2.4 and abs(p[1] - clip.get_y()) <= 2.4]
            return Line(inside[0], inside[-1], color=YELLOW, stroke_width=5)

        line = always_redraw(clipped_line)

        def make_marks():
            errs = line_errors()
            return VGroup(
                *[
                    Circle(0.3, color=ROSE, stroke_width=5).move_to(pipeline(np.array(x, float), 0, 0))
                    for (x, _), e in zip(XOR_PTS, errs)
                    if e
                ]
            )

        marks = always_redraw(make_marks)

        def make_counter():
            k = sum(line_errors())
            col = ROSE if k else GREEN
            return VGroup(T("misclassified:", 28, MUTED), T(str(k), 34, col, weight="BOLD")).arrange(RIGHT).move_to(
                [3.6, -1.9, 0]
            )

        counter = always_redraw(make_counter)
        self.play(Create(line), FadeIn(marks), FadeIn(counter))
        for th, off in [(1.2, 0.2), (2.35, 0.0), (2.35, 0.55), (3.0, -0.3), (3.9, 0.45), (5.5, -0.2), (6.6, 0.1)]:
            self.play(theta.animate.set_value(th), offset.animate.set_value(off), run_time=1.3)
        self.wait(0.5)

        cap = caption("Every straight line gets at least one point wrong.", 30, YELLOW)
        self.play(FadeIn(cap))
        self.wait(1.2)

        proof = VGroup(
            MathTex(r"y = \mathrm{step}(w_1x_1 + w_2x_2 + b)", font_size=30),
            MathTex(r"(0,0)\to 0:\; b < 0", font_size=28),
            MathTex(r"(1,0),(0,1)\to 1:\; w_1 + b > 0,\; w_2 + b > 0", font_size=28),
            MathTex(r"\Rightarrow\; w_1 + w_2 + b > -b > 0", font_size=28, color=YELLOW),
            MathTex(r"\text{but } (1,1)\to 0 \text{ needs } w_1 + w_2 + b < 0 \;\; \text{--- contradiction}", font_size=28, color=ROSE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to([0.35, 0.3, 0], RIGHT, buff=0)
        self.play(FadeOut(table), FadeOut(legend), FadeOut(counter))
        for p in proof:
            self.play(FadeIn(p, shift=RIGHT * 0.2), run_time=0.8)
        self.wait(2)
        self.play(FadeOut(proof), FadeOut(cap), FadeOut(line), FadeOut(marks), FadeOut(coord_lbls))

        # --- hidden layer ----------------------------------------------------
        head = self.heading("2 · Add a hidden layer: stretch, then squash", head)
        net = self.network().move_to([3.7, 1.9, 0])
        eqs = VGroup(
            MathTex(r"z = W x + c", font_size=36),
            MathTex(
                r"W = \begin{bmatrix} 1.5 & 1 \\ 1 & 1.5 \end{bmatrix},\quad c = \begin{bmatrix} -0.5 \\ -2 \end{bmatrix}",
                font_size=30,
            ),
            MathTex(r"h = \sigma(3z)", font_size=36),
        ).arrange(DOWN, buff=0.25).next_to(net, DOWN, buff=0.45)
        self.play(FadeIn(net))
        self.play(Write(eqs[0]), FadeIn(eqs[1]))
        box = SurroundingRectangle(VGroup(eqs[0], eqs[1]), color=TEAL, buff=0.12)
        self.play(Create(box))
        self.play(warp_to(w, 1, 0, run_time=3.5), FadeOut(axis_lbls))
        self.play(FadeIn(caption("Linear map: the grid is stretched and sheared, lines stay straight.", 28, TEAL)))
        lin_cap = self.mobjects[-1]
        self.wait(1.5)

        self.play(Write(eqs[2]), box.animate.become(SurroundingRectangle(eqs[2], color=PURPLE, buff=0.12)))
        sig = self.sigmoid_plot().move_to([3.7, -2.35, 0])
        self.play(FadeIn(sig))
        self.play(FadeOut(lin_cap))
        self.play(warp_to(w, 1, 1, run_time=4.5))
        squash_cap = caption("Sigmoid squashes everything into the unit square — and bends the grid.", 28, PURPLE)
        self.play(FadeIn(squash_cap))
        self.wait(1.5)

        o3, s3 = STAGE_HID
        hid_lbls = VGroup(
            MathTex("h_1", font_size=34, color=MUTED).move_to([o3[0] + s3 + 0.35, o3[1] - 0.2, 0]),
            MathTex("h_2", font_size=34, color=MUTED).move_to([o3[0] - 0.3, o3[1] + s3 + 0.15, 0]),
        )
        self.play(FadeIn(hid_lbls))

        # --- one straight line in hidden space --------------------------------
        head = self.heading("3 · In hidden space, one line is enough", head)
        h1 = np.linspace(0.4 + 1e-6, 1 - 1e-9, 4000)
        hh = np.stack([h1, h1 - 0.4], 1)
        xb = (np.log(hh / (1 - hh)) / K - C) @ W_INV.T
        keep = np.all((xb > -0.5) & (xb < 1.5), axis=1)
        # split into contiguous runs so clipped pieces are not joined by a chord
        edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
        bound = VGroup(
            *[w.add_curve(xb[a:b], stroke_color=YELLOW, stroke_width=6) for a, b in zip(edges[::2], edges[1::2])]
        )
        w.remove(*bound)
        out_eq = MathTex(r"y = \mathrm{step}(h_1 - h_2 - 0.4)", font_size=36, color=YELLOW).move_to(eqs[2])
        self.play(FadeOut(squash_cap), FadeOut(sig), FadeOut(box))
        self.play(Create(bound, run_time=1.5), ReplacementTransform(eqs[2].copy(), out_eq), FadeOut(eqs))
        self.remove(bound)
        w.add(*bound)
        regions = VGroup(
            T("y = 1", 30, ORANGE).move_to([o3[0] + s3 * 1.17, o3[1] + s3 * 0.12, 0]),
            T("y = 0", 30, BLUE).move_to([o3[0] + s3 * 0.3, o3[1] + s3 * 0.65, 0]),
        )
        self.play(FadeIn(regions))
        self.play(*[Indicate(d, scale_factor=1.4) for d in dots.values()])
        self.play(FadeIn(caption("All four points are now linearly separable.", 30, YELLOW)))
        sep_cap = self.mobjects[-1]
        exam = VGroup(
            T("Exam version (2025 Spring Q7c, step units):", 22, MUTED),
            MathTex(r"h_1 = \mathrm{OR},\; h_2 = \mathrm{NAND},\; y = h_1\ \mathrm{AND}\ h_2", font_size=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).next_to(out_eq, DOWN, buff=0.6)
        self.play(FadeIn(exam))
        self.wait(2)

        # --- pull the line back ----------------------------------------------
        head = self.heading("4 · Pull the line back to input space", head)
        self.play(FadeOut(regions), FadeOut(hid_lbls), FadeOut(sep_cap), FadeOut(exam))
        self.play(warp_to(w, 0, 0, run_time=5))
        self.play(FadeIn(axis_lbls))
        self.play(
            FadeIn(caption("The straight hidden-space line becomes a curved boundary around the XOR = 1 points.", 26, YELLOW))
        )
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])

        self.parabola_coda(head)

    # ------------------------------------------------------------------------
    def network(self):
        pos = {
            "x1": [-1.6, 0.5],
            "x2": [-1.6, -0.5],
            "h1": [0, 0.5],
            "h2": [0, -0.5],
            "y": [1.6, 0],
        }
        nodes = {}
        for k, (x, y) in pos.items():
            c = Circle(0.3, color=FG, stroke_width=2).move_to([x, y, 0])
            lbl = MathTex(k[0] + ("_" + k[1:] if len(k) > 1 else ""), font_size=28).move_to(c)
            col = TEAL if k.startswith("h") else (YELLOW if k == "y" else MUTED)
            c.set_stroke(col)
            nodes[k] = VGroup(c, lbl)
        edges = VGroup()
        for a in ("x1", "x2"):
            for b in ("h1", "h2"):
                edges.add(Line(nodes[a][0].get_right(), nodes[b][0].get_left(), stroke_width=2, color=GRID))
        for a in ("h1", "h2"):
            edges.add(Line(nodes[a][0].get_right(), nodes["y"][0].get_left(), stroke_width=2, color=GRID))
        return VGroup(edges, *nodes.values())

    def sigmoid_plot(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1, 0.5],
            x_length=3.2,
            y_length=1.3,
            axis_config=dict(color=MUTED, stroke_width=2, include_ticks=False),
        )
        g = ax.plot(lambda z: 1 / (1 + np.exp(-3 * z)), color=PURPLE, stroke_width=4)
        step = ax.plot(lambda z: 1.0 if z > 0 else 0.0, color=MUTED, stroke_width=2, discontinuities=[0])
        step.set_stroke(opacity=0.5)
        lbl = MathTex(r"\sigma(3z) \approx \mathrm{step}(z)", font_size=26, color=PURPLE).next_to(ax, UP, buff=0.1)
        return VGroup(ax, step, g, lbl)

    def parabola_coda(self, head):
        head = self.heading("5 · Same trick with a hand-made feature: x₁²", head)
        src = T("2023 Fall midterm Q9", 24, MUTED).next_to(head, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(FadeIn(src))

        w = Warpable(fold, state=(0.0,))
        n = 120
        ts1, ts2 = np.linspace(-2, 2, n), np.linspace(-1, 3, n)
        for v in np.arange(-2, 2.01, 0.5):
            col = PURPLE if v < 0 else (MUTED if v == 0 else GRID)
            w.add_curve(np.stack([np.full(n, v), ts2], 1), stroke_color=col, stroke_width=1.6)
        for v in np.arange(-1, 3.01, 0.5):
            w.add_curve(np.stack([ts1, np.full(n, v)], 1), stroke_color=MUTED if v == 0 else GRID, stroke_width=1.4)
        dots = [w.add_dot(x, xor_dot(c).scale(0.7)) for x, c in PARA_PTS]
        xl = MathTex("x_1", font_size=32, color=MUTED).move_to(fold(np.array([2.3, -0.25]), 0))
        self.play(Create(VGroup(*[m for m, _ in w.curves]), lag_ratio=0.02), FadeIn(xl))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.06))
        cap = caption("Orange sits in the middle, blue on both sides: no straight line works.", 28)
        self.play(FadeIn(cap))
        self.wait(1.5)

        feat = MathTex(r"(x_1,\,x_2)\;\mapsto\;(x_1^2,\,x_2)", font_size=40, color=TEAL).move_to([3.7, 2.3, 0])
        self.play(Write(feat), FadeOut(cap))
        ul = MathTex("x_1^2", font_size=32, color=TEAL).move_to(fold(np.array([4.3, -0.25]), 0))
        self.play(warp_to(w, 1.0, run_time=4), ReplacementTransform(xl, ul))
        cap = caption("Squaring folds the left half (purple) onto the right.", 28, PURPLE)
        self.play(FadeIn(cap))
        self.wait(1.2)

        xs = np.linspace(-1.8, 1.8, 400)
        bound = w.add_curve(np.stack([xs, xs**2 - 0.25], 1), stroke_color=YELLOW, stroke_width=6)
        w.remove(bound)
        self.play(Create(bound), FadeOut(cap))
        w.add(bound)
        self.play(FadeIn(caption("Now a straight line separates them.", 28, YELLOW)))
        cap = self.mobjects[-1]
        self.wait(1.2)
        xl = MathTex("x_1", font_size=32, color=MUTED).move_to(fold(np.array([2.3, -0.25]), 0))
        self.play(FadeOut(cap), warp_to(w, 0.0, run_time=4), ReplacementTransform(ul, xl))

        rule = VGroup(
            MathTex(r"w_1x_1 + w_2x_1^2 + w_3x_2 + \theta = 0", font_size=34),
            MathTex(r"\Rightarrow\; x_2 = -\tfrac{w_2}{w_3}x_1^2 - \tfrac{w_1}{w_3}x_1 - \tfrac{\theta}{w_3}", font_size=34, color=YELLOW),
            T("linear in the features, a parabola in (x₁, x₂)", 24, MUTED),
        ).arrange(DOWN, buff=0.3).move_to([3.7, 0.6, 0])
        for r in rule:
            self.play(FadeIn(r, shift=UP * 0.15))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])

        # 2025 Spring Q6: the lift has to be the right one.
        head = self.heading("Careful: a lift only helps if it is the right one", head)
        src = T("2025 Spring midterm Q6", 24, MUTED).next_to(head, DOWN, aligned_edge=LEFT, buff=0.15)
        lift = MathTex(r"(x_1, x_2)\;\mapsto\;(x_1,\,x_2,\,x_1^2 + x_2^2)", font_size=38).move_to([0, 1.9, 0])
        self.play(FadeIn(src), Write(lift))
        line = NumberLine(x_range=[0, 20, 2], length=11, color=MUTED, include_numbers=True, font_size=24).move_to([0, -0.2, 0])
        lbl = MathTex("x_1^2 + x_2^2", font_size=30, color=MUTED).next_to(line, RIGHT, buff=0.2).shift(UP * 0.4)
        self.play(Create(line), FadeIn(lbl))
        pos = [((1, 1), 2), ((2, 2), 8), ((3, 3), 18)]
        neg = [((1, 3), 10), ((2, 1), 5), ((3, 2), 13)]
        marks = VGroup()
        for (p, z), up in [(q, True) for q in pos] + [(q, False) for q in neg]:
            d = xor_dot(1 if up else 0).scale(0.8).move_to(line.n2p(z) + (UP if up else DOWN) * 0.55)
            t = MathTex(f"({p[0]},{p[1]})", font_size=24, color=ORANGE if up else BLUE).next_to(d, UP if up else DOWN, buff=0.12)
            marks.add(VGroup(d, t))
        self.play(LaggedStart(*[FadeIn(m, shift=DOWN * 0.2) for m in marks], lag_ratio=0.25))
        cap = VGroup(
            T("y = +1 and y = −1 interleave on the new axis (2 < 5 < 8 < 10 < 13 < 18) —", 26),
            T("and no plane in the 3-D space separates them either. The claim is invalid.", 26, ROSE),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(cap))
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])
        end = VGroup(
            T("Hidden layer = learned feature map", 40, YELLOW),
            T("stretch (W x + c) → squash (σ) → one straight line", 30, FG),
        ).arrange(DOWN, buff=0.3)
        self.play(FadeIn(end, shift=UP * 0.2))
        self.wait(2.5)
