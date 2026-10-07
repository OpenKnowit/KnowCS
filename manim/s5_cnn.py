"""Topic 5 — From convolution to CNN shapes and parameter counts.

Render:  manim -qh s5_cnn.py ConvToCnn

Examples: lecture 8 (7x7 image, 3x3 kernel, strides 1/2/3, padding 1;
32x32x3 with ten 5x5x3 kernels) and 2022 Spring final Part B Problem 1
(CNN 2,071,656 vs MLP 150,529,000 parameters).
"""

import numpy as np
from manim import *

from common import *

CELL = 0.46


def grid(n, cell=CELL, pad=0, color=GRID):
    """(n + 2 pad)^2 squares; returns VGroup indexed [r * size + c]."""
    size = n + 2 * pad
    g = VGroup()
    for r in range(size):
        for c in range(size):
            inner = pad <= r < pad + n and pad <= c < pad + n
            sq = Square(cell, stroke_width=1.5, stroke_color=MUTED if inner else GRID)
            sq.set_fill(BLUE if inner else BG, opacity=0.25 if inner else 0)
            if not inner:
                sq = DashedVMobject(sq, num_dashes=12)
            sq.move_to([c * cell, -r * cell, 0])
            g.add(sq)
    g.size = size
    return g


def cell_at(g, r, c):
    return g[r * g.size + c]


class ConvToCnn(Explainer):
    def setting(self, n, k, p, s):
        t = MathTex(
            rf"N = {n},\; K = {k},\; P = {p},\; S = {s}",
            font_size=36,
        ).to_edge(UP, buff=1.0)
        return t

    def run_conv(self, n, k, p, s, head, fast_after=1, out_color=TEAL, overflow=False, note=None):
        size = n + 2 * p
        inp = grid(n, pad=p).move_to([-3.4, -0.5, 0])
        zeros = VGroup()
        if p:
            for r in range(size):
                for c in range(size):
                    if not (p <= r < p + n and p <= c < p + n):
                        zeros.add(T("0", 16, GRID).move_to(cell_at(inp, r, c)))
        span = size - k
        n_out = span // s + 1
        out = grid(n_out).move_to([3.2, -0.5, 0])
        for sq in out:
            sq.set_fill(out_color, 0).set_stroke(MUTED)
        st = self.setting(n, k, p, s)
        lbl_in = T(f"input {n}×{n}" + (f" + padding {p}" if p else ""), 24, MUTED).next_to(inp, DOWN, buff=0.25)
        lbl_out = T(f"output {n_out}×{n_out}", 24, out_color).next_to(out, DOWN, buff=0.25)
        self.play(FadeIn(inp), FadeIn(zeros), FadeIn(st), FadeIn(out), FadeIn(lbl_in))

        def kbox(r, c, col=YELLOW):
            tl = cell_at(inp, r, c).get_corner(UL)
            return Square(k * CELL, color=col, stroke_width=5).move_to(tl + np.array([k * CELL / 2, -k * CELL / 2, 0]))

        ker = kbox(0, 0)
        self.play(Create(ker))
        for i in range(n_out):
            for j in range(n_out):
                target = kbox(i * s, j * s)
                rt = 0.45 if i < fast_after else 0.16
                o = out[i * n_out + j]
                self.play(ker.animate.move_to(target), run_time=rt)
                self.play(o.animate.set_fill(out_color, 0.75), run_time=rt * 0.6)
        if overflow:
            bad = kbox(0, n_out * s, ROSE)
            self.play(ker.animate.move_to(kbox(0, 0)), run_time=0.4)
            self.play(Transform(ker, bad), run_time=0.8)
            x = T("does not fit!", 26, ROSE).next_to(bad, UP, buff=0.1)
            self.play(FadeIn(x))
            self.wait(1.2)
            self.play(FadeOut(x))
        self.play(FadeIn(lbl_out))
        if note:
            self.play(FadeIn(note))
        self.wait(1.2)
        return VGroup(inp, zeros, out, ker, st, lbl_in, lbl_out, *( [note] if note else [] ))

    def construct(self):
        self.title_card("From convolution to CNN shapes", "COMP2211 · CNN · appears in 3 of 3 finals")
        head = self.heading("1 · Slide a kernel, watch the output grow or shrink")

        g = self.run_conv(7, 3, 0, 1, head, fast_after=1)
        self.play(FadeOut(g))
        g = self.run_conv(7, 3, 0, 2, head, fast_after=3)
        self.play(FadeOut(g))
        note = MathTex(r"(7 - 3)/3 + 1 = 2.33", font_size=32, color=ROSE).to_edge(DOWN, buff=0.4)
        g = self.run_conv(7, 3, 0, 3, head, fast_after=3, out_color=ORANGE, overflow=True, note=note)
        self.play(FadeOut(g))
        g = self.run_conv(7, 3, 1, 1, head, fast_after=1, note=T("padding (K − 1)/2 keeps the size: “same”", 26, GREEN).to_edge(DOWN, buff=0.4))
        self.play(FadeOut(g))

        self.read_formula(head)

    # ---------------------------------------------------------------------------
    def read_formula(self, head):
        head = self.heading("2 · Read the formula off one row", head)
        n, k, p, s = 7, 3, 1, 2
        size = n + 2 * p
        row = VGroup()
        for c in range(size):
            inner = p <= c < p + n
            sq = Square(0.62, stroke_width=2, stroke_color=MUTED if inner else GRID).set_fill(BLUE if inner else BG, 0.25 if inner else 0)
            if not inner:
                sq = DashedVMobject(sq, num_dashes=12)
            row.add(sq)
        row.arrange(RIGHT, buff=0).move_to([0, 0.2, 0])
        st = self.setting(n, k, p, s)
        self.play(FadeIn(row), FadeIn(st))
        b_all = Brace(row, DOWN, color=MUTED)
        t_all = MathTex(r"N + 2P = 9", font_size=32, color=MUTED).next_to(b_all, DOWN, buff=0.1)
        self.play(GrowFromCenter(b_all), FadeIn(t_all))

        def kb(c, col=YELLOW):
            return Rectangle(width=k * 0.62, height=0.62, color=col, stroke_width=5).move_to(VGroup(*row[c : c + k]))

        ker = kb(0)
        self.play(Create(ker))
        b_k = Brace(ker, UP, color=YELLOW)
        t_k = MathTex("K = 3", font_size=30, color=YELLOW).next_to(b_k, UP, buff=0.1)
        self.play(GrowFromCenter(b_k), FadeIn(t_k))
        self.wait(0.6)
        self.play(FadeOut(b_k), FadeOut(t_k))

        # room the kernel's left edge can travel
        start = row[0].get_left()
        last_c = (size - k) // s * s
        room_line = Line(row[0].get_left() + DOWN * 0.0, row[size - k].get_left(), color=TEAL, stroke_width=6).shift(UP * 0.55)
        t_room = MathTex(r"\text{room} = N + 2P - K = 6", font_size=32, color=TEAL).next_to(room_line, UP, buff=0.15)
        self.play(Create(room_line), FadeIn(t_room))
        hops = VGroup()
        stops = list(range(0, size - k + 1, s))
        for a, b in zip(stops, stops[1:]):
            arc = CurvedArrow(row[a].get_left() + DOWN * 0.45, row[b].get_left() + DOWN * 0.45, angle=PI / 2.5, color=ORANGE, tip_length=0.15)
            hops.add(arc)
        b_all_grp = VGroup(b_all, t_all)
        self.play(b_all_grp.animate.shift(DOWN * 0.7))
        count = Integer(1, font_size=40, color=GREEN).to_edge(RIGHT, buff=1.2).shift(DOWN * 0.3)
        count_lbl = T("positions:", 26, GREEN).next_to(count, LEFT)
        self.play(FadeIn(count), FadeIn(count_lbl))
        for h, c in zip(hops, stops[1:]):
            self.play(Create(h), ker.animate.move_to(kb(c)), count.animate.set_value(count.get_value() + 1), run_time=0.8)
        t_hops = MathTex(r"\text{hops} = 6 / S = 3", font_size=32, color=ORANGE).next_to(hops, DOWN, buff=0.1)
        self.play(FadeIn(t_hops))
        self.wait(0.6)

        f = MathTex(r"\text{output}", "=", r"\Big\lfloor", r"\frac{N - K + 2P}{S}", r"\Big\rfloor", "+", "1", font_size=48).to_edge(DOWN, buff=0.5)
        f[3].set_color(TEAL)
        f[6].set_color(GREEN)
        side = VGroup(
            T("room ÷ stride = number of hops", 22, TEAL),
            T("+1 for the starting position", 22, GREEN),
            T("floor: a partial hop does not fit", 22, ROSE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(f, RIGHT, buff=0.4)
        f_grp = VGroup(f, side)
        f_grp.move_to([0, -3.1, 0])
        self.play(Write(f[0:2]), TransformFromCopy(t_room, f[3]), FadeIn(side[0]))
        self.play(TransformFromCopy(count, f[5:7]), FadeIn(side[1]))
        self.play(FadeIn(f[2]), FadeIn(f[4]), FadeIn(side[2]))
        self.wait(1)
        chk = MathTex(r"\lfloor (7 - 3 + 2)/2 \rfloor + 1 = 4\ \checkmark", font_size=32, color=GREEN).next_to(count, DOWN, buff=0.3).shift(LEFT * 1.2)
        self.play(FadeIn(chk))
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])
        self.params(head)

    # ---------------------------------------------------------------------------
    def params(self, head):
        head = self.heading("3 · Count parameters: shared kernel vs dense layer", head)

        # a 3x3xC kernel drawn as stacked slices
        C = 3
        slices = VGroup()
        cols = [ROSE, GREEN, BLUE]
        for ch in range(C):
            s = VGroup(*[Square(0.34, stroke_width=1.5, stroke_color=FG).set_fill(cols[ch], 0.55) for _ in range(9)])
            s.arrange_in_grid(3, 3, buff=0)
            s.shift(np.array([0.22, 0.22, 0]) * (C - 1 - ch))
            slices.add(s)
        slices.move_to([-4.6, 1.3, 0])
        kt = MathTex(r"3 \times 3 \times C", font_size=34).next_to(slices, DOWN, buff=0.25)
        kp = MathTex(r"= 27 \text{ weights} + 1 \text{ bias} = 28", font_size=30, color=YELLOW).next_to(kt, DOWN, buff=0.15)
        self.play(LaggedStart(*[FadeIn(s) for s in slices], lag_ratio=0.3), FadeIn(kt))
        self.play(FadeIn(kp))

        img = grid(6, cell=0.36).move_to([-0.2, 0.9, 0])
        self.play(FadeIn(img))
        stamp = Square(3 * 0.36, color=YELLOW, stroke_width=4)
        anchors = [(r, c) for r in range(4) for c in range(4)]

        def at(r, c):
            tl = cell_at(img, r, c).get_corner(UL)
            return tl + np.array([1.5 * 0.36, -1.5 * 0.36, 0])

        stamp.move_to(at(0, 0))
        self.play(TransformFromCopy(slices[-1], stamp))
        for r, c in anchors[1:]:
            self.play(stamp.animate.move_to(at(r, c)), run_time=0.12)
        shared = T("the same 28 numbers at every position", 24, YELLOW).next_to(img, DOWN, buff=0.25)
        self.play(FadeIn(shared))

        # dense: every output neuron sees every pixel
        dense_in = VGroup(*[Dot(radius=0.05, color=BLUE) for _ in range(16)]).arrange(DOWN, buff=0.12).move_to([3.0, 0.9, 0])
        dense_out = VGroup(*[Dot(radius=0.05, color=TEAL) for _ in range(8)]).arrange(DOWN, buff=0.3).move_to([5.2, 0.9, 0])
        lines = VGroup(*[Line(a.get_center(), b.get_center(), stroke_width=0.8, color=MUTED, stroke_opacity=0.6) for a in dense_in for b in dense_out])
        self.play(FadeIn(dense_in), FadeIn(dense_out))
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.004, run_time=2))
        dl = T("dense: one weight per\n(input, output) pair", 22, ROSE).next_to(VGroup(dense_in, dense_out), DOWN, buff=0.25)
        self.play(FadeIn(dl))
        self.wait(1)

        # lecture example
        ex = VGroup(
            T("Lecture: 32×32×3 input, ten 5×5×3 kernels, stride 1, pad 2 → 32×32×10", 24, FG),
            MathTex(r"\text{conv: } (5\cdot5\cdot3 + 1)\times 10 = 760", font_size=34, color=GREEN),
            MathTex(r"\text{dense, same shapes: } (32\cdot32\cdot3)(32\cdot32\cdot10) + 10240 = 31{,}467{,}520", font_size=34, color=ROSE),
        ).arrange(DOWN, buff=0.22).to_edge(DOWN, buff=0.45)
        for e in ex:
            self.play(FadeIn(e, shift=UP * 0.1))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])
        self.final_2022(head)

    # ---------------------------------------------------------------------------
    def final_2022(self, head):
        head = self.heading("4 · 2022 Spring final: CNN vs MLP on a 224×224×3 image", head)
        rows = [
            ("7×7 conv, 64, s2, p3", "224×224×3 → 112×112×64", "9,472"),
            ("3×3 max pool, s2, p1", "→ 56×56×64", "0"),
            ("3×3 conv, 128, s2", "→ 27×27×128", "73,856"),
            ("3×3 conv, 256, s2, p1", "→ 14×14×256", "295,168"),
            ("3×3 conv, 512, s2, p1", "→ 7×7×512", "1,180,160"),
            ("global pool + FC 512 → 1000", "→ 1000", "513,000"),
        ]
        tbl = VGroup()
        for a, b, c in rows:
            tbl.add(VGroup(T(a, 22, FG), T(b, 22, MUTED), T(c, 22, YELLOW)))
        for r in tbl:
            r[0].move_to([-6.4, 0, 0], aligned_edge=LEFT)
            r[1].move_to([-2.8, 0, 0], aligned_edge=LEFT)
            r[2].move_to([2.3, 0, 0], aligned_edge=RIGHT)
        tbl.arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([-1.0, 1.0, 0])
        for r in tbl:
            r[0].set_x(-6.4 + r[0].width / 2)
            r[1].set_x(-2.8 + r[1].width / 2)
            r[2].set_x(2.3 - r[2].width / 2)
        hdr = VGroup(T("layer", 20, MUTED), T("shape", 20, MUTED), T("params", 20, MUTED)).next_to(tbl, UP, buff=0.2)
        hdr[0].set_x(-6.4 + hdr[0].width / 2)
        hdr[1].set_x(-2.8 + hdr[1].width / 2)
        hdr[2].set_x(2.3 - hdr[2].width / 2)
        self.play(FadeIn(hdr))
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.1) for r in tbl], lag_ratio=0.3), run_time=3)

        floor_chk = VGroup(
            MathTex(r"\lfloor (224 - 7 + 6)/2 \rfloor + 1 = 112", font_size=28, color=TEAL),
            MathTex(r"\lfloor (56 - 3 + 0)/2 \rfloor + 1 = 27", font_size=28, color=TEAL),
            T("both need the floor", 20, TEAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([5.0, 1.0, 0])
        self.play(FadeIn(floor_chk))
        self.play(Indicate(tbl[0][1], color=TEAL), Indicate(tbl[2][1], color=TEAL))
        tot = MathTex(r"\text{CNN total} = 2{,}071{,}656", font_size=34, color=GREEN).next_to(tbl, DOWN, buff=0.3).align_to(tbl, LEFT)
        mlp = MathTex(r"\text{MLP: } 150{,}528 \to 1000 \Rightarrow 150{,}528 \times 1000 + 1000 = 150{,}529{,}000", font_size=30, color=ROSE).next_to(tot, DOWN, buff=0.2).align_to(tbl, LEFT)
        self.play(Write(tot))
        self.play(Write(mlp))
        self.wait(1.5)
        self.play(FadeOut(VGroup(tbl, hdr, floor_chk)), VGroup(tot, mlp).animate.to_edge(UP, buff=1.0).set_x(0))

        # bars on a linear scale
        full = 11.0
        base = np.array([-5.5, -0.6, 0])
        bar_m = Rectangle(width=full, height=0.6, stroke_width=0).set_fill(ROSE, 0.85).move_to(base + RIGHT * full / 2 + DOWN * 0.6)
        bar_c = Rectangle(width=full * 2.071656 / 150.529, height=0.6, stroke_width=0).set_fill(GREEN, 0.95)
        bar_c.move_to(base + RIGHT * bar_c.width / 2 + UP * 0.6)
        lc = T("CNN  2.07 M", 26, GREEN).next_to(bar_c, RIGHT, buff=0.2)
        lm = T("MLP  150.5 M", 26, BG, weight="BOLD").move_to(bar_m)
        self.play(GrowFromEdge(bar_m, LEFT), FadeIn(lm), run_time=1.5)
        self.play(GrowFromEdge(bar_c, LEFT), FadeIn(lc), run_time=1.0)
        ratio = T("≈ 73× fewer parameters — because kernels are shared, not one weight per pixel.", 26, YELLOW).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(ratio))
        self.wait(3)
