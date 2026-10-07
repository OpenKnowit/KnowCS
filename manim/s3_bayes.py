"""Topic 3 — Bayes and the base rate (lecture 2 virus test), then Naive Bayes and Laplace smoothing.

Render:  manim -qh s3_bayes.py BayesBaseRate
"""

from fractions import Fraction as Fr
from math import prod

import numpy as np
from manim import *

from common import *

GRID_N = 100  # 100 x 100 = 10,000 people
CELL_PX = 8
SIDE = 6.4  # scene units for the whole grid
CENTER = np.array([-3.3, -0.25, 0])
CELL = SIDE / GRID_N

rng = np.random.default_rng(2211)
INFECTED = rng.choice(GRID_N * GRID_N, 10, replace=False)
HEALTHY = np.setdiff1d(np.arange(GRID_N * GRID_N), INFECTED)
FALSE_POS = rng.choice(HEALTHY, 500, replace=False)  # 5% of 9,990 ≈ 499.5


def people_image(color=(100, 116, 139)):
    """All 10,000 people as one bitmap (much faster than 10,000 Dots)."""
    yy, xx = np.mgrid[0:CELL_PX, 0:CELL_PX] + 0.5
    disk = ((xx - CELL_PX / 2) ** 2 + (yy - CELL_PX / 2) ** 2) <= (CELL_PX * 0.33) ** 2
    tile = np.zeros((CELL_PX, CELL_PX, 4), np.uint8)
    tile[disk] = (*color, 255)
    img = np.tile(tile, (GRID_N, GRID_N, 1))
    m = ImageMobject(img)
    m.height = SIDE
    m.move_to(CENTER)
    return m


def cell_pos(idx):
    r, c = divmod(int(idx), GRID_N)
    return CENTER + np.array([(c - (GRID_N - 1) / 2) * CELL, ((GRID_N - 1) / 2 - r) * CELL, 0])


def person(idx, color):
    return Dot(cell_pos(idx), radius=CELL * (0.75 if color == ROSE else 0.42), color=color)


class BayesBaseRate(Explainer):
    def construct(self):
        self.title_card("Bayes and the base rate", "COMP2211 · Naive Bayes · appears in 8 of 9 exams")
        head = self.heading("1 · The virus test (lecture 2)")

        facts = VGroup(
            MathTex(r"P(V) = 0.1\%", font_size=36, color=ROSE),
            MathTex(r"P(+ \mid V) = 99\%", font_size=36),
            MathTex(r"P(- \mid H) = 95\%", font_size=36),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        notes = VGroup(
            T("have the virus", 24, MUTED),
            T("sensitivity", 24, MUTED),
            T("specificity", 24, MUTED),
        )
        for n, f in zip(notes, facts):
            n.next_to(f, RIGHT, buff=0.35)
        fact_grp = VGroup(facts, notes).move_to([3.6, 2.0, 0])
        self.play(FadeIn(fact_grp, shift=LEFT * 0.2))

        grid = people_image()
        count = T("10,000 people", 34).move_to([3.6, 0.3, 0])
        self.play(FadeIn(grid, scale=0.95), FadeIn(count), run_time=1.6)
        self.wait(0.8)

        # --- the 10 infected -------------------------------------------------------
        inf = VGroup(*[person(i, ROSE) for i in INFECTED])
        rings = VGroup(*[Circle(0.16, color=ROSE, stroke_width=3).move_to(d) for d in inf])
        l_inf = VGroup(T("0.1% × 10,000 =", 28), T("10 infected", 30, ROSE, weight="BOLD")).arrange(RIGHT).move_to([3.6, -0.5, 0])
        self.play(FadeIn(inf), LaggedStart(*[Create(r) for r in rings], lag_ratio=0.1), FadeIn(l_inf))
        self.play(*[r.animate.scale(0.55).set_stroke(width=2) for r in rings], run_time=1.2)
        self.wait(0.5)

        # --- who tests positive ------------------------------------------------------
        head = self.heading("2 · Light up everyone who tests positive", head)
        l_tp = VGroup(T("99% of 10 ≈", 26), T("9.9 true +", 28, ROSE, weight="BOLD")).arrange(RIGHT)
        l_fp = VGroup(T("5% of 9,990 ≈", 26), T("499.5 false +", 28, YELLOW, weight="BOLD")).arrange(RIGHT)
        VGroup(l_tp, l_fp).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.6, -1.5, 0])
        self.play(FadeIn(l_tp), *[Indicate(d, color=ROSE, scale_factor=2.5) for d in inf])
        fp = VGroup(*[person(i, YELLOW) for i in FALSE_POS])
        self.play(LaggedStart(*[FadeIn(d, scale=3) for d in fp], lag_ratio=0.004, run_time=3), FadeIn(l_fp))
        self.play(grid.animate.set_opacity(0.25))
        self.wait(1)

        # --- gather the positives ----------------------------------------------------
        head = self.heading("3 · Keep only the positives", head)
        self.play(FadeOut(fact_grp), FadeOut(count), FadeOut(l_inf), VGroup(l_tp, l_fp).animate.move_to([3.6, 2.4, 0]))
        cols, pitch = 34, 0.15
        block_c = np.array([3.6, -0.3, 0])
        positives = list(inf) + list(fp)
        rows = int(np.ceil(len(positives) / cols))
        for k, d in enumerate(positives):
            r, c = divmod(k, cols)
            d.generate_target()
            d.target.move_to(block_c + np.array([(c - (cols - 1) / 2) * pitch, ((rows - 1) / 2 - r) * pitch, 0]))
            d.target.scale(1.8)
        self.play(
            LaggedStart(*[MoveToTarget(d) for d in positives], lag_ratio=0.002, run_time=3.5),
            FadeOut(grid, run_time=2),
            FadeOut(rings, run_time=1),
        )
        block = VGroup(*positives)
        br = Brace(block, DOWN, color=MUTED)
        br_t = T("≈ 510 positive tests", 26, MUTED).next_to(br, DOWN, buff=0.1)
        self.play(GrowFromCenter(br), FadeIn(br_t))
        frac = MathTex(r"\frac{9.9}{9.9 + 499.5}", r"\approx", r"1.9\%", font_size=56).move_to([-3.3, 0.3, 0])
        frac[0][0:3].set_color(ROSE)
        frac[0][4:7].set_color(ROSE)
        frac[0][8:13].set_color(YELLOW)
        frac[2].set_color(GREEN)
        self.play(Write(frac))
        msg = VGroup(
            T("Only about 2 in 100 positives are infected.", 30, YELLOW),
            T("The tiny base rate P(V) lets false alarms swamp the true cases.", 24, MUTED),
        ).arrange(DOWN, buff=0.15).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(msg))
        self.wait(3)

        # --- the formula ---------------------------------------------------------------
        head = self.heading("4 · The same picture as Bayes' rule", head)
        self.play(FadeOut(msg), FadeOut(frac), FadeOut(br), FadeOut(br_t), FadeOut(VGroup(l_tp, l_fp)))
        self.play(block.animate.scale(0.55).to_corner(DR, buff=0.6))
        rule = MathTex(r"P(V \mid +)", "=", r"\frac{P(V)\,P(+ \mid V)}{P(+)}", font_size=54).move_to([0, 2.0, 0])
        self.play(Write(rule))
        self.wait(0.6)
        num =MathTex(r"(0.001)(0.99)", font_size=46, color=ROSE)
        den = MathTex(r"(0.001)(0.99)", "+", r"(0.999)(0.05)", font_size=46)
        den[0].set_color(ROSE)
        den[2].set_color(YELLOW)
        bar = Line(LEFT, RIGHT, color=FG, stroke_width=3)
        eq = MathTex("=", font_size=46)
        bar.set_width(den.width + 0.2)
        num.next_to(bar, UP, buff=0.15)
        den.next_to(bar, DOWN, buff=0.15)
        fracg = VGroup(num, bar, den)
        line2 = VGroup(eq, fracg).arrange(RIGHT, buff=0.3).move_to([-0.6, 0.1, 0])
        self.play(FadeIn(eq), Create(bar))
        self.play(TransformFromCopy(rule[2][0:10], num))
        tp_lbl = T("true positives (red)", 22, ROSE).next_to(num, RIGHT, buff=0.5)
        self.play(FadeIn(tp_lbl))
        self.play(TransformFromCopy(rule[2][11:], den))
        fp_lbl = T("all positives: red + yellow", 22, YELLOW).next_to(den, RIGHT, buff=0.5)
        self.play(FadeIn(fp_lbl))
        res = MathTex(r"= \frac{0.00099}{0.05094} \approx 0.019", font_size=46, color=GREEN).next_to(line2, DOWN, buff=0.6)
        self.play(Write(res))
        self.play(Circumscribe(block, color=ROSE))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])

        self.naive(head)

    # ------------------------------------------------------------------------------
    def naive(self, head):
        head = self.heading("5 · Naive Bayes: one likelihood per feature", head)
        nb = MathTex(
            r"B_{NB} = \arg\max_{B_i}\; P(B_i)\,",
            r"P(e_1 \mid B_i)\,P(e_2 \mid B_i)\cdots P(e_d \mid B_i)",
            font_size=40,
        ).move_to([0, 2.3, 0])
        nb[1].set_color(TEAL)
        self.play(Write(nb))
        naive = T("“naive”: features assumed independent given the class", 24, MUTED).next_to(nb, DOWN, buff=0.2)
        self.play(FadeIn(naive))

        ex = T("Lecture example — disease Z, patient (BP, Fever, Diabetes, Vomit)", 24, FG).move_to([0, 1.05, 0])
        self.play(FadeIn(ex))

        def row(label, factors, color, y):
            lbl = T(label, 26, color).move_to([-5.6, y, 0])
            parts = [MathTex(rf"\frac{{{f.numerator}}}{{{f.denominator}}}" if f.denominator != 1 else str(f.numerator), font_size=34) for f in factors]
            g = VGroup()
            for k, p in enumerate(parts):
                if k:
                    g.add(MathTex(r"\cdot", font_size=34))
                g.add(p)
            g.arrange(RIGHT, buff=0.16).next_to(lbl, RIGHT, buff=0.5)
            return lbl, g, parts

        def product_anim(g, parts, factors, y, color):
            val = DecimalNumber(float(factors[0]), num_decimal_places=6, font_size=36, color=color).move_to([4.9, y, 0])
            eq = MathTex("=", font_size=34).next_to(val, LEFT, buff=0.25)
            anims = []
            self.play(FadeIn(parts[0]), FadeIn(eq), FadeIn(val), run_time=0.5)
            run = factors[0]
            for k, (p, f) in enumerate(zip(parts[1:], factors[1:]), 1):
                run *= f
                self.play(FadeIn(g[2 * k - 1]), FadeIn(p, shift=UP * 0.1), val.animate.set_value(float(run)), run_time=0.6)
            return val, eq

        # patient (High, No fever, Yes, Yes) — straight from the lecture
        q1 = T("(High, No fever, Yes, Yes)", 26, YELLOW).next_to(ex, DOWN, buff=0.25)
        self.play(FadeIn(q1))
        yes_f = [Fr(9, 14), Fr(2, 9), Fr(3, 9), Fr(3, 9), Fr(3, 9)]
        no_f = [Fr(5, 14), Fr(3, 5), Fr(1, 5), Fr(4, 5), Fr(3, 5)]
        ly, gy, py = row("Z = Yes", yes_f, GREEN, -0.6)
        ln, gn, pn = row("Z = No", no_f, ROSE, -1.6)
        self.play(FadeIn(ly), FadeIn(ln))
        vy, ey = product_anim(gy, py, yes_f, -0.6, GREEN)
        vn, en = product_anim(gn, pn, no_f, -1.6, ROSE)
        win = SurroundingRectangle(vn, color=ROSE, buff=0.1)
        verdict = T("predict: No (0.020571 > 0.005291)", 24, ROSE).to_edge(DOWN, buff=0.6)
        self.play(Create(win), FadeIn(verdict))
        self.wait(1.5)

        # --- a zero count ----------------------------------------------------------------
        q2 = T("(Low, No fever, Yes, Yes)", 26, YELLOW).move_to(q1)
        self.play(FadeOut(win), FadeOut(verdict), Transform(q1, q2))
        new_y = MathTex(r"\frac{4}{9}", font_size=34).move_to(py[1])
        new_n = MathTex(r"\frac{0}{5}", font_size=34, color=ROSE).move_to(pn[1])
        self.play(Transform(py[1], new_y), Transform(pn[1], new_n))
        zero_note = T("BP = Low never occurs with Z = No in the training data", 22, ROSE).next_to(gn, DOWN, buff=0.35).align_to(gn, LEFT)
        self.play(FadeIn(zero_note), Indicate(pn[1], color=ROSE, scale_factor=1.6))
        yes_low = [Fr(9, 14), Fr(4, 9), Fr(3, 9), Fr(3, 9), Fr(3, 9)]
        self.play(vy.animate.set_value(float(prod(yes_low))), vn.animate.set_value(0.0), run_time=1.2)
        crosses = VGroup(*[Cross(p, stroke_color=ROSE, stroke_width=4, scale_factor=0.8) for k, p in enumerate(pn) if k != 1])
        self.play(LaggedStart(*[Create(c) for c in crosses], lag_ratio=0.15))
        wipe = VGroup(
            T("One zero wipes out every other likelihood:", 28, ROSE),
            T("the model is now 100% sure the answer is “Yes”.", 28, ROSE),
        ).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(wipe))
        self.wait(2.5)

        # --- Laplace smoothing --------------------------------------------------------------
        self.play(FadeOut(VGroup(wipe, crosses, zero_note, ex, q1, naive)))
        head = self.heading("6 · Fix: α-Laplace smoothing", head)
        lap = MathTex(
            r"P(e_j = v_k \mid B_i) = \frac{\text{Count}(e_j = v_k,\, B_i) + \alpha}{\text{Count}(B_i) + m\,\alpha}",
            font_size=34,
        ).move_to([0, 1.1, 0])
        self.play(FadeOut(nb), Write(lap))
        m_note = T("m = number of values the feature can take · α = 1: pretend you saw each value once more", 22, MUTED).next_to(lap, DOWN, buff=0.2)
        self.play(FadeIn(m_note))

        ys = [Fr(9, 14), Fr(4 + 1, 9 + 3), Fr(3 + 1, 9 + 3), Fr(3 + 1, 9 + 2), Fr(3 + 1, 9 + 2)]
        ns = [Fr(5, 14), Fr(0 + 1, 5 + 3), Fr(1 + 1, 5 + 3), Fr(4 + 1, 5 + 2), Fr(3 + 1, 5 + 2)]
        tex_y = [r"\frac{9}{14}", r"\frac{4+1}{9+3}", r"\frac{3+1}{9+3}", r"\frac{3+1}{9+2}", r"\frac{3+1}{9+2}"]
        tex_n = [r"\frac{5}{14}", r"\frac{0+1}{5+3}", r"\frac{1+1}{5+3}", r"\frac{4+1}{5+2}", r"\frac{3+1}{5+2}"]
        anims = []
        for parts, texs in ((py, tex_y), (pn, tex_n)):
            for k in range(1, 5):
                nt = MathTex(texs[k], font_size=34, color=GREEN if k == 1 and parts is pn else FG)
                anims.append(Transform(parts[k], nt.move_to(parts[k])))
        self.play(*anims, run_time=1.5)
        for g in (gy, gn):
            self.play(g.animate.arrange(RIGHT, buff=0.16).next_to(g.get_left(), RIGHT, buff=0), run_time=0.4)
        py_val, pn_val = float(prod(ys)), float(prod(ns))
        self.play(vy.animate.set_value(py_val), vn.animate.set_value(pn_val), run_time=1.5)
        post = py_val / (py_val + pn_val)
        out = VGroup(
            T(f"No zero any more: P(Yes | E) ≈ {post:.0%} — every symptom counts again.", 28, GREEN),
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(out))
        self.wait(3)
