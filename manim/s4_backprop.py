"""Topic 4 — Backpropagation in the exam's notation (lecture 6 XOR network).

Render:  manim -qh s4_backprop.py BackpropXor

Numbers are the lecture's Round 1 / Step 1 (x = (0, 0), T = 0) worked example.
Step 2 uses recomputed values: the slide prints delta_j1 = -0.027463, the correct
value is -0.027643 (digits swapped), which also shifts its new w2.
"""

import numpy as np
from manim import *

from common import *

POS = {
    "x1": np.array([-6.4, 1.6, 0]),
    "x2": np.array([-6.4, -1.6, 0]),
    "j1": np.array([-4.0, 1.6, 0]),
    "j2": np.array([-4.0, -1.6, 0]),
    "k": np.array([-1.6, 0.0, 0]),
}
# where each weight label sits: (fraction along the edge, direction to push it)
LABEL_AT = {
    "w_1": (0.5, UP),
    "w_4": (0.5, DOWN),
    "w_2": (0.72, RIGHT),
    "w_3": (0.72, RIGHT),
    "w_5": (0.5, UR),
    "w_6": (0.5, DR),
}
NAMES = {"x1": "x_1", "x2": "x_2", "j1": "j_1", "j2": "j_2", "k": "k"}
EDGES = {  # weight name -> (source, target, initial value)
    "w_1": ("x1", "j1", "-0.65"),
    "w_2": ("x2", "j1", "0.64"),
    "w_3": ("x1", "j2", "1.11"),
    "w_4": ("x2", "j2", "0.84"),
    "w_5": ("j1", "k", "0.86"),
    "w_6": ("j2", "k", "-1.38"),
}
R = 0.42


class BackpropXor(Explainer):
    # ---------------------------------------------------------------- helpers
    def build_net(self):
        nodes, edges, wlbl = {}, {}, {}
        for k, p in POS.items():
            col = MUTED if k.startswith("x") else (TEAL if k.startswith("j") else YELLOW)
            c = Circle(R, color=col, stroke_width=3).set_fill(BG, 1).move_to(p)
            nodes[k] = VGroup(c, MathTex(NAMES[k], font_size=34).move_to(p))
        for w, (a, b, v) in EDGES.items():
            e = Line(POS[a], POS[b], buff=R, color=GRID, stroke_width=3)
            edges[w] = e
            wlbl[w] = self.weight_label(w, v, FG)
        return nodes, edges, wlbl

    def weight_label(self, w, value, color):
        a, b, _ = EDGES[w]
        t, side = LABEL_AT[w]
        anchor = POS[a] + (POS[b] - POS[a]) * t
        lbl = MathTex(f"{w} = {value}", font_size=26, color=color)
        lbl.add_background_rectangle(color=BG, opacity=0.85, buff=0.04)
        return lbl.next_to(anchor, side, buff=0.1)

    def set_w(self, w, value, color=GREEN):
        # the on-screen label morphs in place, so self.wlbl[w] stays the live object
        return Transform(self.wlbl[w], self.weight_label(w, value, color))

    def value_tag(self, node, tex, color=FG, side=DOWN):
        return MathTex(tex, font_size=26, color=color).next_to(self.nodes[node], side, buff=0.12)

    def panel(self, *lines, top=2.6, size=32, buff=0.28):
        g = VGroup(*[MathTex(l, font_size=size) if isinstance(l, str) else l for l in lines])
        g.arrange(DOWN, aligned_edge=LEFT, buff=buff)
        g.next_to([0.3, top, 0], DR, buff=0)
        return g

    def flow(self, names, color=YELLOW, reverse=False):
        anims = []
        for w in names:
            e = self.edges[w].copy().set_stroke(color, 6)
            if reverse:
                e = Line(e.get_end(), e.get_start(), color=color, stroke_width=6)
            anims.append(ShowPassingFlash(e, time_width=0.6))
        return anims

    # ------------------------------------------------------------------ scene
    def construct(self):
        self.title_card("Backpropagation, exam notation", "COMP2211 · MLP · appears in 3 of 3 finals")
        head = self.heading("The lecture's XOR network")
        self.nodes, self.edges, self.wlbl = self.build_net()
        self.play(
            LaggedStart(*[Create(e) for e in self.edges.values()], lag_ratio=0.08),
            LaggedStart(*[FadeIn(n) for n in self.nodes.values()], lag_ratio=0.08),
        )
        self.play(LaggedStart(*[FadeIn(l) for l in self.wlbl.values()], lag_ratio=0.1))
        setup = self.panel(
            r"f(x) = \frac{1}{1 + e^{-x}} \quad\text{(sigmoid)}",
            r"\theta_1 = \theta_2 = \theta_3 = 0,\quad \eta = 0.5",
            r"\text{Round 1, Step 1: } x = (0, 0),\ T = 0",
        )
        self.play(FadeIn(setup, shift=LEFT * 0.2))
        self.wait(2)

        # ------------------------------------------------------- forward pass
        head = self.heading("1 · Forward pass: numbers flow left to right", head)
        self.play(FadeOut(setup))
        tx1, tx2 = self.value_tag("x1", "0", YELLOW, LEFT), self.value_tag("x2", "0", YELLOW, LEFT)
        self.play(FadeIn(tx1), FadeIn(tx2))
        fw = self.panel(
            r"\Sigma_1 = x_1 w_1 + x_2 w_2 = 0",
            r"O_{j_1} = f(\Sigma_1 + \theta_1) = f(0) = 0.5",
            r"\Sigma_2 = x_1 w_3 + x_2 w_4 = 0",
            r"O_{j_2} = f(0) = 0.5",
            r"\Sigma_3 = O_{j_1} w_5 + O_{j_2} w_6",
            r"\phantom{\Sigma_3} = 0.5(0.86) + 0.5(-1.38) = -0.26",
            r"O_k = f(-0.26) = 0.435364",
            size=30,
        )
        self.play(*self.flow(["w_1", "w_2"]), FadeIn(fw[0]), run_time=1.2)
        tj1 = self.value_tag("j1", "O = 0.5", TEAL, UP)
        self.play(FadeIn(fw[1]), FadeIn(tj1))
        self.play(*self.flow(["w_3", "w_4"]), FadeIn(fw[2]), run_time=1.2)
        tj2 = self.value_tag("j2", "O = 0.5", TEAL, DOWN)
        self.play(FadeIn(fw[3]), FadeIn(tj2))
        self.play(*self.flow(["w_5", "w_6"]), FadeIn(fw[4]), run_time=1.2)
        self.play(FadeIn(fw[5]))
        tk = self.value_tag("k", "O = 0.435364", YELLOW, UP)
        self.play(FadeIn(fw[6]), FadeIn(tk))
        err = MathTex(r"E = \tfrac12 (O_k - T_k)^2 = \tfrac12(0.435364 - 0)^2 = 0.094771", font_size=30, color=ROSE)
        err.next_to(fw, DOWN, aligned_edge=LEFT, buff=0.4)
        self.play(Write(err))
        self.wait(2)

        # --------------------------------------------------- derive delta_k
        head = self.heading("2 · Chain rule, one factor at a time", head)
        self.play(FadeOut(fw), err.animate.next_to([0.3, 2.6, 0], DR, buff=0))
        kw = dict(font_size=36)
        chain = MathTex(
            r"\frac{\partial E}{\partial w_5}", "=",
            r"\frac{\partial E}{\partial O_k}", r"\cdot",
            r"\frac{\partial O_k}{\partial \Sigma_k}", r"\cdot",
            r"\frac{\partial \Sigma_k}{\partial w_5}",
            **kw,
        ).next_to(err, DOWN, aligned_edge=LEFT, buff=0.55)
        self.play(Write(chain))
        f1 = MathTex(r"(O_k - T_k)", font_size=32, color=ROSE)
        f2 = MathTex(r"O_k(1 - O_k)", font_size=32, color=PURPLE)
        f3 = MathTex(r"O_{j_1}", font_size=32, color=TEAL)
        why = [
            T("from E = ½(O − T)²", 20, ROSE),
            T("sigmoid slope f′ = f(1 − f)", 20, PURPLE),
            T("Σ = O_j1·w5 + O_j2·w6 + θ", 20, TEAL),
        ]
        shown = None
        VGroup(f1, f2, f3).arrange(RIGHT, buff=0.35).next_to(chain, DOWN, buff=0.75).align_to(chain, LEFT)
        for f, part, w in zip((f1, f2, f3), (chain[2], chain[4], chain[6]), why):
            arr = Arrow(part.get_bottom(), f.get_top(), buff=0.06, color=f.get_color(), stroke_width=3, max_tip_length_to_length_ratio=0.2)
            w.next_to([0.3, f.get_bottom()[1] - 0.3, 0], RIGHT, buff=0)
            swap = [FadeIn(w)] if shown is None else [FadeOut(shown), FadeIn(w)]
            self.play(Indicate(part, color=f.get_color()), GrowArrow(arr), FadeIn(f, shift=DOWN * 0.1), *swap, run_time=1.1)
            shown = w
            self.wait(0.8)
        self.play(FadeOut(shown))
        br = Brace(VGroup(f1, f2), DOWN, buff=0.2, color=YELLOW)
        dk = MathTex(r"\delta_k = (O_k - T_k)\,O_k(1 - O_k)", font_size=38, color=YELLOW).next_to(br, DOWN, buff=0.15)
        dk.shift(RIGHT * max(0, 0.3 - dk.get_left()[0]))
        self.play(GrowFromCenter(br), Write(dk))
        dkv = MathTex(r"= (0.435364)(0.435364)(0.564636) = 0.107022", font_size=27, color=YELLOW).next_to(dk, DOWN, aligned_edge=LEFT)
        self.play(FadeIn(dkv))
        tdk = self.value_tag("k", r"\delta_k = 0.107022", YELLOW, DOWN)
        self.play(TransformFromCopy(dkv, tdk))
        self.wait(1.5)
        derive = Group(*[m for m in self.mobjects if m not in (head, tx1, tx2, tj1, tj2, tk, tdk, *self.nodes.values(), *self.edges.values(), *self.wlbl.values())])

        # -------------------------------------------------- output-layer update
        head = self.heading("3 · Update the output weights:  w ← w − η δ O", head)
        self.play(FadeOut(derive))
        up = self.panel(
            MathTex(r"w_{jk} \leftarrow w_{jk} - \eta\,\delta_k\,O_j", font_size=38, color=YELLOW),
            r"w_5 = 0.86 - (0.5)(0.107022)(0.5) = 0.833245",
            r"w_6 = -1.38 - (0.5)(0.107022)(0.5) = -1.406756",
            r"\theta_3 = 0 - (0.5)(0.107022) = -0.053511",
            size=30,
        )
        self.play(Write(up[0]))
        self.play(FadeIn(up[1]), self.set_w("w_5", "0.833245"))
        self.play(FadeIn(up[2]), self.set_w("w_6", "-1.406756"))
        self.play(FadeIn(up[3]))
        self.wait(1.5)

        # ------------------------------------------------------ delta backwards
        head = self.heading("4 · Send δ backwards to the hidden layer", head)
        bk = self.panel(
            MathTex(r"\delta_j = O_j(1 - O_j)\sum_{k} \delta_k\, w_{jk}", font_size=38, color=ROSE),
            r"\delta_{j_1} = 0.5(1 - 0.5)(0.107022)(0.86) = 0.023010",
            r"\delta_{j_2} = 0.5(1 - 0.5)(0.107022)(-1.38) = -0.036923",
            T("uses the old w5, w6 (as in the lecture)", 22, MUTED),
            size=30,
        )
        self.play(FadeOut(up), Write(bk[0]))
        self.play(*self.flow(["w_5"], ROSE, reverse=True), FadeIn(bk[1]), run_time=1.3)
        tdj1 = MathTex(r"\delta = 0.023010", font_size=26, color=ROSE).next_to(tj1, UP, buff=0.08)
        self.play(FadeIn(tdj1))
        self.play(*self.flow(["w_6"], ROSE, reverse=True), FadeIn(bk[2]), run_time=1.3)
        tdj2 = MathTex(r"\delta = -0.036923", font_size=26, color=ROSE).next_to(tj2, DOWN, buff=0.08)
        self.play(FadeIn(tdj2), FadeIn(bk[3]))
        self.wait(1.5)

        # ------------------------------------------------------ input weights
        head = self.heading("5 · Update the input weights:  w ← w − η δ O", head)
        iw = self.panel(
            MathTex(r"w_{ij} \leftarrow w_{ij} - \eta\,\delta_j\,O_i", font_size=38, color=YELLOW),
            r"O_i = x_i = 0 \;\Rightarrow\; w_1, w_2, w_3, w_4 \text{ unchanged}",
            r"\theta_1 = 0 - (0.5)(0.023010) = -0.011505",
            r"\theta_2 = 0 - (0.5)(-0.036923) = 0.018462",
            size=30,
        )
        self.play(FadeOut(bk), Write(iw[0]))
        self.play(FadeIn(iw[1]), *[Indicate(self.wlbl[w], color=MUTED) for w in ("w_1", "w_2", "w_3", "w_4")])
        self.play(FadeIn(iw[2]), FadeIn(iw[3]))
        trap = T("Exam trap: a zero input means a zero gradient for its weights.", 24, YELLOW).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(trap))
        self.wait(2.5)

        # ------------------------------------------------------ step 2, quickly
        head = self.heading("6 · Step 2: x = (0, 1), T = 1 — now w₂ and w₄ move", head)
        self.play(FadeOut(iw), FadeOut(trap), FadeOut(VGroup(tdk, tdj1, tdj2, tj1, tj2, tk)))
        nx2 = self.value_tag("x2", "1", YELLOW, LEFT)
        self.play(Transform(tx2, nx2))
        s2 = self.panel(
            r"O_{j_1} = 0.652148,\quad O_{j_2} = 0.702339,\quad O_k = 0.377980",
            r"\delta_k = (0.377980 - 1)(0.377980)(0.622020) = -0.146244",
            r"w_5 \to 0.880931,\quad w_6 \to -1.355400",
            r"\delta_{j_1} = -0.027643,\quad \delta_{j_2} = 0.043010",
            r"w_2 = 0.64 - (0.5)(-0.027643)(1) = 0.653822",
            r"w_4 = 0.84 - (0.5)(0.043010)(1) = 0.818495",
            size=27,
        )
        self.play(*self.flow(["w_2", "w_4", "w_5", "w_6"]), FadeIn(s2[0]), run_time=1.3)
        self.play(FadeIn(s2[1]))
        self.play(FadeIn(s2[2]), self.set_w("w_5", "0.880931"), self.set_w("w_6", "-1.355400"))
        self.play(*self.flow(["w_5", "w_6"], ROSE, reverse=True), FadeIn(s2[3]), run_time=1.2)
        self.play(*self.flow(["w_2", "w_4"], ROSE, reverse=True), run_time=1.0)
        self.play(FadeIn(s2[4]), self.set_w("w_2", "0.653822"))
        self.play(FadeIn(s2[5]), self.set_w("w_4", "0.818495"))
        foot = T("Note: the slide prints δj1 = −0.027463 (two digits swapped); the value above is recomputed.", 18, MUTED).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(foot))
        self.wait(2.5)

        # ------------------------------------------------------ after training
        head = self.heading("7 · After 10,000 rounds", head)
        self.play(FadeOut(s2), FadeOut(foot), FadeOut(tx1), FadeOut(tx2))
        final = {"w_1": "-6.251654", "w_2": "6.229623", "w_3": "-5.717901", "w_4": "5.967069", "w_5": "9.533777", "w_6": "-9.290053"}
        self.play(*[self.set_w(w, v, YELLOW) for w, v in final.items()])
        tbl = MathTex(
            r"\begin{array}{c|c|c} (x_1, x_2) & T & O_k \\ \hline (0,0) & 0 & 0.016973 \\ (0,1) & 1 & 0.984139 \\ (1,0) & 1 & 0.980513 \\ (1,1) & 0 & 0.015179 \end{array}",
            font_size=34,
        ).move_to([3.6, 0.3, 0])
        self.play(FadeIn(tbl))
        self.play(FadeIn(caption("The network has learned XOR.", 30, GREEN)))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not head])

        self.descent_coda(head)

    # --------------------------------------------------------------- coda
    def descent_coda(self, head):
        head = self.heading("Coda · Gradient descent and the learning rate", head)
        rule = MathTex(r"w \leftarrow w - \eta\,\frac{\partial E}{\partial w}", font_size=40).next_to(head, DOWN, buff=0.3).set_x(0)
        self.play(Write(rule))
        cfgs = [(0.05, "too small: crawls", ORANGE), (0.35, "about right", GREEN), (1.08, "too large: overshoots, diverges", ROSE)]
        panels = []
        for k, (eta, label, col) in enumerate(cfgs):
            ax = Axes(
                x_range=[-2.6, 2.6, 1],
                y_range=[0, 6.5, 2],
                x_length=3.9,
                y_length=3.2,
                axis_config=dict(color=MUTED, include_ticks=False, stroke_width=2),
            ).move_to([(k - 1) * 4.4, -0.6, 0])
            curve = ax.plot(lambda w: w**2, x_range=[-2.5, 2.5], color=FG, stroke_width=3)
            name = MathTex(rf"\eta = {eta}", font_size=32, color=col).next_to(ax, UP, buff=0.15)
            lab = T(label, 22, col).next_to(ax, DOWN, buff=0.25)
            panels.append((ax, curve, name, lab, eta, col))
            self.play(Create(ax), Create(curve), FadeIn(name), run_time=0.8)
        balls, ws = [], []
        for ax, curve, name, lab, eta, col in panels:
            w0 = 1.9
            b = Dot(ax.c2p(w0, w0**2), radius=0.11, color=col)
            balls.append(b)
            ws.append(w0)
        self.play(*[FadeIn(b, scale=2) for b in balls])
        for step in range(7):
            anims = []
            for idx, ((ax, curve, name, lab, eta, col), b) in enumerate(zip(panels, balls)):
                w = ws[idx]
                nw = w - eta * 2 * w
                if abs(nw) > 2.5:
                    continue
                ws[idx] = nw
                trail = Dot(b.get_center(), radius=0.05, color=col, fill_opacity=0.5)
                self.add(trail)
                path = ArcBetweenPoints(ax.c2p(w, w**2), ax.c2p(nw, nw**2), angle=-PI / 3 if nw < w else PI / 3)
                anims.append(MoveAlongPath(b, path))
            if not anims:
                break
            self.play(*anims, run_time=0.7)
        self.play(*[FadeIn(p[3]) for p in panels])
        out = Arrow(balls[2].get_center(), balls[2].get_center() + UP * 1.2 + RIGHT * 0.3, color=ROSE, buff=0.1)
        self.play(GrowArrow(out))
        self.play(FadeIn(caption("Every final asks this: too small → slow; too large → oscillate or diverge.", 26, YELLOW)))
        self.wait(3)
