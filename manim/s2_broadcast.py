"""Topic 2 — Broadcasting for pairwise squared distances.

Render:  manim -qh s2_broadcast.py BroadcastDistances

3-D layout (cube (i, j, k) of an (n, m, d) array):
    axis 0 (i) -> -Z (down),  axis 1 (j) -> +Y (depth),  axis 2 (k) -> +X (right)
"""

import numpy as np
from manim import *

from common import *

X = np.array([[0, 0], [2, 1], [1, 3]])
Y = np.array([[1, 0], [0, 2], [3, 3], [2, 2]])
N, D = X.shape
M = Y.shape[0]
DIFF = X[:, None, :] - Y[None, :, :]
SQ = DIFF**2
D2 = SQ.sum(axis=2)

S = 0.62  # cube pitch
AX_COL = {0: ROSE, 1: GREEN, 2: BLUE}
X_COLS = ["#1d4ed8", "#2563eb", "#3b82f6"]  # one shade per row of X
Y_COLS = ["#c2410c", "#ea580c", "#f97316", "#fb923c"]  # one shade per row of Y

FRONT = dict(phi=90 * DEGREES, theta=-90 * DEGREES)
ISO = dict(phi=68 * DEGREES, theta=-58 * DEGREES)
SIDE = dict(phi=90 * DEGREES, theta=0 * DEGREES)

XO, YO, MO = [-4.2, 0, 1.0], [0.6, 0, 1.3], [-2.4, -1.0, -0.2]  # block origins (cube 0,0,0)
# fixed-in-frame text follows frame_center, so keep the camera centred and shift the blocks instead
ISO_FRAME = dict(zoom=1.15)
ISO_SHIFT = np.array([-1.0, -1.0, -1.2])
PANEL_AT, PANEL_FS = [1.6, 2.9, 0], 28


def cell(value, color, face="front", opacity=0.9):
    """A cube with its value written on one face."""
    c = Cube(side_length=S * 0.9, fill_color=color, fill_opacity=opacity, stroke_color=BG, stroke_width=1.5)
    t = Text(str(int(value)), font=FONT, font_size=26, color=WHITE, weight="BOLD")
    if face == "front":  # faces -Y, readable from the front camera
        t.rotate(PI / 2, RIGHT).move_to([0, -S * 0.46, 0])
    elif face == "side":  # faces +X, readable from the side camera
        t.rotate(PI / 2, RIGHT).rotate(PI / 2, OUT).move_to([S * 0.46, 0, 0])
    elif face == "top":
        t.move_to([0, 0, S * 0.46])
    return VGroup(c, t)


def pos(origin, i, j, k):
    return np.array(origin) + np.array([k * S, j * S, -i * S])


def heat(v, vmax):
    return interpolate_color(ManimColor("#1e3a8a"), ManimColor(YELLOW), v / vmax)


class BroadcastDistances(Explainer, ThreeDScene):
    def fixed(self, *mobs):
        self.add_fixed_in_frame_mobjects(*mobs)
        self.remove(*mobs)
        return mobs[0] if len(mobs) == 1 else VGroup(*mobs)

    def construct(self):
        self.title_card("Broadcasting for pairwise distances", "COMP2211 · NumPy · appears in 6 of 6 midterms")
        self.set_camera_orientation(**FRONT, zoom=1.0)

        head = self.fixed(T("1 · Two point sets, every pair", 30, MUTED).to_corner(UL, buff=0.4))
        self.play(FadeIn(head))

        xo, yo = XO, YO
        Xc = VGroup(*[cell(X[i, k], X_COLS[i]).move_to(pos(xo, i, 0, k)) for i in range(N) for k in range(D)])
        Yc = VGroup(*[cell(Y[j, k], Y_COLS[j]).move_to(pos(yo, j, 0, k)) for j in range(M) for k in range(D)])
        xl = self.fixed(MathTex(r"X:\ (n, d) = (3, 2)", font_size=34, color=BLUE).move_to([-3.9, -1.3, 0]))
        yl = self.fixed(MathTex(r"Y:\ (m, d) = (4, 2)", font_size=34, color=ORANGE).move_to([0.9, -1.9, 0]))
        rows = self.fixed(
            T("each row is a point in d = 2 dimensions", 24, MUTED).move_to([-3.9, -1.85, 0]),
        )
        self.play(LaggedStart(*[FadeIn(c) for c in Xc], lag_ratio=0.08), FadeIn(xl))
        self.play(LaggedStart(*[FadeIn(c) for c in Yc], lag_ratio=0.08), FadeIn(yl), FadeIn(rows))
        goal = self.fixed(
            MathTex(r"\text{Goal: } D[i, j] = \lVert X_i - Y_j \rVert^2 \quad (n, m) = (3, 4)", font_size=34).to_edge(
                DOWN, buff=0.5
            )
        )
        self.play(Write(goal))
        self.wait(1.5)

        # --- swing into 3-D ----------------------------------------------------
        self.play(FadeOut(xl), FadeOut(yl), FadeOut(rows), FadeOut(goal))
        self.remove(*Xc, *Yc)
        self.add(Xc, Yc)
        self.move_camera(
            **ISO, **ISO_FRAME, added_anims=[Xc.animate.shift(ISO_SHIFT), Yc.animate.shift(ISO_SHIFT)], run_time=2.5
        )
        xo, yo = list(np.add(XO, ISO_SHIFT)), list(np.add(YO, ISO_SHIFT))

        panel = VGroup()

        def panel_line(tex, color=FG, hi=None):
            parts = tex if isinstance(tex, (list, tuple)) else [tex]
            line = MathTex(*parts, font_size=PANEL_FS, color=color)
            if hi is not None:
                line[hi].set_color(GREEN)
            if len(panel):
                line.next_to(panel[-1], DOWN, aligned_edge=LEFT, buff=0.22)
            else:
                line.next_to(PANEL_AT, RIGHT, buff=0)
            panel.add(line)
            self.fixed(line)
            return line

        legend = self.fixed(
            VGroup(
                *[
                    VGroup(Line(ORIGIN, RIGHT * 0.5, color=AX_COL[a], stroke_width=6), T(f"axis {a}", 22, AX_COL[a])).arrange(
                        RIGHT, buff=0.15
                    )
                    for a in range(3)
                ]
            )
            .arrange(RIGHT, buff=0.4)
            .to_edge(DOWN, buff=0.35)
        )
        self.play(FadeIn(legend))

        def axis_arrows(origin, n0, n1, n2):
            o = np.array(origin) + np.array([-S / 2, -S / 2, S / 2])
            return VGroup(
                Arrow3D(o, o + np.array([0, 0, -n0 * S]), color=AX_COL[0], thickness=0.012),
                Arrow3D(o, o + np.array([0, n1 * S, 0]), color=AX_COL[1], thickness=0.012),
                Arrow3D(o, o + np.array([n2 * S, 0, 0]), color=AX_COL[2], thickness=0.012),
            )

        head2 = self.fixed(T("2 · Line up the axes", 30, MUTED).to_corner(UL, buff=0.4))
        self.play(FadeOut(head), FadeIn(head2))
        head = head2

        l1 = panel_line([r"\texttt{X[:, None]}\;\to\;(3,\ ", "1", r",\ 2)"], hi=1)
        xa = axis_arrows(xo, N, 1, D)
        self.play(Write(l1), FadeIn(xa))
        note = self.fixed(T("same numbers, new size-1 axis 1", 24, GREEN).next_to(l1, DOWN, aligned_edge=LEFT))
        self.play(FadeIn(note), Indicate(xa[1], color=GREEN, scale_factor=1.3))
        self.wait(1)
        self.play(FadeOut(note))

        l2 = panel_line([r"\texttt{Y}\;(4,\,2)\;\to\;(", "1", r",\ 4,\ 2)"], hi=1)
        rule = self.fixed(
            T("shapes align from the right;", 22, MUTED),
            T("a missing leading axis counts as 1", 22, MUTED),
        )
        rule.arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(l2, DOWN, aligned_edge=LEFT)
        self.play(Write(l2), FadeIn(rule))
        # Y's rows rotate from axis 0 (down) into axis 1 (depth).
        top_edge = np.array(yo) + np.array([0, -S / 2, S / 2])
        self.play(Rotate(Yc, angle=PI / 2, axis=RIGHT, about_point=top_edge), run_time=2)
        # place Y's slab so its origin cube is at yo again (row j now at depth j)
        ya = axis_arrows(yo, 1, M, D)
        self.play(FadeIn(ya))
        self.wait(1.2)
        self.play(FadeOut(rule))

        # --- stretch -------------------------------------------------------------
        head2 = self.fixed(T("3 · Stretch every size-1 axis", 30, MUTED).to_corner(UL, buff=0.4))
        self.play(FadeOut(head), FadeIn(head2))
        head = head2
        l3 = panel_line(r"(3, 1, 2)\ \&\ (1, 4, 2)\;\to\;(3, 4, 2)", YELLOW)
        self.play(Write(l3))

        Xb = VGroup(*[cell(X[i, k], X_COLS[i]).move_to(pos(xo, i, j, k)) for j in range(M) for i in range(N) for k in range(D)])
        Yb = VGroup(*[cell(Y[j, k], Y_COLS[j], face="top").move_to(pos(yo, i, j, k)) for i in range(N) for j in range(M) for k in range(D)])
        # copies of X slide back along axis 1; copies of Y slide down along axis 0
        self.remove(Xc)
        self.add(Xb[: N * D])
        for j in range(1, M):
            sl = VGroup(*Xb[j * N * D : (j + 1) * N * D])
            self.play(TransformFromCopy(VGroup(*Xb[: N * D]), sl), run_time=0.6)
        self.wait(0.3)
        self.remove(Yc)
        Yrow0 = VGroup(*Yb[: M * D])
        self.add(Yrow0)
        for i in range(1, N):
            sl = VGroup(*Yb[i * M * D : (i + 1) * M * D])
            self.play(TransformFromCopy(Yrow0, sl), run_time=0.7)
        self.play(
            Transform(xa, axis_arrows(xo, N, M, D)),
            Transform(ya, axis_arrows(yo, N, M, D)),
        )
        self.wait(1.2)

        # --- subtract, square, sum -------------------------------------------------
        head2 = self.fixed(T("4 · Subtract, square, sum over axis 2", 30, MUTED).to_corner(UL, buff=0.4))
        self.play(FadeOut(head), FadeIn(head2))
        head = head2
        mo = MO
        l4 = panel_line(r"\texttt{diff = X[:, None] - Y}\quad(3, 4, 2)")
        Db = VGroup(
            *[
                cell(DIFF[i, j, k], "#0f766e", face="front").move_to(pos(mo, i, j, k))
                for j in range(M)
                for i in range(N)
                for k in range(D)
            ]
        )
        self.play(Write(l4), FadeOut(xa), FadeOut(ya))
        self.play(
            Xb.animate.move_to(Db.get_center()).set_opacity(0),
            Yb.animate.move_to(Db.get_center()).set_opacity(0),
            FadeIn(Db),
            run_time=2,
        )
        self.remove(Xb, Yb)
        self.wait(0.8)

        l5 = panel_line(r"\texttt{sq = diff ** 2}\quad(3, 4, 2)")
        Sb = VGroup(
            *[
                cell(SQ[i, j, k], "#a16207", face="front").move_to(pos(mo, i, j, k))
                for j in range(M)
                for i in range(N)
                for k in range(D)
            ]
        )
        self.play(Write(l5), Transform(Db, Sb), run_time=1.5)
        self.wait(0.8)

        l6 = panel_line(r"\texttt{D2 = sq.sum(axis=2)}\quad(3, 4)", YELLOW)
        vmax = D2.max()
        Rb = VGroup(
            *[cell(D2[i, j], heat(D2[i, j], vmax), face="side").move_to(pos(mo, i, j, 0.5)) for j in range(M) for i in range(N)]
        )
        sa = Arrow3D(
            np.array(mo) + np.array([-S / 2, -S * 0.9, S * 0.7]),
            np.array(mo) + np.array([D * S - S / 2, -S * 0.9, S * 0.7]),
            color=AX_COL[2],
            thickness=0.015,
        )
        self.play(Write(l6), FadeIn(sa))
        self.play(ReplacementTransform(Db, Rb), FadeOut(sa), run_time=2)
        self.wait(0.5)

        # swing round to read D2 as a matrix (rows = axis 0, columns = axis 1)
        self.play(FadeOut(legend), FadeOut(panel))
        self.move_camera(**SIDE, frame_center=Rb.get_center() + np.array([0, 0.6, 0]), zoom=1.25, run_time=2.5)
        cap = self.fixed(
            MathTex(r"D_2[i, j] = \lVert X_i - Y_j \rVert^2", font_size=40, color=YELLOW).to_edge(DOWN, buff=0.6)
        )
        rl = self.fixed(T("rows: points of X", 24, BLUE).to_edge(LEFT, buff=1.0).shift(DOWN * 0.3))
        cl = self.fixed(T("columns: points of Y", 24, ORANGE).to_edge(UP, buff=1.2))
        self.play(Write(cap), FadeIn(rl), FadeIn(cl))
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=1, frame_center=ORIGIN)
        self.fixed_in_frame_mobjects.clear() if hasattr(self, "fixed_in_frame_mobjects") else None

        self.expansion()

    # ---------------------------------------------------------------------------
    def expansion(self):
        head = self.heading("5 · The memory-friendly version")
        why = T("The (n, m, d) tensor can be huge. Expand the square instead:", 28, MUTED).next_to(head, DOWN, aligned_edge=LEFT, buff=0.3)
        self.play(FadeIn(why))

        kw = dict(font_size=46)
        e1 = MathTex(r"\lVert x - y \rVert^2", "=", r"(x - y)\cdot(x - y)", **kw)
        e2 = MathTex(r"\lVert x - y \rVert^2", "=", r"x\cdot x", "-", r"2\,x\cdot y", "+", r"y\cdot y", **kw)
        e3 = MathTex(r"\lVert x - y \rVert^2", "=", r"\lVert x \rVert^2", "+", r"\lVert y \rVert^2", "-", r"2\,x\cdot y", **kw)
        for e in (e1, e2, e3):
            e.move_to(UP * 1.4)
        self.play(Write(e1))
        self.wait(1)
        self.play(TransformMatchingTex(e1, e2))
        self.wait(1)
        e2b = MathTex(r"\lVert x - y \rVert^2", "=", r"\lVert x \rVert^2", "-", r"2\,x\cdot y", "+", r"\lVert y \rVert^2", **kw).move_to(e2)
        self.play(TransformMatchingShapes(e2, e2b))
        self.play(TransformMatchingTex(e2b, e3))
        self.wait(0.8)

        cols = [BLUE, ORANGE, PURPLE]
        e3[2].set_color(cols[0])
        e3[4].set_color(cols[1])
        e3[6].set_color(cols[2])
        self.play(*[Indicate(e3[k], color=c) for k, c in zip((2, 4, 6), cols)])

        lines = VGroup(
            code("XX = (X**2).sum(axis=1)[:, None]   # (n, 1)", 24, cols[0]),
            code("YY = (Y**2).sum(axis=1)[None, :]   # (1, m)", 24, cols[1]),
            code("XY = X @ Y.T                       # (n, m)", 24, cols[2]),
            code("D2 = XX + YY - 2 * XY              # (n, m)", 24, YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(DOWN * 1.35)
        box = SurroundingRectangle(lines, color=GRID, buff=0.25, corner_radius=0.1)
        self.play(Create(box))
        for k, ln in zip((2, 4, 6), lines[:3]):
            arrow = Arrow(e3[k].get_bottom(), ln.get_top() + UP * 0.02, buff=0.1, color=ln.color, stroke_width=3, max_tip_length_to_length_ratio=0.08)
            self.play(GrowArrow(arrow), FadeIn(ln, shift=DOWN * 0.1))
            self.play(FadeOut(arrow), run_time=0.4)
        self.play(FadeIn(lines[3], shift=DOWN * 0.1))
        bc = T("(n, 1) + (1, m) → (n, m): broadcasting again, but no d-axis is ever materialised", 24, YELLOW).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(bc))
        self.wait(2.5)

        # same matrix as the 3-D route
        self.play(*[FadeOut(m) for m in (why, box, lines, bc)], e3.animate.scale(0.8).move_to(UP * 2.2))
        xx = (X**2).sum(axis=1)[:, None]
        yy = (Y**2).sum(axis=1)[None, :]
        xy = X @ Y.T

        def mat(a, color=FG):
            return Matrix(a, element_to_mobject=lambda v: MathTex(str(int(v)), color=color), h_buff=0.85, v_buff=0.65).scale(0.7)

        terms = VGroup(
            mat(xx, BLUE), MathTex("+", font_size=40), mat(yy, ORANGE), MathTex("-\\,2", font_size=40), mat(xy, PURPLE),
            MathTex("=", font_size=40), mat(D2, YELLOW),
        ).arrange(RIGHT, buff=0.25).move_to(DOWN * 0.3)
        self.play(LaggedStart(*[FadeIn(t) for t in terms], lag_ratio=0.25), run_time=3)
        same = T("Same D2 as the 3-D route — use the expansion when n·m·d gets large.", 26, GREEN).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(same))
        self.wait(3)
