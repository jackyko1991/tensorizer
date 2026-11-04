"""Solve the period tower by square roots, run Richmond's construction step by step, and test constructibility."""

import torch

from .equation import (
    circle_circle,
    euler_phi,
    fermat_factorisation,
    is_power_of_two,
    line_circle,
    line_line,
    product_in_periods,
    solve_quadratic,
)

# ---------------------------------------------------------------- the period tower by radicals


def periods_by_radicals(eta: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Rebuild every period from η_{0,0} = −1 by one quadratic per level: t² − η_{m,i} t + (c_i + C_{ik} η_{m,k}) = 0.

    Only square roots are used; the true periods `eta` only decide which root is which (as Gauss decided by
    comparing the sizes of cosines).   (5, 16) complex → (5, 16) complex"""
    tower = torch.zeros_like(eta)
    tower[0, 0] = -1
    for m in range(4):  # 4 levels, each batched over its 2^m quadratics
        n = 2**m
        const, coef = product_in_periods(order, m)
        s = tower[m, :n]  # (n,) sums
        p = const.to(eta.dtype) + coef.to(eta.dtype) @ tower[m, :n]  # (n,) products, repeated k ⇒ summed
        plus, minus = solve_quadratic(s, p)
        first = torch.where((plus - eta[m + 1, :n]).abs() < (minus - eta[m + 1, :n]).abs(), plus, minus)
        tower[m + 1, :n] = first
        tower[m + 1, n : 2 * n] = s - first
    return tower


# ---------------------------------------------------------------- Richmond's construction (1893)


class Construction:
    """A ruler-and-compass log: each step draws one circle (centre through a point) or one line (through two points)."""

    def __init__(self, points: dict):
        self.points = dict(points)
        self.steps: list[dict] = []

    def circle(self, centre: str, through: str, label: str) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.points[centre]
        r = (self.points[through] - c).norm()
        self.steps.append({"kind": "circle", "centre": c, "radius": r, "label": label, "new": []})
        return c, r

    def line(self, a: str, b: str, label: str) -> tuple[torch.Tensor, torch.Tensor]:
        pa, pb = self.points[a], self.points[b]
        self.steps.append({"kind": "line", "a": pa, "b": pb, "label": label, "new": []})
        return pa, pb

    def mark(self, name: str, point: torch.Tensor) -> torch.Tensor:
        self.points[name] = point
        self.steps[-1]["new"].append(name)
        return point


def _far(pts: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Of two candidate points (2, 2), the one farther from ref."""
    return pts[(pts - ref).norm(dim=1).argmax()]


def _perpendicular_foot_vertex(con: Construction, foot: str, vertex: str, gamma) -> None:
    """Perpendicular to the diameter at `foot`, where it meets the circle Γ above the axis is `vertex` (4 steps)."""
    o, x_axis = con.points["O"], (con.points["A"], con.points["P0"])
    c, r = con.circle(foot, "O", f"circle about {foot} through O")
    con.mark(f"S_{foot}", _far(line_circle(*x_axis, c, r), o))  # the mirror image of O in the foot
    c1, r1 = con.circle("O", f"S_{foot}", f"circle about O through S_{foot}")
    c2, r2 = con.circle(f"S_{foot}", "O", f"circle about S_{foot} through O")
    up, down = circle_circle(c1, r1, c2, r2)
    con.points[f"U_{foot}"], con.points[f"D_{foot}"] = up, down
    a, b = con.line(f"U_{foot}", f"D_{foot}", f"perpendicular at {foot}")
    pts = line_circle(a, b, *gamma)
    con.mark(vertex, pts[pts[:, 1].argmax()])


def richmond(dtype: torch.dtype, device: torch.device) -> dict:
    """H. W. Richmond's construction of the regular 17-gon in the unit circle, with P0 = (1, 0).

    J = OB/4 on the vertical radius; E on the diameter with ∠OJE = ∠OJP0 / 4; F with ∠EJF = 45°; the circle on FP0
    meets OB at K; the circle about E through K meets the diameter at N3 = cos(6π/17) and N5 = cos(10π/17); the
    perpendiculars there give P3 and P5. Every point comes from one of the three intersection functions."""
    z = torch.zeros(2, dtype=dtype, device=device)
    con = Construction({"O": z, "P0": torch.tensor([1.0, 0.0], dtype=dtype, device=device)})
    gamma = con.circle("O", "P0", "the circle Γ")
    x_axis = con.line("O", "P0", "diameter through P0")
    con.mark("A", line_circle(*x_axis, *gamma)[0])

    # perpendicular diameter: equal circles about A and P0
    up, down = circle_circle(
        *con.circle("A", "P0", "circle about A through P0"), *con.circle("P0", "A", "circle about P0 through A")
    )
    con.points["X1"], con.points["X2"] = up, down
    y_axis = con.line("X1", "X2", "perpendicular diameter")
    pts = line_circle(*y_axis, *gamma)
    con.mark("B", pts[pts[:, 1].argmax()])

    # J = OB/4 by halving twice
    c, r = con.circle("B", "O", "circle about B through O")
    con.points["Y1"], con.points["Y2"] = circle_circle(*gamma, c, r)
    con.mark("M", line_line(*con.line("Y1", "Y2", "halve OB"), *y_axis))
    pts = circle_circle(
        *con.circle("M", "O", "circle about M through O"), *con.circle("O", "M", "circle about O through M")
    )
    con.points["Z1"], con.points["Z2"] = pts
    con.mark("J", line_line(*con.line("Z1", "Z2", "halve OM: OJ = OB/4"), *y_axis))

    # E: quarter the angle OJP0 by bisecting twice
    j = con.points["J"]
    ray = con.line("J", "P0", "line JP0")
    small = con.circle("J", "O", "circle about J through O")
    con.mark("Q", line_circle(*ray, *small)[1])
    for name, target in (("Q", "R"), ("Q2", "R2")):
        pts = circle_circle(
            *con.circle("O", name, f"circle about O through {name}"),
            *con.circle(name, "O", f"circle about {name} through O"),
        )
        con.points[target] = _far(pts, j)
        bisector = con.line("J", target, "bisect the angle at J" if target == "R" else "bisect again: ∠OJE = ∠OJP0/4")
        if target == "R":
            pts = line_circle(*bisector, *small)
            con.mark("Q2", pts[pts[:, 1].argmin()])
    con.mark("E", line_line(*bisector, *x_axis))

    # F: ∠EJF = 45°, half of a right angle at J
    c, r = con.circle("J", "E", "circle about J through E")
    con.mark("E'", line_circle(con.points["E"], j, c, r)[1])  # the other end of the diameter through E
    pts = circle_circle(
        *con.circle("E", "E'", "circle about E through E'"), *con.circle("E'", "E", "circle about E' through E")
    )
    con.points["W1"], con.points["W2"] = pts
    perp = con.line("W1", "W2", "perpendicular to JE at J")
    pts = line_circle(*perp, c, r)
    con.points["W"] = pts[pts[:, 0].argmin()]  # on the circle about J, left of the vertical
    pts = circle_circle(
        *con.circle("E", "W", "circle about E through W"), *con.circle("W", "E", "circle about W through E")
    )
    con.points["V1"], con.points["V2"] = pts
    con.mark("F", line_line(*con.line("V1", "V2", "bisect: ∠EJF = 45°"), *x_axis))

    # K: the circle on the diameter FP0 meets OB
    pts = circle_circle(
        *con.circle("F", "P0", "circle about F through P0"), *con.circle("P0", "F", "circle about P0 through F")
    )
    con.points["G1"], con.points["G2"] = pts
    con.mark("Mf", line_line(*con.line("G1", "G2", "midpoint of FP0"), *x_axis))
    pts = line_circle(*y_axis, *con.circle("Mf", "P0", "circle on the diameter FP0"))
    con.mark("K", pts[pts[:, 1].argmax()])

    # N3, N5: the circle about E through K meets the diameter
    n5, n3 = line_circle(*x_axis, *con.circle("E", "K", "circle about E through K"))  # sorted along O → P0
    con.mark("N5", n5)
    con.mark("N3", n3)
    _perpendicular_foot_vertex(con, "N3", "P3", gamma)
    _perpendicular_foot_vertex(con, "N5", "P5", gamma)

    # P4 halves the arc P3P5
    pts = circle_circle(
        *con.circle("P3", "P5", "circle about P3 through P5"), *con.circle("P5", "P3", "circle about P5 through P3")
    )
    con.points["H1"], con.points["H2"] = pts
    pts = line_circle(*con.line("H1", "H2", "bisect the arc P3P5"), *gamma)
    con.mark("P4", pts[(pts - con.points["P3"]).norm(dim=1).argmin()])

    # step the side around: the circle about P_k through P_(k−1) meets Γ again at P_(k+1)
    def step(centre: int, back: int, new: int, label: str = "") -> None:
        c, r = con.circle(f"P{centre}", f"P{back}", label or f"circle about P{centre} through P{back}")
        name = f"P{new}" if f"P{new}" not in con.points else f"P{new}'"
        con.mark(name, _far(circle_circle(*gamma, c, r), con.points[f"P{back}"]))

    for k in (5, 6, 7):
        step(k, k - 1, k + 1)  # P6, P7, P8
    for k in (3, 2, 1):
        step(k, k + 1, k - 1)  # P2, P1 and back to P0 (a check)
    for k in range(1, 9):  # the circle about P0 through P_k meets Γ again at the mirror image P_(17−k)
        step(0, k, 17 - k, f"circle about P0 through P{k}: mirror P{17 - k}")

    vertices = torch.stack([con.points[f"P{k}"] for k in range(17)])
    return {"points": con.points, "steps": con.steps, "vertices": vertices, "gamma": gamma}


# ---------------------------------------------------------------- Gauss–Wantzel


def constructibility(n_max: int, device: torch.device | None = None) -> dict:
    """Which regular n-gons, 1 ≤ n ≤ n_max, are constructible, two ways at once.   → dict of (N,) tensors"""
    n = torch.arange(1, n_max + 1, device=device)
    phi = euler_phi(n_max, device)[1:]
    two_power, divides, fermat = fermat_factorisation(n)
    return {
        "n": n,
        "phi": phi,
        "phi_test": is_power_of_two(phi) & (n >= 3),
        "fermat": fermat,
        "two_power": two_power,
        "divides": divides,
    }
