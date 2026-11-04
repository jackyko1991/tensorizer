"""Assemble the Hodge diamonds shown frame by frame, and the Picard-number / Hodge-class tables of Fermat varieties."""

import torch

from .equation import (
    euler_characteristic,
    euler_formula,
    fermat_diamond,
    hodge_class_count,
    kunneth,
    line_gram,
    torus_diamond,
)

NAMES = {
    (1, 3): "plane cubic (elliptic curve)",
    (2, 2): "quadric surface",
    (2, 3): "cubic surface",
    (2, 4): "quartic K3 surface",
    (3, 5): "quintic Calabi–Yau threefold",
    (4, 3): "cubic fourfold",
}


def cases(d_max: int, g_max: int) -> list[dict]:
    """Fermat hypersurfaces, complex tori and products: label, diamond, χ and (for even n) the Hodge-class count."""
    out = []
    for n, degrees in (
        (1, (3, 4, 5)),
        (2, range(2, min(d_max, 7) + 1)),
        (3, (3, 4, 5)),
        (4, range(3, min(d_max, 6) + 1)),
    ):
        for d in degrees:
            h = fermat_diamond(n, d)
            label = NAMES.get((n, d), f"Fermat hypersurface, n = {n}, d = {d}")
            out.append(
                {
                    "label": f"{label}: $x_0^{{{d}}} + \\dots + x_{{{n + 1}}}^{{{d}}} = 0$",
                    "h": h,
                    "chi": euler_characteristic(h),
                    "chi_formula": euler_formula(n, d),
                    "hodge": hodge_class_count(n, d) if n % 2 == 0 else None,
                    "fermat": (n, d),
                }
            )
    for g in range(1, g_max + 1):
        h = torus_diamond(g)
        out.append({"label": f"complex torus of dimension {g}", "h": h, "chi": euler_characteristic(h), "torus": g})
    k3, e, c3 = fermat_diamond(2, 4), torus_diamond(1), fermat_diamond(1, 4)
    for label, h in (
        ("K3 × elliptic curve (Künneth)", kunneth(k3, e)),
        ("genus-3 curve × genus-3 curve", kunneth(c3, c3)),
    ):
        out.append({"label": label, "h": h, "chi": euler_characteristic(h)})
    return out


def surface_table(d_max: int) -> dict:
    """For Fermat surfaces d = 3 … d_max: h^{1,1}, Picard number ρ (Shioda) and the rank of the lattice of lines."""
    ds = list(range(3, d_max + 1))
    h11 = [int(fermat_diamond(2, d)[1, 1]) for d in ds]
    rho = [hodge_class_count(2, d) for d in ds]
    ranks, signs = [], []
    for d in ds:
        ev = torch.linalg.eigvalsh(line_gram(d).double())
        tol = 1e-8 * ev.abs().max()
        ranks.append(int((ev.abs() > tol).sum()))
        signs.append((int((ev > tol).sum()), int((ev < -tol).sum())))
    return {"d": ds, "h11": h11, "rho": rho, "lines": ranks, "signature": signs}


def fourfold_table(d_max: int) -> dict:
    """For Fermat fourfolds d = 3 … d_max: h^{2,2} and the dimension of rational Hodge classes in H⁴."""
    ds = list(range(3, d_max + 1))
    return {
        "d": ds,
        "h22": [int(fermat_diamond(4, d)[2, 2]) for d in ds],
        "hodge": [hodge_class_count(4, d) for d in ds],
    }


def euler_check(n_max: int, d_max: int) -> torch.Tensor:
    """(n, d, χ from the diamond, χ from the formula) for every Fermat hypersurface in range.   → (K, 4)"""
    rows = [
        (n, d, euler_characteristic(fermat_diamond(n, d)), euler_formula(n, d))
        for n in range(1, n_max + 1)
        for d in range(1, d_max + 1)
    ]
    return torch.tensor(rows, dtype=torch.int64)
