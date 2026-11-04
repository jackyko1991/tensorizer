"""Solvers: Lane–Emden polytropes (RK4, batched over n), the brown-dwarf temperature maximum, and the ignition
depth of accreted shells (fixed-point iteration for the base temperature + autograd instability test)."""

import math

import torch

from problems._common.integrators import integrate

from . import equation as eq
from . import tensors


def polytropes(n: torch.Tensor, dxi: float = 1e-3, xi_max: float = 9.0) -> dict:
    """Integrate Lane–Emden for every index in n at once; return the surface ξ₁ and structure constants (B,)."""
    xi0 = 1e-3
    traj = integrate(
        lambda y, t: eq.lane_emden_rhs(y, xi0 + t, n), tensors.lane_emden_start(n, xi0), dxi, int(round(xi_max / dxi))
    )  # fmt: skip  (T, B, 2)
    xi = xi0 + dxi * torch.arange(traj.shape[0], dtype=n.dtype)
    theta, dtheta = traj[..., 0], traj[..., 1]
    k = (theta > 0).sum(0) - 1  # last step inside the star, (B,)
    cols = torch.arange(len(n))
    frac = theta[k, cols] / (theta[k, cols] - theta[k + 1, cols])  # linear interpolation to θ = 0
    xi1 = xi[k] + frac * dxi
    d1 = dtheta[k, cols] + frac * (dtheta[k + 1, cols] - dtheta[k, cols])
    return {
        "n": n,
        "xi1": xi1,
        "dtheta1": d1,
        "omega": -(xi1**2) * d1,  # mass constant ω_n
        "rho_ratio": -xi1 / (3 * d1),  # ρ_c / ρ̄
        "W": 1 / (4 * math.pi * (n + 1) * d1**2),  # P_c = W G M² / R⁴
    }


def brown_dwarf_max_temperature(M: torch.Tensor, poly: dict) -> torch.Tensor:
    """Highest central temperature a contracting n = 3/2 body of mass M ever reaches: (M,) → (M,).

    T_c is evaluated on a (masses × radii) grid by broadcasting, then maximised over the radii.
    """
    i = int(torch.nonzero(poly["n"] == 1.5)[0])
    R = tensors.radius_grid()
    X, Y, Z = tensors.COMPOSITIONS["solar"]
    T = eq.central_temperature(M[:, None], R[None, :], poly["W"][i], poly["rho_ratio"][i], X, Y, Z)  # (M, R)
    return torch.nan_to_num(T, nan=0.0).amax(dim=1)


def crossing(x: torch.Tensor, f: torch.Tensor, level: float) -> float:
    """First x where an increasing f(x) reaches `level` (log–log interpolation)."""
    k = int((f < level).sum()) - 1
    lx, lf = x.log(), f.log()
    t = (math.log(level) - lf[k]) / (lf[k + 1] - lf[k])
    return float(torch.exp(lx[k] + t * (lx[k + 1] - lx[k])))


def base_temperature(c: dict, y: torch.Tensor, P: torch.Tensor, iterations: int = 30) -> tuple:
    """T⁴ = T_floor⁴ + 3κ y F/(a c) with F = ṁ (Q_b + f_comp c_p T) + f_H ½ ε_HCNO y, solved by fixed-point
    iteration (κ and ρ depend on T). c: dict of (B, 1); y, P: (B, Y). Returns T, ρ (B, Y)."""
    mdot = c["Mdot"] / (4 * math.pi * c["R"] ** 2)  # g cm⁻² s⁻¹
    mu = 1 / (2 * c["X"] + 0.75 * c["Y"] + 0.5 * c["Z"])
    cp = 2.5 * eq.K_B / (mu * eq.M_U)
    T = c["T_floor"].expand_as(y).clone()
    for _ in range(iterations):
        rho = eq.density(P, T, c["X"], c["Y"], c["Z"])
        kappa = eq.opacity(rho, T, c["X"], c["Z"])
        flux = mdot * (c["Q_b"] + c["f_comp"] * cp * T) + c["f_H"] * 0.5 * eq.eps_hot_cno(c["Z"]) * y
        T = (c["T_floor"] ** 4 + 3 * kappa * y * flux / (eq.A_RAD * eq.C)) ** 0.25
    return T, eq.density(P, T, c["X"], c["Y"], c["Z"])


def ignition(cases: dict, y: torch.Tensor | None = None) -> dict:
    """Ignition column y_ign and critical accreted mass ΔM = 4πR² y_ign for every case (B,)."""
    y = tensors.column_grid() if y is None else y
    c = {k: v[:, None] for k, v in cases.items() if k != "kind"}  # (B, 1) for broadcasting against y (1, Y)
    g = eq.G * c["M"] / c["R"] ** 2 / torch.sqrt(1 - 2 * eq.G * c["M"] / (c["R"] * eq.C**2))  # GR boost (NS)
    yy = y[None, :].expand(len(cases["M"]), -1)
    P = g * yy  # hydrostatic: pressure = weight of the column above
    T, rho = base_temperature(c, yy, P)
    unstable = eq.instability(rho, T, yy, c["X"], c["Y"], c["Z"]) > 0  # (B, Y)
    k = unstable.double().argmax(dim=1)  # first unstable depth
    rows = torch.arange(len(k))
    y_ign = y[k]
    return {
        "y": y_ign,
        "T": T[rows, k],
        "rho": rho[rows, k],
        "dM": 4 * math.pi * cases["R"] ** 2 * y_ign,
        "ignited": unstable.any(dim=1),
        "T_track": T,
        "unstable": unstable,
    }
