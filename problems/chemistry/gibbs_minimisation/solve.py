"""Batched Newton iteration for the Gibbs minimum (Gordon–McBride / RAND), and the adiabatic flame temperature."""

import math

import torch

from problems._common.progress import steps

from .equation import (
    chemical_potential_log,
    element_totals,
    enthalpy,
    newton_system,
    species_update,
    standard_potential,
)
from .tensors import T0

TRACE = math.log(1e-8)  # species below this mole fraction count as "trace" for the step control


def equilibrium(g: torch.Tensor, A: torch.Tensor, b: torch.Tensor, P: float, n_iter: int = 60, start=None) -> dict:
    """Minimise G/RT for every row of the batch at once.   g (B, S), A (E, S), b (B, E), P in bar.

    Each Newton step solves one (E+1)×(E+1) system per row (torch.linalg.solve on a (B, E+1, E+1) batch) for
    the element potentials π and Δln n_tot, then updates every ln n_j. The step is damped as in NASA's CEA
    code: majors may change by at most a factor e² per step, and trace species may not jump above 1e-4.
    Returns n (B, S), the element potentials pi (B, E) and the last step size.
    """
    B, S = g.shape
    if start is None:  # all species equally abundant, n_tot = Σ b / 2
        ln_ntot = torch.log(b.sum(-1) / 2)
        ln_n = (ln_ntot - math.log(S))[:, None].expand(B, S).clone()
    else:
        ln_n, ln_ntot = start["ln_n"].clone(), start["ln_ntot"].clone()

    for _ in steps(n_iter, "Newton steps"):
        n, n_tot = torch.exp(ln_n), torch.exp(ln_ntot)
        mu = chemical_potential_log(ln_n, ln_ntot, g, P)  # (B, S)
        J, rhs = newton_system(A, n, n_tot, mu, b)
        sol = torch.linalg.solve(J, rhs)  # (B, E+1)
        pi, dln_ntot = sol[:, :-1], sol[:, -1]
        dln_n = species_update(A, mu, pi, dln_ntot)  # (B, S)

        # step control: λ ≤ 1, majors change by ≤ e², trace species stop at mole fraction 1e-4
        major = (ln_n - ln_ntot[:, None]) > TRACE
        big = torch.where(major, dln_n.abs(), torch.zeros_like(dln_n)).amax(-1)
        lam1 = 2.0 / torch.maximum(5 * dln_ntot.abs(), big).clamp_min(1e-300)
        rise = dln_n - dln_ntot[:, None]
        cap = torch.where(~major & (rise > 0), (-(ln_n - ln_ntot[:, None]) - 9.2103) / rise, torch.inf)
        lam = torch.minimum(torch.minimum(lam1, cap.abs().amin(-1)), torch.ones_like(lam1))
        ln_n = ln_n + lam[:, None] * dln_n
        ln_ntot = ln_ntot + lam * dln_ntot

    return {"n": torch.exp(ln_n), "pi": pi, "ln_n": ln_n, "ln_ntot": ln_ntot, "step": (lam[:, None] * dln_n).abs()}


def temperature_sweep(T: torch.Tensor, A: torch.Tensor, b: torch.Tensor, P: float, th: dict, n_iter: int = 60):
    """Equilibrium composition at every temperature of the batch T (B,) for ONE feed b (E,)."""
    g = standard_potential(T, th)  # (B, S)
    return equilibrium(g, A, b.expand(len(T), -1), P, n_iter)


def flame_temperature(A, b, H_react, P, th, n_bisect: int = 26, t_lo: float = 900.0, t_hi: float = 3600.0):
    """Adiabatic flame temperature for every feed row of b (F, E): find T with H_products(T, n_eq(T)) = H_reactants.

    Bisection, all F feeds in parallel; each step warm-starts Newton from the previous composition.
    """
    lo = torch.full_like(H_react, t_lo)
    hi = torch.full_like(H_react, t_hi)
    state = None
    for _ in steps(n_bisect, "bisection"):
        mid = 0.5 * (lo + hi)
        state = equilibrium(standard_potential(mid, th), A, b, P, n_iter=60 if state is None else 12, start=state)
        too_hot = enthalpy(state["n"], mid, th) > H_react  # products hold more enthalpy than was released
        hi = torch.where(too_hot, mid, hi)
        lo = torch.where(too_hot, lo, mid)
    return 0.5 * (lo + hi), state["n"]


def frozen_flame_temperature(n_cold: torch.Tensor, H_react: torch.Tensor, th: dict) -> torch.Tensor:
    """T_ad if the products kept a fixed (cold, undissociated) composition n_cold (F, S): H is linear in T,

    H_react = n_j ΔfH_j + n_j c_pj (T - T0)   ⇒   T = T0 + (H_react - n_j ΔfH_j) / (n_j c_pj).
    """
    h_form = torch.einsum("fj,j->f", n_cold, th["dH"])  # kJ
    heat_capacity = 1e-3 * torch.einsum("fj,j->f", n_cold, th["cp"])  # kJ/K
    return T0 + (H_react - h_form) / heat_capacity


def residuals(A: torch.Tensor, b: torch.Tensor, n: torch.Tensor, pi: torch.Tensor, g: torch.Tensor, P: float):
    """Largest violations of the two optimality conditions: element balance and μ_j/RT = π_i A_ij (majors only)."""
    balance = (element_totals(A, n) - b).abs().max().item()
    ln_n, ln_ntot = torch.log(n), torch.log(n.sum(-1))
    mu = chemical_potential_log(ln_n, ln_ntot, g, P)
    major = (ln_n - ln_ntot[:, None]) > TRACE
    kkt = (mu - torch.einsum("bi,ij->bj", pi, A))[major].abs().max().item()
    return balance, kkt
