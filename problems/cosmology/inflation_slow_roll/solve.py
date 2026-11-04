"""Integrate every potential at once in e-folds (RK4 on a (2, B) state), then read off the pivot-scale predictions."""

import math

import torch

from problems._common.integrators import rk4_step
from problems._common.progress import steps

from . import equation


def integrate(phi0: torch.Tensor, psi0: torch.Tensor, model: dict, dN: float = 0.02, N_max: float = 95.0) -> dict:
    """Trajectories φ, ψ, ε_H of shape (T, B) on the grid N = 0, dN, 2dN, … Each field stops (is frozen) at the
    step where its ε_H first rises through 1 after inflation began; N_end (B,) is that crossing, interpolated."""
    state = torch.stack([phi0, psi0])  # (2, B)
    B = phi0.shape[0]
    started = 0.5 * psi0**2 < 1  # (B,) inflating already? a kinetic start (ε > 1) begins later
    ended = torch.zeros(B, dtype=torch.bool, device=phi0.device)
    N_end = torch.full((B,), math.nan, dtype=phi0.dtype, device=phi0.device)
    history = [state]
    for t in steps(int(N_max / dN), "e-fold steps"):
        alive = (~ended).to(state.dtype)  # (B,) a finished field stops moving
        new = rk4_step(lambda y, _, alive=alive: equation.rhs(y, model) * alive, state, t * dN, dN)
        eps_old, eps_new = 0.5 * state[1] ** 2, 0.5 * new[1] ** 2
        crossing = started & ~ended & (eps_new >= 1)
        N_end = torch.where(crossing, (t + (1 - eps_old) / (eps_new - eps_old)) * dN, N_end)
        ended = ended | crossing
        started = started | (eps_new < 1)
        state = new
        history.append(state)
        if ended.all():
            break
    traj = torch.stack(history)  # (T, 2, B)
    N = torch.arange(len(traj), dtype=phi0.dtype, device=phi0.device) * dN
    phi, psi = traj[:, 0], traj[:, 1]
    return {"N": N, "phi": phi, "psi": psi, "eps": 0.5 * psi**2, "N_end": N_end, "dN": dN}


def at_e_folds(sol: dict, field: torch.Tensor, N_target: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of field (T, B) at times N_target (S, B) → (S, B), with one gather per neighbour."""
    u = (N_target / sol["dN"]).clamp(0, len(sol["N"]) - 1.000001)
    i0 = u.floor().long()
    frac = u - i0
    lo = torch.gather(field, 0, i0)
    hi = torch.gather(field, 0, i0 + 1)
    return lo + frac * (hi - lo)


def predictions(sol: dict, model: dict, N_star: torch.Tensor) -> dict:
    """n_s and r at N_* e-folds before the end, (S,) values of N_* → every entry (S, B)."""
    N_target = sol["N_end"][None, :] - N_star[:, None]  # (S, B)
    # ε₂ along the whole trajectory (from the equation of motion) and ε₃ = d ln ε₂ / dN by central differences
    e2_traj = equation.epsilon_2(sol["psi"], sol["phi"], model)  # (T, B), the batch broadcast over all times
    e3_traj = torch.gradient(e2_traj, spacing=sol["dN"], dim=0)[0] / e2_traj
    phi = at_e_folds(sol, sol["phi"], N_target)
    psi = at_e_folds(sol, sol["psi"], N_target)
    e1, e2, e3 = 0.5 * psi**2, at_e_folds(sol, e2_traj, N_target), at_e_folds(sol, e3_traj, N_target)
    ns, r = equation.ns_r_hubble_flow(e1, e2, e3)
    eps_v, eta_v = equation.slow_roll_potential(phi.flatten(), _repeat(model, len(N_star)))
    eps_v, eta_v = eps_v.view_as(phi), eta_v.view_as(phi)
    ns_v, r_v = equation.ns_r_potential(eps_v, eta_v)
    return {"phi": phi, "eps_H": e1, "e2": e2, "e3": e3, "ns": ns, "r": r,
            "eps_V": eps_v, "eta_V": eta_v, "ns_V": ns_v, "r_V": r_v}  # fmt: skip


def _repeat(model: dict, S: int) -> dict:
    """The same batch of potentials S times, to evaluate (S, B) field values in one flat call."""
    return {**model, "w": model["w"].repeat(S, 1), "kinds": model["kinds"] * S}


def energy_scale(eps_v: torch.Tensor, A_s: float) -> torch.Tensor:
    """V_*^{1/4} in GeV from the scalar amplitude A_s = V / (24π² ε_V M_pl⁴), with M_pl = 2.435e18 GeV."""
    return (24 * math.pi**2 * eps_v * A_s) ** 0.25 * 2.435e18
