r"""Integrate the Boltzmann equations for the whole (ε, K) grid at once with an exponential integrator.

For strong washout the rates D and W reach thousands per unit z, so an explicit method would need tiny steps. Both
equations are linear, of the form dy/dz = −a y + s, and over one step with a, s frozen at the midpoint the exact
solution is y_{n+1} = y_n e^{−a h} + s h (1 − e^{−a h})/(a h). This is stable for any step and exact in both limits.
The neutrino equation is written for the departure from equilibrium Δ = N_N − N^{eq}:

    dΔ/dz = −D Δ − dN^{eq}/dz,        dN_{B−L}/dz = ε D Δ − W N_{B−L}.
"""

import torch

from problems._common.progress import steps

from . import equation


def solve(z: torch.Tensor, eps: torch.Tensor, K: torch.Tensor, initial: str = "thermal", equilibrium: bool = False):
    """Abundances along z for every model: eps (E, 1) and K (1, K) broadcast to (E, K).

    initial: "thermal" (N_N = N^eq at the start) or "zero" (no heavy neutrinos yet, produced by inverse decays).
    equilibrium=True holds N_N = N^eq by hand (infinitely fast interactions), the test of Sakharov's third condition.
    Returns N_N (1, K, T), N_{B−L} (E, K, T) and the final N_{B−L} (E, K).
    """
    shape = torch.broadcast_shapes(eps.shape, K.shape)
    delta = torch.zeros_like(K) if initial == "thermal" else -equation.n_eq(z[0]) * torch.ones_like(K)  # (1, K)
    n_bl = torch.zeros(shape, dtype=z.dtype, device=z.device)  # (E, K)
    deltas, n_bls = [delta], [n_bl]
    for i in steps(len(z) - 1, "z steps"):
        h = z[i + 1] - z[i]
        zm = (z[i + 1] * z[i]).sqrt()  # midpoint in ln z
        D, W = equation.decay_rate(zm, K), equation.washout(zm, K)  # (1, K)
        if equilibrium:
            new_delta = torch.zeros_like(delta)
        else:
            new_delta = delta * torch.exp(-D * h) - equation.dn_eq_dz(zm) * h * equation.phi(D * h)
        source = eps * D * (delta + new_delta) / 2  # (E, K): ε D Δ at the midpoint
        n_bl = n_bl * torch.exp(-W * h) + source * h * equation.phi(W * h)
        delta = new_delta
        deltas.append(delta)
        n_bls.append(n_bl)
    n_n = torch.stack(deltas, -1) + equation.n_eq(z)  # (1, K, T)
    n_bl_t = torch.stack(n_bls, -1)  # (E, K, T)
    return {"z": z, "n_n": n_n, "n_bl": n_bl_t, "n_bl_final": n_bl_t[..., -1]}


def efficiency(n_bl_final: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    r"""κ_f = N_{B−L}^f / (\tfrac34 ε): the share of the maximal asymmetry that survives washout."""
    return n_bl_final / (0.75 * eps)


def baryon_to_photon(n_bl_final: torch.Tensor) -> torch.Tensor:
    return equation.baryon_to_photon(n_bl_final)


def required_epsilon(kappa: torch.Tensor, eta: float) -> torch.Tensor:
    r"""ε needed to reach η: η = (a_{sph}/f) \tfrac34 ε κ_f solved for ε."""
    return eta / (equation.A_SPH / equation.DILUTION * 0.75 * kappa)
