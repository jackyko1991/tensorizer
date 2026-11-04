"""Light clocks, Einstein's rod and train, the 1905 mass-loss argument, and the γ of Frisch & Smith's muons."""

import torch

from . import equation, tensors


def light_clocks(beta: torch.Tensor, D: float, t_max: float, n: int):
    """Photon paths in the lab for clocks moving at speeds beta (B,): times (T,), x and y (B, T).

    The photon moves at c = 1 along a zigzag: horizontal speed β (it rides with the clock), vertical speed √(1 − β²).
    """
    t = torch.linspace(0.0, t_max, n, dtype=torch.float64)
    x = beta[:, None] * t  # (B, T)
    phase = torch.remainder(t * torch.sqrt(1 - beta[:, None] ** 2) / D, 2.0)  # 0 → 1 up, 1 → 2 down
    y = D * (1 - (phase - 1).abs())
    return t, x, y


def count_ticks(y: torch.Tensor) -> torch.Tensor:
    """Completed round trips (returns to the bottom mirror) along the last axis: (B,)."""
    dy = y.diff(dim=-1)
    return ((dy[..., :-1] < 0) & (dy[..., 1:] >= 0)).sum(dim=-1)


def contracted_length(L0: float, beta: torch.Tensor) -> torch.Tensor:
    """Length L of a clock lying along the motion that ticks together with an identical clock standing across it.

    Across: 2 L0 γ (the light-clock diagonal). Along, length L: L/(1 − β) + L/(1 + β) = 2 L γ². Equal ⇒ L = L0/γ.
    """
    g = equation.gamma(beta)
    across = 2 * L0 * g
    along_per_length = 1 / (1 - beta) + 1 / (1 + beta)
    return across / along_per_length


def kinetic_energy_drop(L: float, beta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Einstein (1905): K0 − K1 = (energy of the light in the moving frame) − (in the rest frame) = γL − L."""
    return equation.pulse_energies(L, beta, theta).sum(dim=-1) - L


def annihilate_at_rest(m: float) -> torch.Tensor:
    """e⁺e⁻ at rest → two back-to-back photons: (2, 4) four-momenta (E, p_x, p_y, p_z) in MeV."""
    return torch.tensor([[m, m, 0.0, 0.0], [m, -m, 0.0, 0.0]], dtype=torch.float64)


def muon_flight(data: dict, dEdx: float, n_speeds: int = 41, n_z: int = 2001) -> dict:
    """Muons fly from the summit down to sea level, slowing as they ionise the air: lab and proper flight times.

    Batched over the published speed band at the summit (B,) and over heights (Z,).
    """
    z_top, z_bot = data["altitude"]
    beta_top = torch.linspace(float(data["beta_min"][0]), float(data["beta_max"][0]), n_speeds, dtype=torch.float64)
    z = torch.linspace(float(z_top), float(z_bot), n_z, dtype=torch.float64)  # (Z,) going down
    column = tensors.atmosphere(z_top) - tensors.atmosphere(z)  # (Z,) g/cm² crossed so far
    E = tensors.M_MUON * equation.gamma(beta_top)[:, None] - dEdx * column  # (B, Z) MeV
    g = E / tensors.M_MUON
    beta = torch.sqrt(1 - 1 / g**2)
    s = z_top - z  # (Z,) distance flown so far, m
    t_lab = torch.trapezoid(1 / (beta * tensors.C), s, dim=-1)  # (B,) μs: ∫ dz / v
    tau = torch.trapezoid(1 / (beta * g * tensors.C), s, dim=-1)  # (B,) μs: ∫ dz / (γ v), the muon's own clock
    return {"z": z, "gamma": g, "beta": beta, "t_lab": t_lab, "tau": tau}


def muon_gamma(data: dict, dEdx: float = 1.82) -> dict:
    """Fit the time-dilation factor γ to the two count rates (autograd Newton steps on χ²), compare with the speeds."""
    flight = muon_flight(data, dEdx)
    t_lab = flight["t_lab"].mean()
    gamma_pred = float(t_lab / flight["tau"].mean())  # dilation the measured speeds predict

    N_top, N_bot = data["rate"]
    r = N_bot / N_top  # surviving fraction
    sigma_r = r * torch.sqrt((data["err"] / data["rate"]).pow(2).sum())

    def chi2(g):
        return ((r - torch.exp(-t_lab / (g * tensors.TAU_MUON))) / sigma_r) ** 2

    g = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    for _ in range(30):  # Newton's method with autograd derivatives
        (d1,) = torch.autograd.grad(chi2(g), g, create_graph=True)
        (d2,) = torch.autograd.grad(d1, g)
        with torch.no_grad():
            g -= d1 / d2
    (d1,) = torch.autograd.grad(chi2(g), g, create_graph=True)
    (d2,) = torch.autograd.grad(d1, g)
    return {
        "gamma_fit": float(g.detach()),
        "gamma_err": float(torch.sqrt(2 / d2)),  # Δχ² = 1
        "gamma_pred": gamma_pred,
        "t_lab": float(t_lab),
        "tau_rest": float(t_lab) / float(g.detach()),
        "no_dilation_rate": float(N_top * torch.exp(-t_lab / tensors.TAU_MUON)),
        "beta_bottom": (float(flight["beta"][:, -1].min()), float(flight["beta"][:, -1].max())),
        "flight": flight,
    }
