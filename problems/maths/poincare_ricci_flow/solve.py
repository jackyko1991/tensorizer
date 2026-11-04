"""Time stepping of both flows, and the surgery step that cuts a pinching neck and glues in a round cap."""

import math

import torch

from problems._common.progress import steps

from . import tensors
from .equation import area, gauss_curvature, neck_radius, profile, s3_rhs, surface_rhs


def flow_surface(u0: torch.Tensor, w: torch.Tensor, lap: torch.Tensor, t_max: float, n_save: int) -> dict:
    """Normalised Ricci flow on S² for a batch of conformal factors u0 (B, N), RK4 with the explicit stability limit
    dt ≈ 0.4 h² min e^{2u} / 2. Returns t (S,), u and K (S, B, N), area (S, B)."""
    n = u0.shape[-1]
    h = math.pi / n
    u = u0.clone()
    dt0 = 0.2 * h**2 * float(torch.exp(2 * u0).min())
    n_steps = max(1, math.ceil(t_max / dt0))
    dt = t_max / n_steps
    save_at = {round(k * n_steps / max(1, n_save - 1)) for k in range(n_save)}
    t, us = [], []
    for step in steps(n_steps + 1, "surface flow"):
        if step in save_at:
            t.append(step * dt)
            us.append(u)
        if step == n_steps:
            break
        k1 = surface_rhs(u, w, lap)
        k2 = surface_rhs(u + 0.5 * dt * k1, w, lap)
        k3 = surface_rhs(u + 0.5 * dt * k2, w, lap)
        k4 = surface_rhs(u + dt * k3, w, lap)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    u_all = torch.stack(us)  # (S, B, N)
    return {
        "t": torch.tensor(t, dtype=torch.float64),
        "u": u_all,
        "K": torch.stack([gauss_curvature(v, lap) for v in u_all]),
        "area": torch.stack([area(v, w) for v in u_all]),
    }


def flow_s3(
    u: torch.Tensor,
    grid: tuple,
    t_max: float,
    psi_stop: float,
    save_every: int = 20,
    radius_stop: float = 0.0,
) -> dict:
    """Ricci flow of g = e^{2u}(dx² + sin²x g_{S²}) for a batch u (B, N), forward Euler with dt = 0.2 min (h x_η e^u)²
    (the explicit diffusion limit). Stops at t_max, when the neck radius drops below psi_stop (time for surgery) or
    when max ψ < radius_stop (the piece is about to vanish). Returns per-step t and neck (T,) and snapshots of u
    (S, B, N) at t_save (S,)."""
    x, xe, _, h = grid
    u = u.clone()
    t, t_steps, ts, necks, us = 0.0, [], [], [], []
    step = 0
    while True:
        psi = torch.exp(u) * torch.sin(x)
        neck = neck_radius(psi).min().item()
        necks.append(neck)
        t_steps.append(t)
        done = t >= t_max - 1e-15 or neck < psi_stop or psi.max().item() < radius_stop or not math.isfinite(t)
        if step % save_every == 0 or done:
            ts.append(t)
            us.append(u.clone())
        if done:
            break
        dt = min(0.2 * h**2 * (xe**2 * torch.exp(2 * u)).min().item(), t_max - t)  # (physical cell size)²
        u = u + dt * s3_rhs(u, grid)
        t += dt
        step += 1
    return {
        "t": torch.tensor(t_steps, dtype=torch.float64),
        "neck": torch.tensor(necks, dtype=torch.float64),
        "t_save": torch.tensor(ts, dtype=torch.float64),
        "u": torch.stack(us),
    }


def surgery(u: torch.Tensor, grid: tuple, keep: float = 0.4) -> torch.Tensor:
    """A cartoon of Perelman's surgery: on the left piece, cut where ψ first falls to keep·max ψ (discarding the
    whole high-curvature neck region) and glue in a round spherical cap that meets the profile tangentially:
    ψ = R sin(θ₀ + (s − s_cut)/R) with R sin θ₀ = ψ_cut and cos θ₀ = ψ_s(cut), ending at θ = π. Then back to
    conformal form. (1, N) → (1, N). By symmetry the right piece is a mirror copy and evolves identically."""
    s, psi = profile(u, grid)  # (1, N) at the cell centres
    s, psi = s[0], psi[0]
    mid = psi[1:-1]
    is_min = (mid < psi[:-2]) & (mid < psi[2:])
    neck = int(torch.where(is_min, mid, torch.inf).argmin()) + 1  # cell of the neck
    top = int(psi[:neck].argmax())  # the left bulb
    j = top + int(torch.nonzero(psi[top:] < keep * psi[top])[0])  # first cell past the bulb below the cut level
    r = psi[j].item()
    slope = ((psi[j + 1] - psi[j - 1]) / (s[j + 1] - s[j - 1])).clamp(-0.99, 0.99).item()  # ψ_s at the cut
    theta0 = math.acos(slope)
    big_r = r / math.sin(theta0)
    zero = torch.zeros(1, dtype=s.dtype, device=s.device)
    s_cap = torch.linspace(0, big_r * (math.pi - theta0), 400, dtype=s.dtype, device=s.device)[1:]
    s_all = torch.cat([zero, s[: j + 1], s[j] + s_cap])
    psi_all = torch.cat([zero, psi[: j + 1], big_r * torch.sin(theta0 + s_cap / big_r)])
    psi_all[-1] = 0.0
    fine = torch.linspace(0, s_all[-1].item(), 20001, dtype=s.dtype, device=s.device)
    k = torch.searchsorted(s_all, fine).clamp(1, len(s_all) - 1)  # resample finely before converting
    frac = (fine - s_all[k - 1]) / (s_all[k] - s_all[k - 1])
    psi_fine = psi_all[k - 1] + frac * (psi_all[k] - psi_all[k - 1])
    return tensors.conformal_from_profile(fine, psi_fine, grid[0])
