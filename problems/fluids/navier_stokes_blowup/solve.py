"""RK4 time stepping of the whole batch of runs (one viscosity each) in Fourier space, with diagnostics every step."""

import torch

from problems._common.progress import steps

from .equation import curl_hat, energy, enstrophy, max_vorticity, rhs, spectrum


def simulate(
    u0: torch.Tensor,
    nu: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    mask: torch.Tensor,
    t_max: float,
    dt: float,
    n_save: int,
) -> dict:
    """u0 (B, 3, N, N, N) real, nu (B,). Returns the time series E, Ω, ‖ω‖_∞ (T, B), n_save snapshots of max_z |ω|
    (S, B, N, N) with their spectra (S, B, K), the Euler run's full ω field at the same times (S, 3, N, N, N), and the
    final û."""
    n = u0.shape[-1]
    shape = (n, n, n)
    u_hat = torch.fft.rfftn(u0, dim=(-3, -2, -1)) * mask
    n_steps = max(1, round(t_max / dt))
    dt = t_max / n_steps
    save_at = {round(s * n_steps / max(1, n_save - 1)) for s in range(n_save)}
    t, e, z, wm, slices, spectra, t_save, w_first = [], [], [], [], [], [], [], []

    def record(step: int) -> None:
        u = torch.fft.irfftn(u_hat, s=shape, dim=(-3, -2, -1))
        w = torch.fft.irfftn(curl_hat(u_hat, k), s=shape, dim=(-3, -2, -1))
        t.append(step * dt)
        e.append(energy(u))
        z.append(enstrophy(w))
        wm.append(max_vorticity(w))
        if step in save_at:
            t_save.append(step * dt)
            slices.append(torch.sqrt((w**2).sum(dim=1)).amax(dim=-1))  # max over z of |ω|: (B, N, N)
            spectra.append(spectrum(u_hat, k, k2))
            w_first.append(w[0].float())  # first run of the batch (Euler), for the 3D view

    record(0)
    for step in steps(n_steps, "RK4 steps"):
        k1 = rhs(u_hat, nu, k, k2, mask)
        k2_ = rhs(u_hat + 0.5 * dt * k1, nu, k, k2, mask)
        k3 = rhs(u_hat + 0.5 * dt * k2_, nu, k, k2, mask)
        k4 = rhs(u_hat + dt * k3, nu, k, k2, mask)
        u_hat = u_hat + dt / 6 * (k1 + 2 * k2_ + 2 * k3 + k4)
        record(step + 1)

    return {
        "t": torch.tensor(t, dtype=torch.float64),
        "energy": torch.stack(e),
        "enstrophy": torch.stack(z),
        "max_vorticity": torch.stack(wm),
        "t_save": torch.tensor(t_save, dtype=torch.float64),
        "slices": torch.stack(slices),
        "spectra": torch.stack(spectra),
        "w_first": torch.stack(w_first),
        "u_hat": u_hat,
    }
