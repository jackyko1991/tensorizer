"""Velocity Verlet for all state points at once: Berendsen-thermostatted equilibration, then an NVE production run."""

import torch

from problems._common.progress import steps

from . import equation as eq


def evaluate(x: torch.Tensor, L: torch.Tensor):
    """Forces (S, N, 3), potential energy (S,), r² (S, N, N) and f/r (S, N, N) for the current positions."""
    d = eq.minimum_image(x, L)
    r2, u, f_over_r = eq.lj_pairs(d)
    return eq.forces(d, f_over_r), eq.potential_energy(u), r2, f_over_r


def verlet(x, v, L, dt, n_steps, T_target=None, tau_T=0.1, record=None, label="MD steps"):
    """Velocity Verlet: v += ½ dt F;  x += dt v;  F(x);  v += ½ dt F   (mass 1).

    With T_target (S,), a Berendsen thermostat rescales v by √(1 + dt/τ_T (T_target/T - 1)) after each step.
    record(step, x, v, U, r2, f_over_r) is called after every step (for measurements). Returns x, v.
    """
    f, U, r2, f_over_r = evaluate(x, L)
    for step in steps(n_steps, label):
        v = v + 0.5 * dt * f
        x = torch.remainder(x + dt * v, L[:, None, None])  # wrap back into the box
        f, U, r2, f_over_r = evaluate(x, L)
        v = v + 0.5 * dt * f
        if T_target is not None:
            T = eq.kinetic_temperature(v)
            v = v * torch.sqrt(1 + dt / tau_T * (T_target / T - 1)).clamp(0.8, 1.25)[:, None, None]
        if record is not None:
            record(step, x, v, U, r2, f_over_r)
    return x, v


def equilibrate(x, v, L, T, T_start, dt, n_steps):
    """First third at T_start (melts the lattice of the fluids), then at the target T, both with the thermostat."""
    hot = n_steps // 3
    x, v = verlet(x, v, L, dt, hot, T_target=T_start, label="melting")
    return verlet(x, v, L, dt, n_steps - hot, T_target=T, label="equilibration")


def production(x, v, L, rho, dt, n_steps, r_max, n_bins=120, sample_every=10, n_frames=60):
    """NVE run: total energy, temperature and virial pressure every step; g(r) and frames every few steps."""
    S, N, _ = x.shape
    out = {"E": [], "T": [], "P": [], "frames": [], "frame_step": [], "g_running": []}
    counts = torch.zeros(S, n_bins, device=x.device)
    samples = [0]
    frame_every = max(1, n_steps // n_frames)

    def record(step, x, v, U, r2, f_over_r):
        T = eq.kinetic_temperature(v)
        out["E"].append((U + 0.5 * torch.einsum("sia,sia->s", v, v)) / N)  # energy per atom
        out["T"].append(T)
        out["P"].append(eq.virial_pressure(r2, f_over_r, T, rho, L))
        if step % sample_every == 0:
            counts.add_(eq.pair_histogram(r2, r_max, n_bins))
            samples[0] += 1
        if step % frame_every == 0:
            out["frames"].append(x.clone())
            out["frame_step"].append(step)
            out["g_running"].append(eq.radial_distribution(counts, samples[0], N, rho, r_max)[0])

    x, v = verlet(x, v, L, dt, n_steps, record=record, label="production")
    g, r = eq.radial_distribution(counts, samples[0], N, rho, r_max)
    return {
        "E": torch.stack(out["E"]),  # (K, S)
        "T": torch.stack(out["T"]),
        "P": torch.stack(out["P"]),
        "frames": torch.stack(out["frames"]),  # (F, S, N, 3)
        "frame_step": out["frame_step"],
        "g_running": torch.stack(out["g_running"]),  # (F, S, n_bins)
        "g": g,  # (S, n_bins)
        "r": r,  # (n_bins,)
        "x": x,
        "v": v,
    }


def first_peak(r: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Position of the highest point of g(r) for each state point.   (S,)"""
    return r[g.argmax(-1)]
