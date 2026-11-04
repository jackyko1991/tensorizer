"""Physics checks, written first (TDD): what any correct spectral Navier–Stokes / Euler code must satisfy.

Run with `pytest problems/fluids/navier_stokes_blowup/test.py`. Each test is a law of incompressible flow, not an
implementation detail: the velocity stays divergence-free, kinetic energy decays at exactly dE/dt = −2ν Ω (and is
conserved by Euler), the 2D Taylor–Green vortex decays as e^{−2νt}, and the 3D Taylor–Green flow starts with the
analytic acceleration obtained by hand from the pressure p₀ = (cos 2x + cos 2y)(cos 2z + 2)/16.
"""

import math

import torch

from . import equation, plot, solve, tensors

CPU = torch.device("cpu")
N = 16


def test_taylor_green_is_divergence_free_and_stays_so():
    u0 = tensors.taylor_green(N, CPU)[None]  # (1, 3, N, N, N)
    k, k2, mask = tensors.wavenumbers(N, CPU)
    out = solve.simulate(u0, torch.tensor([0.01], dtype=torch.float64), k, k2, mask, t_max=1.0, dt=0.02, n_save=3)
    u_hat = out["u_hat"]  # (1, 3, N, N, M) at the end
    div = torch.einsum("ixyz,bixyz->bxyz", k.to(u_hat.dtype), u_hat)  # k_i û_i (the i of i k·û dropped)
    assert div.abs().max() < 1e-12 * u_hat.abs().max()  # ∇·u = 0 to round-off


def test_energy_decays_at_twice_nu_times_enstrophy():
    u0 = tensors.taylor_green(N, CPU)[None].repeat(3, 1, 1, 1, 1)
    nu = torch.tensor([0.0, 0.05, 0.1], dtype=torch.float64)
    k, k2, mask = tensors.wavenumbers(N, CPU)
    out = solve.simulate(u0, nu, k, k2, mask, t_max=1.0, dt=0.005, n_save=3)
    e, z, t = out["energy"], out["enstrophy"], out["t"]  # (T, B)
    de_dt = (e[2:] - e[:-2]) / (t[2:] - t[:-2])[:, None]  # central difference in time
    rhs = -2 * nu * z[1:-1]
    assert torch.allclose(de_dt, rhs, atol=1e-6, rtol=1e-4)  # dE/dt = −2ν Ω with Ω = ½⟨|ω|²⟩
    assert abs(e[-1, 0] / e[0, 0] - 1) < 1e-8  # Euler (ν = 0) conserves energy


def test_2d_taylor_green_decays_exactly():
    """u = (sin x cos y, −cos x sin y, 0) e^{−2νt} is an exact Navier–Stokes solution: (u·∇)u is a pure gradient."""
    nu = 0.1
    u0 = tensors.taylor_green_2d(N, CPU)[None]
    k, k2, mask = tensors.wavenumbers(N, CPU)
    out = solve.simulate(u0, torch.tensor([nu], dtype=torch.float64), k, k2, mask, t_max=2.0, dt=0.01, n_save=2)
    u_end = torch.fft.irfftn(out["u_hat"], s=(N, N, N), dim=(-3, -2, -1))
    assert torch.allclose(u_end, u0 * math.exp(-2 * nu * 2.0), atol=1e-9)


def test_3d_taylor_green_early_time_matches_analytic_acceleration():
    """∂u/∂t at t = 0 from −(u·∇)u − ∇p with p₀ = (cos 2x + cos 2y)(cos 2z + 2)/16, plus viscosity −3ν u₀."""
    nu, t = 0.02, 1e-3
    x = tensors.grid(N, CPU)  # (N,)
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    accel = torch.stack(
        [
            -torch.sin(2 * X) * torch.cos(2 * Z) / 8,
            -torch.sin(2 * Y) * torch.cos(2 * Z) / 8,
            (torch.cos(2 * X) + torch.cos(2 * Y)) * torch.sin(2 * Z) / 8,
        ]
    )
    u0 = tensors.taylor_green(N, CPU)
    k, k2, mask = tensors.wavenumbers(N, CPU)
    out = solve.simulate(u0[None], torch.tensor([nu], dtype=torch.float64), k, k2, mask, t_max=t, dt=t / 4, n_save=2)
    u_t = torch.fft.irfftn(out["u_hat"][0], s=(N, N, N), dim=(-3, -2, -1))
    measured = (u_t - u0 * math.exp(-3 * nu * t)) / t
    assert (measured - accel).abs().max() < 5e-3  # agrees up to the O(t) second-order term


def test_enstrophy_grows_for_euler_and_viscosity_tames_it():
    u0 = tensors.taylor_green(N, CPU)[None].repeat(2, 1, 1, 1, 1)
    nu = torch.tensor([0.0, 0.1], dtype=torch.float64)
    k, k2, mask = tensors.wavenumbers(N, CPU)
    out = solve.simulate(u0, nu, k, k2, mask, t_max=3.0, dt=0.02, n_save=3)
    z, wmax = out["enstrophy"], out["max_vorticity"]
    assert z[-1, 0] > 2 * z[0, 0]  # vortex stretching: Euler enstrophy grows
    assert z[-1, 1] < z[-1, 0] and wmax[-1, 1] < wmax[-1, 0]  # viscosity regularises
    assert abs(z[0, 0] - 3 / 8) < 1e-12  # E₀ = 1/8 and every mode has |k|² = 3, so Ω₀ = 3 E₀ = 3/8
    bkm = equation.bkm_integral(out["t"], wmax)  # ∫ ‖ω‖_∞ dt
    assert torch.all(bkm[1:] >= bkm[:-1])


def test_vorticity_iso_surface_vtk_parses_back(tmp_path):
    """The interactive result: a closed iso-surface |ω| = ½ max|ω|₀ per frame, written as VTK PolyData and read back."""
    u0 = tensors.taylor_green(N, CPU)[None]
    k, k2, mask = tensors.wavenumbers(N, CPU)
    out = solve.simulate(u0, torch.tensor([0.0], dtype=torch.float64), k, k2, mask, t_max=1.0, dt=0.05, n_save=3)
    pvd = plot.export_vtk(out, None, tmp_path, n_frames=3)
    assert pvd.read_text().count("<DataSet") == 3
    first = plot.read_vtp(pvd.parent / "iso_000.vtp")
    assert first["Polys"] > 0 and first["Points"] == len(first["arrays"]["vorticity"])
    assert first["connectivity"].max() < first["Points"] and len(first["offsets"]) == first["Polys"]
    level = 0.5 * out["w_first"][0].norm(dim=0).max().item()
    assert first["arrays"]["vorticity"].min() > 0.9 * level  # one step inside the surface |ω| exceeds the level
