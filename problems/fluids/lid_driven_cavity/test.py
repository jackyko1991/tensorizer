"""Physics checks, written first (TDD): what any correct lid-driven cavity code must satisfy.

Run with `pytest problems/fluids/lid_driven_cavity/test.py`. The streamfunction solves ∇²ψ = −ω, the velocity it
gives is divergence-free (incompressibility), and at Re = 100 the flow matches the Ghia et al. benchmark.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def cavity_run():
    n = 41
    y, h = tensors.grid(n, CPU)
    V, lam = tensors.laplacian_eigen(n - 2, h, CPU)
    omega, _ = tensors.at_rest(n, CPU)
    times, omegas, psis = solve.simulate(omega, V, lam, h, 0.01, 1.0, 25.0, save_every=500)
    return {"y": y, "h": h, "V": V, "lam": lam, "omega": omegas[-1], "psi": psis[-1]}


def test_poisson_solver_residual(cavity_run):
    c = cavity_run
    f = torch.randn(39, 39, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    psi = solve.poisson_solve(f, c["V"], c["lam"])
    assert (equation.laplacian(psi, c["h"]) - f).abs().max() < 1e-9 * f.abs().max()
    assert torch.allclose(equation.laplacian(c["psi"], c["h"]), -c["omega"][1:-1, 1:-1], atol=1e-8)  # ∇²ψ = −ω


def test_velocity_is_divergence_free(cavity_run):
    u = equation.velocity(cavity_run["psi"], cavity_run["h"])
    assert equation.divergence(u, cavity_run["h"]).abs().max() < 1e-10  # incompressible: ∇·u = 0


def test_matches_ghia_at_re100(cavity_run):
    u = solve.centreline_u(cavity_run["psi"], cavity_run["h"])
    assert solve.ghia_deviation(cavity_run["y"], u) < 0.02  # |u| ≤ 1, Ghia min is -0.21
    psi = cavity_run["psi"]
    i, j = divmod(int(psi.argmin()), psi.shape[0])
    assert abs(cavity_run["y"][i] - 0.617) < 0.05 and abs(cavity_run["y"][j] - 0.734) < 0.05  # primary vortex centre
    assert psi.min().item() == pytest.approx(-0.1034, rel=0.05)
