"""Physics checks, written first (TDD): what any correct free wave packet code must satisfy.

Run with `pytest problems/quantum/free_wave_packet/test.py`. Each test is a physical law, not an implementation
detail: the centre moves at the group velocity, the width follows the exact Gaussian spreading law and the norm is
conserved.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_centre_width_and_norm_match_exact_solution():
    # σ = standard deviation of |ψ|²; exact: ⟨x⟩ = x0 + k0 t, σ(t) = σ0 √(1 + (t/(2σ0²))²)
    x, k, dx = tensors.grid(4096, 300.0, CPU)
    x0, sigma0, k0 = -40.0, 1.5, 2.0
    times, psi = solve.solve(tensors.gaussian_packet(x, x0, sigma0, k0, dx), k, t_max=30.0, n_steps=30)
    mean, width = equation.mean_and_width(psi, x, dx)
    # Ehrenfest: ⟨x⟩ moves at the group velocity k0
    assert torch.allclose(mean, solve.exact_centre(times, x0, k0), atol=1e-6)
    assert torch.allclose(width, solve.exact_width(times, sigma0), rtol=1e-6)  # exact Gaussian spreading
    # unitary evolution: probability conserved
    assert torch.allclose(equation.norm(psi, dx), torch.ones(len(times), dtype=torch.float64), atol=1e-12)
