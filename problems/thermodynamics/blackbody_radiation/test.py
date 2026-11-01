"""Physics checks, written first (TDD): what any correct blackbody-radiation code must satisfy.

Run with `pytest problems/thermodynamics/blackbody_radiation/test.py`. The Planck peak obeys Wien's displacement law
λ_max T = b and the total energy density obeys the Stefan–Boltzmann law U = 4σT⁴/c.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_wien_peak_and_stefan_boltzmann():
    T = tensors.temperatures(3000.0, 10000.0, 8, CPU)
    lam = tensors.wavelengths(CPU)
    r = solve.solve(T, lam)
    assert r["u"].shape == (8, len(lam))

    wien = r["lam_max"] * T  # Wien's displacement law: λ_max T = b for every temperature
    assert torch.allclose(wien, torch.full_like(wien, 2.898e-3), rtol=5e-3)

    U = r["U"]
    assert torch.allclose(U / U[0], (T / T[0]) ** 4, rtol=1e-4)  # ratio test: U ∝ T⁴
    assert torch.allclose(U, equation.stefan_boltzmann(T), rtol=1e-4)  # and the prefactor 4σ/c
