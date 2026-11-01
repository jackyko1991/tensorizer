"""Physics checks, written first (TDD): what any correct binding-energy code must satisfy.

Run with `pytest problems/nuclear/binding_energy/test.py`. Each test is a physical law, not an implementation
detail: B/A peaks near iron at about 8.7 MeV, the most stable Z follows the valley of stability and the liquid drop
reproduces measured binding energies of medium and heavy nuclei to 1 %.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")

COEFF = {"a_V": 15.8, "a_S": 18.3, "a_C": 0.714, "a_A": 23.2, "a_P": 12.0}


def binding_table():
    z, n = tensors.grid(110, 176, CPU)
    return z, n, equation.binding_energy(z, n, COEFF)


def test_binding_per_nucleon_peaks_near_iron():
    z, n, b = binding_table()
    mass, _, best = solve.most_stable(b, z, n)
    a_peak, ba_peak = solve.peak(mass, best)
    assert 56 <= a_peak <= 62  # iron peak: the most tightly bound nuclei
    assert 8.6 < ba_peak < 8.9


def test_valley_of_stability():
    z, n, b = binding_table()
    mass, z_best, _ = solve.most_stable(b, z, n)
    odd = (mass % 2 == 1) & (mass > 20) & (mass < 240)  # odd A: no pairing term to make Z jump
    formula = mass / (2 + 0.0154 * mass ** (2 / 3))
    assert torch.all((z_best[odd] - formula[odd]).abs() <= 1.0)  # valley of stability Z ≈ A / (2 + 0.0154 A^⅔)


def test_energies_of_known_nuclei():
    _, _, b = binding_table()
    for name, zz, a, ba in tensors.MEASURED:
        if a >= 40:  # the liquid drop is good to ~1 % for medium and heavy nuclei
            assert math.isclose(b[zz - 1, a - zz].item(), ba * a, rel_tol=0.01), name
    he4 = b[1, 2].item()
    assert 0.6 * 28.3 < he4 < 28.3  # ⁴He is famously under-bound by the liquid drop
