"""Physics checks, written first (TDD): what any correct Ising-model code must satisfy.

Run with `pytest problems/thermodynamics/ising_model/test.py`. The lattice orders below the critical temperature and
disorders above it, and the convolution energy equals the explicit nearest-neighbour bond sum (−2 per spin when
fully aligned).
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def run_ising(temps, L=16, n_sweeps=300, seed=0):
    T = torch.tensor(temps)
    s0 = tensors.random_spins(len(T), L, CPU, seed=seed)
    return solve.solve(s0, T, tensors.checkerboard(L, CPU), n_sweeps)


def test_orders_when_cold_and_disorders_when_hot():
    r = run_ising([0.5, 1.0, 10.0])
    m = r["m"][-50:].abs().mean(dim=0)  # (B,)
    # a cold quench can be trapped in two stripe domains, so ask that one of the cold lattices orders fully
    assert max(m[0].item(), m[1].item()) > 0.95  # T < T_c ≈ 2.27: spontaneous magnetisation
    assert m[2] < 0.15  # T > T_c: paramagnet
    assert r["E"][-1, 2] > -0.5  # hot: hardly more aligned bonds than anti-aligned ones


def test_energy_conv_matches_explicit_bond_sum():
    s = tensors.random_spins(3, 12, CPU, seed=5)
    # every bond counted once: right and down neighbours, wrapping around with torch.roll
    right = torch.roll(s, shifts=-1, dims=-1)
    down = torch.roll(s, shifts=-1, dims=-2)
    explicit = -(s * right + s * down).sum(dim=(1, 2, 3)) / 144
    assert torch.allclose(equation.energy_per_spin(s), explicit)

    ordered = torch.ones(1, 1, 8, 8)
    assert equation.energy_per_spin(ordered).item() == -2.0  # ground state: E/N = −2J on the square lattice
    assert torch.equal(equation.neighbor_sum(ordered), 4 * ordered)
