"""Physics checks, written first (TDD): what any correct shell-model code must satisfy.

Run with `pytest problems/nuclear/shell_model/test.py`. Each test is a physical law, not an implementation detail:
without spin–orbit the oscillator magic numbers appear, with it the real magic numbers 2…126, and each level holds
2j + 1 nucleons.
"""

import torch

from . import solve, tensors

CPU = torch.device("cpu")


def shell_scheme(v_so: float, mass_number: int = 208, v0: float = 44.0):
    r, h = tensors.radial_grid(300, 18.0, CPU)
    d2 = tensors.second_difference(len(r), h, CPU)
    ell, j = tensors.channels(8, CPU)
    radius = 1.27 * mass_number ** (1 / 3)
    energies = solve.bound_levels(r, d2, ell, j, v0, v_so, radius, 0.67, 1.27)
    return solve.level_scheme(energies, ell, j), ell, j


def _shell_inputs():
    r, h = tensors.radial_grid(300, 18.0, CPU)
    ell, j = tensors.channels(8, CPU)
    return r, tensors.second_difference(len(r), h, CPU), ell, j


def test_without_spin_orbit_gives_oscillator_magic_numbers():
    scheme, ell, j = shell_scheme(0.0)
    _, magic = solve.largest_gaps(scheme, 5)
    assert {2, 8, 20, 40} <= set(int(x) for x in magic.tolist())  # harmonic-oscillator shell closures
    # without spin–orbit the j = l ± ½ partners are degenerate
    e = solve.bound_levels(*_shell_inputs(), 44.0, 0.0, 1.27 * 208 ** (1 / 3), 0.67, 1.27)
    assert torch.allclose(e[1], e[2], equal_nan=True)  # p3/2 and p1/2


def test_spin_orbit_gives_real_magic_numbers():
    scheme, ell, j = shell_scheme(22.0)
    _, magic = solve.largest_gaps(scheme, 7)
    assert set(int(x) for x in magic.tolist()) == {2, 8, 20, 28, 50, 82, 126}  # Mayer–Jensen magic numbers


def test_levels_sorted_with_2j_plus_1_degeneracy():
    scheme, ell, j = shell_scheme(22.0)
    assert torch.all(torch.diff(scheme["energy"]) >= 0)
    occupancy = torch.diff(scheme["filled"], prepend=torch.zeros(1, dtype=scheme["filled"].dtype))
    assert torch.equal(occupancy, 2 * j[scheme["channel"]] + 1)  # Pauli: each level holds 2j + 1 nucleons
    first = scheme["channel"][0]
    assert ell[first] == 0 and j[first] == 0.5  # 1s1/2 is the lowest level
    k = {(int(a), float(b)): i for i, (a, b) in enumerate(zip(ell.tolist(), j.tolist()))}
    e = dict(zip(zip(scheme["channel"].tolist(), scheme["radial"].tolist()), scheme["energy"].tolist()))
    assert e[(k[(4, 4.5)], 0)] < e[(k[(4, 3.5)], 0)]  # 1g9/2 below 1g7/2: spin–orbit favours j = l + ½
