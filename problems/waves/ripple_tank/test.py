"""Physics checks, written first (TDD): what any correct ripple-tank code must satisfy.

Run with `pytest problems/waves/ripple_tank/test.py`. The CFL stability limit is enforced, and two-source interference
puts nodal lines at path differences (m + ½) λ.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_cfl_enforced():
    assert equation.time_step(1.0, 1.0, 0.7) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        equation.time_step(1.0, 1.0, 0.75)


def test_nodal_lines_at_half_odd_path_difference():
    lam = 16.0
    X, Y = tensors.grid(CPU)
    gamma = tensors.damping(CPU)
    blob, s1, s2 = tensors.sources(X, Y, 3 * lam)
    dt = equation.time_step(tensors.H, tensors.C, 0.5)
    _, _, intensity = solve.solve(blob, gamma, tensors.stencil(CPU), lam, tensors.C, tensors.H, dt, 300.0, n_frames=2)
    D = equation.path_difference(X, Y, s1, s2)
    far = (gamma == 0) & (Y > tensors.SOURCE_Y + 3 * lam)
    I_node, I_anti = solve.nodal_contrast(intensity[-1], D, lam, far)
    assert I_node < 0.2 * I_anti  # destructive interference on the nodal lines
    # a cut along a row far from the sources: intensity minima sit near |r1 - r2| = (m + 1/2) λ
    row = int(tensors.SOURCE_Y + 6 * lam)
    I_row, D_row = intensity[-1][:, row], D[:, row]
    a = torch.arange(tensors.BORDER + 1, tensors.N - tensors.BORDER - 1)
    minima = a[(I_row[a] < I_row[a - 1]) & (I_row[a] < I_row[a + 1])]
    z = D_row[minima] / lam - 0.5
    assert len(minima) >= 4
    assert (z - torch.round(z)).abs().max() < 0.15  # minima at |r₁ − r₂| = (m + ½) λ
