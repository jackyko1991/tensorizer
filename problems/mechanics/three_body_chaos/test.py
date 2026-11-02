"""Physics checks, written first (TDD): what any correct three-body code must satisfy.

Run with `pytest problems/mechanics/three_body_chaos/test.py`. Energy, momentum and the centre of mass are conserved
through close encounters, pairwise forces obey Newton's third law, and a 10⁻⁶ offset grows exponentially (positive
Lyapunov exponent).
"""


import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_pythagorean_setup():
    m, x, v = tensors.initial_state(0.0, CPU)
    d = equation.pair_distances(x)[0]  # (3, 3)
    # each mass sits opposite the side of the same length: |x_4 - x_5| = 3, |x_3 - x_5| = 4, |x_3 - x_4| = 5
    assert torch.allclose(torch.stack([d[1, 2], d[0, 2], d[0, 1]]), m)
    assert torch.allclose(equation.center_of_mass(x[0], m), torch.zeros(2, dtype=torch.float64))
    assert (v == 0).all()


def test_einsum_force_matches_pairwise_loop():
    m, x, _ = tensors.initial_state(1e-3, CPU)
    a = equation.acceleration(x, m, G=1.0)  # (K, 3, 2)
    for k in range(2):
        for i in range(3):
            expected = sum(m[j] * (x[k, j] - x[k, i]) / (x[k, j] - x[k, i]).norm() ** 3 for j in range(3) if j != i)
            assert torch.allclose(a[k, i], expected)
    assert torch.allclose(equation.momentum(a, m), torch.zeros(2, 2, dtype=torch.float64), atol=1e-12)  # Newton III


@pytest.fixture(scope="module")
def runs():
    """Run 0 = reference, run 1 = twin moved by δ = 1e-6, integrated together to t = 50."""
    m, x0, v0 = tensors.initial_state(1e-6, CPU)
    times, xs, vs, _ = solve.solve(m, x0, v0, 1.0, t_max=50.0, tol=1e-10)
    return m, times, xs, vs


def test_conserves_energy_momentum_and_centre_of_mass(runs):
    m, _, xs, vs = runs
    e_err = solve.energy_error(xs, vs, m, 1.0)[:, 0]
    assert e_err.max() < 1e-6  # through close encounters at distance ~0.01
    assert equation.momentum(vs[:, 0], m).abs().max() < 1e-10
    assert equation.center_of_mass(xs[:, 0], m).abs().max() < 1e-10


def test_is_chaotic(runs):
    m, times, xs, vs = runs
    dist = equation.phase_space_distance(xs, vs)  # reference vs δ = 1e-6 twin
    assert dist[0] == pytest.approx(1e-6)
    assert dist[-10:].min() > 0.1  # grew by more than five orders of magnitude
    lam, *_ = solve.lyapunov_fit(times, dist)
    assert lam > 0.05


def test_twin_with_zero_offset_stays_identical():
    m, x0, v0 = tensors.initial_state(0.0, CPU)
    _, xs, vs, _ = solve.solve(m, x0, v0, 1.0, t_max=20.0, tol=1e-10)
    # Rows of a batch can take different SIMD paths (CPUs and BLAS builds differ), so the twins may differ in the last
    # bits; chaos grows that round-off (to ~3e-6 on some CPUs), but it stays orders of magnitude below where the real
    # δ = 1e-6 twin ends up (> 0.1): only an actual offset makes the trajectories separate.
    assert (xs[:, 1] - xs[:, 0]).abs().max() < 1e-4
    assert equation.phase_space_distance(xs, vs).max() < 1e-4


def test_lyapunov_fit_recovers_known_rate():
    t = torch.linspace(0, 20, 401, dtype=torch.float64)
    dist = 1e-6 * torch.exp(0.4 * t)  # pure exponential growth, saturating above 0.1 at t ≈ 28.8 → never
    dist = torch.cat([dist, torch.full((50,), 1.0, dtype=torch.float64)])
    t = torch.cat([t, 20 + 0.05 * torch.arange(1, 51, dtype=torch.float64)])
    lam, *_ = solve.lyapunov_fit(t, dist)
    assert lam == pytest.approx(0.4, rel=0.15)
