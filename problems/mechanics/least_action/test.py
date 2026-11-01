"""Physics checks, written first (TDD): what any correct least-action code must satisfy.

Run with `pytest problems/mechanics/least_action/test.py`. Minimising the discrete action recovers the exact parabola
and sine paths, its gradient is the Euler-Lagrange residual, and the true path is a minimum only for short times (a
saddle beyond half a period).
"""

import pytest
import torch

from . import equation, solve, tensors


def _setup(system: str, T: float, N: int = 30, omega: float = 2.0, q_start: float = 0.0, q_end: float = 1.0):
    m, g = 1.0, 9.81
    V, dV = equation.gravity(m, g) if system == "gravity" else equation.oscillator(m, omega)
    t = tensors.time_grid(T, N)
    D = equation.difference_matrix(N)
    exact = equation.exact_path(t, q_start, q_end, system, g, omega)
    q0 = tensors.trial_paths(t, q_start, q_end, 4, 1.0)
    return {"m": m, "V": V, "dV": dV, "t": t, "D": D, "dt": T / N, "exact": exact, "q0": q0}


def test_difference_matrix():
    D = equation.difference_matrix(4)
    q = torch.tensor([[0.0, 1.0, 4.0, 9.0, 16.0]], dtype=torch.float64)
    assert torch.equal(torch.einsum("nk,bk->bn", D, q), torch.tensor([[1.0, 3.0, 5.0, 7.0]], dtype=torch.float64))


def test_trial_paths_keep_end_points():
    s = _setup("gravity", 1.0)
    q0 = s["q0"]
    assert torch.allclose(q0[:, 0], torch.zeros(4, dtype=q0.dtype), atol=1e-12)
    assert torch.allclose(q0[:, -1], torch.ones(4, dtype=q0.dtype), atol=1e-12)


def test_gradient_descent_finds_parabola():
    s = _setup("gravity", 1.2)
    hist = solve.relax(s["q0"], s["D"], s["dt"], s["m"], s["V"], 400, "gradient")
    # constant force: the Verlet recurrence is exact, so the discrete optimum *is* the sampled parabola
    assert (hist["paths"][-1] - s["exact"]).abs().max() < 1e-6
    S_exact = equation.action(s["exact"][None], s["D"], s["dt"], s["m"], s["V"])
    assert torch.allclose(hist["S"][-1], S_exact.expand(4), atol=1e-9)


@pytest.mark.parametrize("N", [20, 40])
def test_newton_finds_oscillator_sine_to_discretisation_accuracy(N):
    s = _setup("oscillator", 1.2, N=N)
    hist = solve.relax(s["q0"], s["D"], s["dt"], s["m"], s["V"], 150, "newton")
    err = (hist["paths"][-1] - s["exact"]).abs().max().item()
    assert err < 0.5 * (2.0 * s["dt"]) ** 2  # O(Δt²): halves the step, quarters the error
    if N == 40:
        coarse = _setup("oscillator", 1.2, N=20)
        h2 = solve.relax(coarse["q0"], coarse["D"], coarse["dt"], 1.0, coarse["V"], 150, "newton")
        err20 = (h2["paths"][-1] - coarse["exact"]).abs().max().item()
        assert 3.0 < err20 / err < 5.0


def test_gradient_equals_minus_dt_times_euler_lagrange_residual():
    s = _setup("oscillator", 1.0)
    q = s["q0"].clone()
    x = q[:, 1:-1].clone().requires_grad_(True)
    S = equation.action(solve.with_ends(x, q[:, :1], q[:, -1:]), s["D"], s["dt"], s["m"], s["V"])
    (dS,) = torch.autograd.grad(S.sum(), x)
    resid = equation.euler_lagrange_residual(q, s["dt"], s["m"], s["dV"])
    assert torch.allclose(dS, -s["dt"] * resid, atol=1e-12)


def test_gradient_vanishes_at_the_optimum():
    s = _setup("oscillator", 1.2)
    hist = solve.relax(s["q0"], s["D"], s["dt"], s["m"], s["V"], 600, "gradient")
    assert hist["grad"][-1].abs().max() < 1e-7
    resid = equation.euler_lagrange_residual(hist["paths"][-1], s["dt"], s["m"], s["dV"])
    assert resid.abs().max() < 1e-5


def test_optimum_has_least_action_for_short_T():
    s = _setup("oscillator", 1.0)
    hist = solve.relax(s["q0"][:1], s["D"], s["dt"], s["m"], s["V"], 150, "newton")
    best = hist["paths"][-1]  # (1, N+1)
    S_best = equation.action(best, s["D"], s["dt"], s["m"], s["V"])
    torch.manual_seed(1)
    bump = torch.randn(64, best.shape[1], dtype=best.dtype) * 0.05
    bump[:, 0] = bump[:, -1] = 0.0  # keep the end points fixed
    S_pert = equation.action(best + bump, s["D"], s["dt"], s["m"], s["V"])
    assert (S_pert > S_best).all()


def test_hessian_tridiagonal_and_positive_definite_for_short_T():
    s = _setup("oscillator", 1.0)
    H = solve.hessian(s["q0"], s["D"], s["dt"], s["m"], s["V"])  # (B, N−1, N−1)
    n = H.shape[-1]
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    assert H[:, (i - j).abs() > 1].abs().max() == 0
    m, dt, omega = s["m"], s["dt"], 2.0
    assert torch.allclose(
        torch.diagonal(H, dim1=1, dim2=2), torch.full((4, n), 2 * m / dt - m * omega**2 * dt, dtype=H.dtype)
    )
    assert torch.allclose(torch.diagonal(H, 1, dim1=1, dim2=2), torch.full((4, n - 1), -m / dt, dtype=H.dtype))
    assert torch.linalg.eigvalsh(H).min() > 0


def test_long_T_true_path_is_a_saddle():
    s = _setup("oscillator", 2.5)  # longer than half a period π/ω ≈ 1.57 s
    H = solve.hessian(s["exact"][None], s["D"], s["dt"], s["m"], s["V"])[0]
    eig = torch.linalg.eigvalsh(H)
    assert eig.min() < 0 < eig.max()
    hist = solve.relax(s["q0"], s["D"], s["dt"], s["m"], s["V"], 150, "newton")
    assert hist["grad"][-1].abs().max() < 1e-8  # Newton still finds the stationary path
    assert (hist["paths"][-1] - s["exact"]).abs().max() < 0.02
