"""Physics checks, written first (TDD): what any correct Langevin-equation code must satisfy.

Run with `pytest problems/others/langevin_noise/test.py`. Each test is a physical law, not an implementation
detail: equipartition ⟨½kx²⟩ = ⟨½mv²⟩ = ½k_BT with Gaussian positions, the damped oscillator at T = 0, the
fluctuation–dissipation noise amplitude ∝ √Δt, and the overdamped relaxation of ⟨x²⟩.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _langevin(T=1.0, gamma=0.5, k=1.0, m=1.0, B=2000, t_max=40.0, dt=None, seed=0, x0=None):
    x, v = tensors.initial_state(B, CPU)
    if x0 is not None:
        x = torch.full_like(x, x0)
    dt = dt or tensors.time_step(gamma, k, m)
    return solve.solve(x, v, gamma, k, m, T, t_max, dt, tensors.generator(CPU, seed), n_saved=400)


def test_equipartition():
    T, k, m = 1.5, 2.0, 0.5
    times, x, v = _langevin(T=T, gamma=1.0, k=k, m=m, B=4000, t_max=30.0, dt=0.002)
    late = times > 10.0
    x2 = equation.ensemble_mean_square(x)[late].mean().item()
    v2 = equation.ensemble_mean_square(v)[late].mean().item()
    x2_eq, v2_eq = equation.equipartition(T, k, m)
    assert abs(x2 / x2_eq - 1) < 0.04, (x2, x2_eq)  # equipartition: k⟨x²⟩ = k_B T
    assert abs(v2 / v2_eq - 1) < 0.04, (v2, v2_eq)  # equipartition: m⟨v²⟩ = k_B T
    # the final positions are Gaussian with variance k_B T / k: kurtosis ⟨x⁴⟩/⟨x²⟩² ≈ 3
    xe = x[-1]
    assert abs((xe**4).mean() / (xe**2).mean() ** 2 - 3) < 0.25  # Boltzmann distribution e^{-kx²/2k_BT}


def test_zero_temperature_is_damped_oscillator():
    gamma, k, m = 0.5, 1.0, 1.0
    times, x, v = _langevin(T=0.0, gamma=gamma, k=k, m=m, B=3, t_max=20.0, dt=0.001)
    exact = equation.damped_oscillator(times, tensors.X0, gamma, k, m)
    assert torch.allclose(x[:, 0], x[:, 2])  # no noise: every particle follows the same path
    assert (x[:, 0] - exact).abs().max() < 5e-3  # T → 0 limit: the analytic damped oscillator
    # the overdamped branch of the analytic formula solves the ODE too (finite differences)
    t = torch.linspace(0, 5, 5001, dtype=torch.float64)
    xo = equation.damped_oscillator(t, 1.0, 5.0, 1.0, 1.0)
    h = (t[1] - t[0]).item()
    acc = (xo[2:] - 2 * xo[1:-1] + xo[:-2]) / h**2
    vel = (xo[2:] - xo[:-2]) / (2 * h)
    assert (acc + 5.0 * vel + xo[1:-1]).abs().max() < 1e-5  # m ẍ + γ ẋ + k x = 0


def test_noise_increment_scales_with_sqrt_dt():
    gamma, T, m = 0.7, 2.0, 1.3
    for dt in (1e-2, 1e-4):
        amp = equation.noise_amplitude(gamma, T, m, dt)
        assert math.isclose(amp**2, 2 * gamma * T * dt / m**2, rel_tol=1e-12)  # fluctuation–dissipation theorem
        # one step from rest at x = 0: the velocity change is pure noise with variance 2γk_BTΔt/m²
        x = torch.zeros(200_000, dtype=torch.float64)
        xi = torch.randn(x.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        _, dv = equation.euler_maruyama_step(x, x.clone(), xi, gamma, 1.0, m, T, dt)
        assert abs(dv.var().item() / (2 * gamma * T * dt / m**2) - 1) < 0.02
    # halving Δt shrinks the kick by √2, not by 2
    ratio = equation.noise_amplitude(gamma, T, m, 0.01) / equation.noise_amplitude(gamma, T, m, 0.005)
    assert math.isclose(ratio, math.sqrt(2), rel_tol=1e-12)  # Wiener process: ⟨ΔW²⟩ = Δt


def test_overdamped_variance_relaxation():
    T, gamma, k, m = 1.0, 5.0, 1.0, 0.05  # m/γ = 0.01 ≪ γ/k = 5: overdamped Brownian motion
    times, x, _ = _langevin(T=T, gamma=gamma, k=k, m=m, B=4000, t_max=6.0, dt=0.001, x0=0.0)
    x2 = equation.ensemble_mean_square(x)
    expected = T / k * (1 - torch.exp(-2 * k / gamma * times))  # Ornstein–Uhlenbeck: ⟨x²⟩ = (k_BT/k)(1 - e^{-2kt/γ})
    assert (x2 - expected)[times > 0.5].abs().max() < 0.08
    # rate from the half-way point: e^{-2kt/γ} = 1/2 at t = γ ln2 / 2k
    i = int((x2 > 0.5 * T / k).int().argmax())
    rate = math.log(2) / times[i].item()
    assert abs(rate / (2 * k / gamma) - 1) < 0.15


def test_reproducible_with_seed():
    _, xa, _ = _langevin(B=50, t_max=5.0, seed=3)
    _, xb, _ = _langevin(B=50, t_max=5.0, seed=3)
    _, xc, _ = _langevin(B=50, t_max=5.0, seed=4)
    assert torch.equal(xa, xb)
    assert not torch.equal(xa, xc)
