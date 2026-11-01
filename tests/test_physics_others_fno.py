"""Physics and tensor checks for problems/others/neural_operator (Fourier neural operator for Burgers)."""

import math

import pytest
import torch

from problems.fluids.burgers_shock.solve import cole_hopf
from problems.others.neural_operator import equation as eq
from problems.others.neural_operator import solve
from problems.others.neural_operator import tensors as tz

CPU = torch.device("cpu")


def _rel(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


# --- spectral convolution -------------------------------------------------------------------------------


def test_spectral_conv_einsum_equals_loop_over_modes():
    gen = torch.Generator().manual_seed(1)
    B, C, N, m = 3, 4, 32, 6
    v = torch.randn(B, C, N, generator=gen, dtype=torch.float64)
    R = torch.complex(torch.randn(C, C, m, generator=gen), torch.randn(C, C, m, generator=gen)).to(torch.complex128)

    v_hat = torch.fft.rfft(v)
    out_hat = torch.zeros_like(v_hat)
    for b in range(B):
        for k in range(m):
            for o in range(C):
                out_hat[b, o, k] = sum(R[i, o, k] * v_hat[b, i, k] for i in range(C))  # explicit Σ_i
    expected = torch.fft.irfft(out_hat, n=N)
    assert torch.allclose(eq.spectral_conv(v, R), expected, atol=1e-12)


def test_truncation_zeroes_modes_at_and_above_m():
    gen = torch.Generator().manual_seed(2)
    C, N, m = 3, 64, 5
    v = torch.randn(2, C, N, generator=gen, dtype=torch.float64)
    R = torch.randn(C, C, m, generator=gen, dtype=torch.float64).to(torch.complex128)
    out_hat = torch.fft.rfft(eq.spectral_conv(v, R))
    assert out_hat[..., m:].abs().max() < 1e-10
    assert out_hat[..., :m].abs().max() > 1e-3


def test_spectral_conv_is_exactly_resolution_invariant_for_band_limited_input():
    """K v depends only on the Fourier modes of v, not on the grid: the same R on 64 and 128 points agree."""
    gen = torch.Generator().manual_seed(3)
    C, m = 3, 8
    R = torch.randn(C, C, m, generator=gen, dtype=torch.float64).to(torch.complex128)
    coeffs = torch.randn(C, 6, 2, generator=gen, dtype=torch.float64)  # modes k = 1 … 6 < 32
    outs = []
    for n in (64, 128):
        x, _ = tz.grid(n, CPU)
        v = tz.initial_conditions(coeffs, x)[None]  # (1, C, n): C band-limited channels
        outs.append(eq.spectral_conv(v, R)[0])
    assert torch.allclose(outs[1][:, ::2], outs[0], atol=1e-12)


def test_fno_runs_on_any_grid_and_agrees_for_band_limited_input():
    """The whole (untrained) FNO on 64 and 128 points: equal up to the small aliasing of the GELU harmonics."""
    w = tz.init_fno(width=8, n_modes=8, n_layers=3, seed=0, device=CPU)
    coeffs = tz.random_modes(4, 3, seed=5, device=CPU)
    outs = []
    with torch.no_grad():
        for n in (64, 128):
            x, _ = tz.grid(n, CPU)
            u0 = tz.initial_conditions(coeffs, x).float()
            out = eq.fno(u0, x.float(), w)
            assert out.shape == (4, n)
            outs.append(out)
    assert _rel(outs[1][:, ::2], outs[0]) < 1e-2


# --- the data solver ------------------------------------------------------------------------------------


def test_solver_conserves_mass():
    x, dx = tz.grid(128, CPU)
    u0 = tz.initial_conditions(tz.random_modes(5, 5, seed=0, device=CPU), x) + 0.3  # non-zero mean
    uT = solve.solve_burgers(u0, nu=0.05, t_final=1.0, dx=dx)
    assert torch.allclose(eq.mass(uT, dx), eq.mass(u0, dx), atol=1e-10)


def test_small_sine_decays_at_rate_nu_k_squared():
    """At tiny amplitude the advection term is negligible: u = ε e^{-νk²T} sin kx."""
    x, dx = tz.grid(64, CPU)
    nu, T, eps = 0.1, 1.0, 1e-4
    for k in (1, 3):
        u0 = eps * torch.sin(k * x)[None]
        uT = solve.solve_burgers(u0, nu, T, dx)
        assert _rel(uT, u0 * math.exp(-nu * k**2 * T)) < 1e-3


def test_solver_matches_cole_hopf_reference():
    """u_0 = sin x: the exact Cole–Hopf solution from problems/fluids/burgers_shock."""
    x, dx = tz.grid(256, CPU)
    nu, T = 0.1, 1.0
    uT = solve.solve_burgers(torch.sin(x)[None], nu, T, dx)[0]
    assert _rel(uT, cole_hopf(x, T, nu, amplitude=1.0, offset=0.0)) < 1e-4


# --- training -------------------------------------------------------------------------------------------


def _small_problem(seed=0):
    x_f, dx_f = tz.grid(128, CPU)
    coeffs = tz.random_modes(160, 4, seed=seed, device=CPU)
    u0_f = tz.initial_conditions(coeffs, x_f)
    uT_f = solve.solve_burgers(u0_f, 0.1, 1.0, dx_f)
    s = 4  # 32-point training grid
    return {
        "x": x_f[::s].float(),
        "u0_train": u0_f[:120, ::s].float(),
        "uT_train": uT_f[:120, ::s].float(),
        "u0_test": u0_f[120:, ::s].float(),
        "uT_test": uT_f[120:, ::s].float(),
    }


def _train_small(data, epochs=15):
    w = tz.init_fno(width=12, n_modes=8, n_layers=2, seed=0, device=CPU)
    return w, solve.train(w, data, epochs=epochs, lr=5e-3, batch=20, seed=0)


@pytest.fixture(scope="module")
def trained():
    data = _small_problem()
    return data, *_train_small(data)


def test_trained_fno_beats_identity_baseline(trained):
    data, w, hist = trained
    baseline = eq.relative_l2(data["u0_test"], data["uT_test"]).item()
    assert hist["test_err"][-1] < 0.3 * baseline
    assert hist["test_err"][-1] < hist["test_err"][0]


def test_training_is_reproducible_with_the_seed(trained):
    data, _, hist = trained
    _, again = _train_small(data)
    assert torch.equal(again["train_err"], hist["train_err"])
    assert torch.equal(again["pred_test"][-1], hist["pred_test"][-1])
