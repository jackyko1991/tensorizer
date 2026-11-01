"""Physics checks for the biophysics kinetics problems: Michaelis–Menten, FRAP diffusion, Hodgkin–Huxley."""

import math

import torch

from problems.biophysics.hodgkin_huxley import equation as hh_eq
from problems.biophysics.hodgkin_huxley import solve as hh_solve
from problems.biophysics.hodgkin_huxley import tensors as hh_t
from problems.biophysics.membrane_diffusion_frap import equation as fr_eq
from problems.biophysics.membrane_diffusion_frap import solve as fr_solve
from problems.biophysics.membrane_diffusion_frap import tensors as fr_t
from problems.biophysics.michaelis_menten import equation as mm_eq
from problems.biophysics.michaelis_menten import solve as mm_solve
from problems.biophysics.michaelis_menten import tensors as mm_t

CPU = torch.device("cpu")
F64 = torch.float64


# ---------------------------------------------------------------- Michaelis–Menten


def _mm_run(E0=0.1, t_end=30.0):
    k1, k_1, k2 = (k.to(F64) for k in mm_t.rate_constants(CPU))
    S0 = mm_t.substrate_concentrations(CPU).to(F64)
    y0 = mm_t.initial_state(E0, S0, len(k1))
    K_m, V_max = mm_eq.michaelis_constant(k1, k_1, k2), mm_eq.max_velocity(k2, E0)
    times, full, S_q = mm_solve.solve(y0, k1, k_1, k2, V_max, K_m, t_end)
    return times, full, S_q, S0, k1, k_1, k2, K_m, V_max


def test_mm_conserves_enzyme_and_substrate():
    E0 = 0.5
    _, full, _, S0, *_ = _mm_run(E0=E0)
    E, S, ES, P = full.unbind(1)
    assert (E + ES - E0).abs().max() < 1e-10
    assert (S + ES + P - S0).abs().max() < 1e-9
    assert full.min() > -1e-12  # concentrations stay non-negative
    assert (P[1:] - P[:-1]).min() >= 0  # product only accumulates


def test_mm_rate_law_limits():
    V_max, K_m = torch.tensor([2.0], dtype=F64), torch.tensor([5.0], dtype=F64)
    v = mm_eq.mm_rate(torch.tensor([5.0, 1e9, 1e-6], dtype=F64), V_max, K_m)[0]
    assert math.isclose(v[0].item(), 1.0, rel_tol=1e-12)  # half speed at S = K_m
    assert math.isclose(v[1].item(), 2.0, rel_tol=1e-6)  # saturation at V_max
    assert math.isclose(v[2].item(), 2.0 / 5.0 * 1e-6, rel_tol=1e-6)  # linear regime v ≈ (V_max/K_m) S


def test_mm_reduced_model_matches_integrated_law():
    times, _, S_q, S0, *_, K_m, V_max = _mm_run(t_end=30.0)
    for i in (100, len(times) // 2, len(times) - 1):
        t_exact = mm_eq.integrated_mm_time(S_q[i], S0, V_max, K_m)  # (B, C)
        assert (t_exact - times[i]).abs().max() < 1e-6


def test_mm_qssa_valid_for_small_enzyme_and_fails_for_large():
    def gap(E0):
        _, full, S_q, S0, *_ = _mm_run(E0=E0, t_end=20.0)
        return ((full[:, 3] - (S0 - S_q)).abs() / S0).max().item()

    small, large = gap(0.01), gap(2.0)
    assert small < 2e-3
    assert large > 10 * small


def test_mm_lineweaver_burk_recovers_constants():
    # exact Michaelis–Menten data: the fit is exact
    S = torch.tensor([[1.0, 2.0, 5.0, 10.0, 50.0]], dtype=F64)
    v = mm_eq.mm_rate(S, torch.tensor([3.0], dtype=F64), torch.tensor([7.0], dtype=F64))
    K_fit, V_fit = mm_eq.lineweaver_burk(S, v)
    assert math.isclose(K_fit.item(), 7.0, rel_tol=1e-10) and math.isclose(V_fit.item(), 3.0, rel_tol=1e-10)

    # initial rates measured on the full mass-action simulation
    _, full, _, _, _, _, k2, K_m, V_max = _mm_run(t_end=10.0)
    S_v, v0 = mm_solve.initial_rate(full, k2)
    K_fit, V_fit = mm_eq.lineweaver_burk(S_v, v0)
    assert torch.allclose(K_fit, K_m, rtol=1e-2)
    assert torch.allclose(V_fit, V_max, rtol=1e-2)


# ---------------------------------------------------------------- FRAP


def test_frap_laplacian_stencil():
    x, h = fr_t.grid(1.0, CPU)
    x = x.to(F64)
    w = fr_t.stencil(CPU).to(F64)
    assert math.isclose(w.sum().item(), 0.0)
    const = torch.full((1, len(x), len(x)), 3.0, dtype=F64)
    assert fr_eq.laplacian(const, w, h).abs().max() < 1e-12  # ∇² of a constant is zero, edges included
    quad = (x[:, None] ** 2 + x[None, :] ** 2)[None]  # ∇²(x² + y²) = 4
    lap = fr_eq.laplacian(quad, w, h)[0, 1:-1, 1:-1]
    assert torch.allclose(lap, torch.full_like(lap, 4.0), atol=1e-9)


def _frap_run(w=1.0, t_end=12.0, D=None):
    x, h = fr_t.grid(w, CPU)
    mask = fr_t.bleach_mask(x, w)
    D = fr_t.diffusion_coefficients(CPU)[:2] if D is None else D  # lipid and membrane protein
    c0 = fr_t.initial_state(mask, 0.2, len(D))
    times, F, _, snaps = fr_solve.solve(c0, D, fr_t.stencil(CPU), h, mask, t_end, 3)
    F_inf = c0.mean(dim=(1, 2))
    f = fr_eq.normalized_recovery(F, F[0], F_inf)
    return times, f, snaps, F_inf, D


def test_frap_conserves_dye_and_recovers():
    times, f, snaps, F_inf, _ = _frap_run()
    assert (snaps.mean(dim=(2, 3)) - F_inf).abs().max() < 1e-5  # no-flux edges: total fluorescence constant
    assert math.isclose(f[0, 0].item(), 0.0, abs_tol=1e-6)
    assert (f[1:] - f[:-1]).min() > -1e-5  # recovery never goes backwards
    assert f[-1, 0] > 0.95  # the lipid has nearly fully recovered


def test_frap_half_time_and_fit_match_soumpasis():
    w = 1.0
    times, f, _, _, D = _frap_run(w=w)
    t_half = fr_solve.half_time(times, f)
    assert torch.allclose(t_half, fr_eq.soumpasis_half_time(w, D), rtol=0.03)
    assert torch.allclose(fr_solve.fit_diffusion(times, f, w, t_half), D, rtol=0.05)
    # the formula itself: f(t½) = ½ and f → 1 at long times
    tau = fr_eq.diffusion_time(w, D)
    assert torch.allclose(fr_eq.soumpasis(fr_eq.soumpasis_half_time(w, D)[:1], tau[:1]), torch.tensor(0.5), atol=3e-3)
    assert fr_eq.soumpasis(torch.tensor([1e4]), tau).min() > 0.99


def test_frap_half_time_scales_as_w_squared():
    D = torch.tensor([1.0])
    t1 = fr_solve.half_time(*_frap_run(w=1.0, t_end=1.0, D=D)[:2])
    t2 = fr_solve.half_time(*_frap_run(w=2.0, t_end=4.0, D=D)[:2])
    assert math.isclose((t2 / t1).item(), 4.0, rel_tol=1e-3)  # doubling the spot: 4× slower, same D


# ---------------------------------------------------------------- Hodgkin–Huxley


def _hh_run(currents, temp=6.3, t_end=150.0):
    g, E = hh_t.channels(CPU)
    I_ext = torch.tensor(currents)
    y0 = hh_t.initial_state(len(I_ext), CPU)
    times, traj = hh_solve.solve(y0, I_ext, g, E, hh_t.C_M, hh_t.temperature_factor(temp), t_end)
    spikes = hh_solve.spike_times(times, traj[:, 0])
    return traj, spikes, hh_solve.firing_rate(spikes, t_end)


def test_hh_resting_state_is_a_fixed_point():
    g, E = (v.to(F64) for v in hh_t.channels(CPU))
    V = torch.tensor([-65.0], dtype=F64)
    y0 = torch.cat([V[None], hh_eq.steady_state(V)])
    m, h, n = y0[1:, 0].tolist()
    assert abs(m - 0.0529) < 1e-3 and abs(h - 0.5961) < 1e-3 and abs(n - 0.3177) < 1e-3
    dy = hh_eq.rhs(y0, torch.zeros(1, dtype=F64), g, E, hh_t.C_M, 1.0)
    # E_L = −54.387 mV (the classic rounded value) makes −65 mV the rest to within ~0.01 mV
    assert dy.abs().max() < 1e-2
    traj, spikes, _ = _hh_run([0.0], t_end=50.0)
    assert (traj[:, 0] + 65.0).abs().max() < 0.05 and len(spikes[0]) == 0


def test_hh_rate_functions_are_continuous_at_removable_singularities():
    for V0 in (-40.0, -55.0):
        V = torch.tensor([V0 - 1e-3, V0, V0 + 1e-3], dtype=F64)
        alpha, _ = hh_eq.rates(V)
        assert torch.isfinite(alpha).all()
        assert (alpha[:, 1] - 0.5 * (alpha[:, 0] + alpha[:, 2])).abs().max() < 1e-6
    assert math.isclose(hh_eq.rates(torch.tensor([-40.0], dtype=F64))[0][0].item(), 1.0)  # α_m(−40) = 1/ms


def test_hh_threshold_and_f_i_curve():
    currents = [0.0, 2.0, 4.0, 10.0, 15.0, 20.0]
    traj, spikes, rate = _hh_run(currents)
    gates = traj[:, 1:]
    assert gates.min() >= 0.0 and gates.max() <= 1.0
    n = [len(s) for s in spikes]
    assert n[0] == 0 and n[1] == 0  # below threshold: no spike
    assert n[2] == 1 and rate[2] == 0  # a single spike, then a new resting state
    assert 35.0 < traj[:, 0].max() < 50.0  # spike peak below E_Na = 50 mV
    assert 60.0 < rate[3] < 75.0  # classic ≈ 68 Hz at 10 µA/cm²
    assert (rate[3:][1:] > rate[3:][:-1]).all()  # rate rises with current


def test_hh_type_ii_onset_and_temperature():
    _, _, rate = _hh_run([6.0, 6.5, 7.0])
    first = rate[rate > 0]
    assert len(first) > 0 and first[0] > 40.0  # repetitive firing starts at a finite rate (Hopf, type II)
    _, _, warm = _hh_run([10.0], temp=15.0)
    _, _, cold = _hh_run([10.0], temp=6.3)
    assert warm[0] > 1.3 * cold[0]  # faster gates ⇒ faster firing
