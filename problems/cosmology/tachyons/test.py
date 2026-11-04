"""Physics checks, written first (TDD): what any correct (speculative) tachyon kinematics must satisfy.

Run with `pytest problems/cosmology/tachyons/test.py`. Each test is a law of special relativity applied to an
imaginary mass: the invariant m² is negative in every frame, tachyons are always faster than light, a fast enough
boost reverses the time order of emission and reception (and the sign of the energy, Feinberg's reinterpretation),
rapidities add, the antitelephone loop closes exactly where predicted and tachyon condensation lowers the vacuum
energy. No tachyon has ever been observed; this is the mathematics of a hypothesis.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
MU = 2.0  # |m| of the tachyon, units with c = 1


def test_invariant_mass_squared_is_negative_in_every_frame():
    p = torch.linspace(2.01, 40.0, 50, dtype=torch.float64)  # momenta above |m|
    m = torch.tensor(1j * MU, dtype=torch.complex128)  # imaginary mass
    e = equation.energy(p, m)  # complex sqrt of p² + m²
    assert e.imag.abs().max() < 1e-12  # still a real energy
    four = torch.stack([e.real, p], dim=-1)  # (P, 2) = (E, p)
    beta = torch.linspace(-0.95, 0.95, 21, dtype=torch.float64)
    boosted = equation.boost(four, beta)  # (B, P, 2)
    m2 = equation.minkowski_square(boosted)
    assert torch.allclose(m2, torch.full_like(m2, -(MU**2)), atol=1e-9)  # E'² − p'² = m² = −μ² < 0 everywhere


def test_tachyons_are_always_superluminal_and_speed_up_as_they_lose_energy():
    p = torch.logspace(math.log10(MU * 1.000001), 4, 400, dtype=torch.float64)
    m = torch.tensor(1j * MU, dtype=torch.complex128)
    e = equation.energy(p, m).real
    v = equation.velocity(p, e)
    assert (v > 1).all()  # v > c for every real momentum
    assert (v.diff() < 0).all() and (e.diff() > 0).all()  # more energy → slower
    assert v[0] > 100 and v[-1] == pytest.approx(1.0, abs=1e-6)  # v → ∞ as E → 0, v → c as E → ∞
    gamma = equation.lorentz_factor(v.to(torch.complex128))  # imaginary for v > c
    assert torch.allclose((m * gamma).real, e, rtol=1e-6)  # E = mγ: two imaginary factors give a real energy
    assert torch.allclose(v, torch.sqrt(1 + MU**2 / e**2), rtol=1e-12)


def test_time_order_and_energy_sign_flip_beyond_c_squared_over_v():
    u = torch.tensor([1.5, 2.0, 4.0, 10.0], dtype=torch.float64)  # signal speeds
    events = torch.stack([1 / u, torch.ones_like(u)], dim=-1)  # reception at x = 1, t = 1/u; emission at origin
    beta = torch.linspace(-0.99, 0.99, 199, dtype=torch.float64)
    dt = equation.boost(events, beta)[..., 0]  # (B, U) arrival time in the moving frame
    flipped = dt < 0
    assert torch.equal(flipped, beta[:, None] * u[None, :] > 1)  # order reverses iff β > c²/v
    e = MU / torch.sqrt(u**2 - 1)  # tachyon energy E = μ/√(u² − 1)
    four = torch.stack([e, u * e], dim=-1)
    e_boost = equation.boost(four, beta)[..., 0]
    assert torch.equal(e_boost < 0, flipped)  # negative energy exactly when it runs backward in time
    assert torch.equal(solve.reinterpret(dt, e_boost)["emitter_is_receiver"], flipped)  # Feinberg: antitachyon


def test_rapidity_addition():
    eta1 = torch.linspace(-2, 2, 9, dtype=torch.float64)[:, None]
    eta2 = torch.linspace(-1.5, 1.5, 7, dtype=torch.float64)[None, :]
    w = equation.velocity_add(torch.tanh(eta1), -torch.tanh(eta2))  # seen from a frame moving at −tanh η₂
    assert torch.allclose(w, torch.tanh(eta1 + eta2), atol=1e-12)  # subluminal rapidities add
    zeta = torch.linspace(0.1, 2.0, 9, dtype=torch.float64)[:, None]  # tachyon u = coth ζ > 1
    u_seen = equation.velocity_add(1 / torch.tanh(zeta), torch.tanh(eta2))
    assert torch.allclose(u_seen, 1 / torch.tanh(zeta - eta2), atol=1e-9)  # coth ζ → coth(ζ − η)
    assert (u_seen.abs() > 1).all()  # a tachyon is a tachyon in every frame


def test_antitelephone_closes_where_predicted():
    u = torch.linspace(1.2, 12.0, 60, dtype=torch.float64)[None, :]
    beta = torch.linspace(0.05, 0.95, 40, dtype=torch.float64)[:, None]
    loop = equation.antitelephone(u, beta, t_send=1.0)
    closed_form = u**2 * (1 - beta**2) / (u - beta) ** 2  # t_back / t_send
    assert torch.allclose(loop["t_back"], closed_form, rtol=1e-10)
    threshold = (1 + torch.sqrt(1 - beta**2)) / beta
    paradox = loop["t_back"] < 1.0
    assert torch.equal(paradox, u > threshold)  # the reply arrives before the message is sent
    assert (loop["t_receive"] > 0).all()


def test_condensation_lowers_the_vacuum_energy():
    m2, lam = -1.0, 0.5
    v = equation.vacuum(m2, lam)
    assert v == pytest.approx(math.sqrt(2.0))
    phi = torch.linspace(-3, 3, 6001, dtype=torch.float64, requires_grad=True)
    pot = equation.potential(phi, m2, lam)
    assert pot.detach().min().item() == pytest.approx(-(m2**2) / (4 * lam), rel=1e-6)  # V_min = −m⁴/4λ < V(0) = 0
    (grad,) = torch.autograd.grad(pot.sum(), phi, create_graph=True)
    (curv,) = torch.autograd.grad(grad.sum(), phi)
    at_v = curv[(phi - v).abs().argmin()].item()
    assert at_v == pytest.approx(-2 * m2, rel=1e-3)  # fluctuations about the true vacuum have mass² = 2|m²| > 0
    assert curv[3000].item() == pytest.approx(m2)  # at φ = 0 the mass² is the tachyonic m² < 0


def test_lattice_condensation_grows_then_settles():
    m2, lam = -1.0, 1.0
    phi0 = tensors.noise_field(128, 1e-4, seed=0)
    free = solve.condense(phi0, m2, lam, length=40.0, t_max=12.0, dt=0.01, friction=0.0)
    mean0 = phi0.mean().abs()
    early = free["phi"][int(round(5.0 / 0.01))].mean().abs()
    assert early.item() == pytest.approx(mean0.item() * math.cosh(5.0), rel=0.05)  # k = 0 mode grows as cosh(|m| t)
    drift = (free["energy"] - free["energy"][0]).abs().max().item()
    assert drift < 1e-3 * free["kinetic"].max().item()  # no friction: total energy conserved (leapfrog)
    damped = solve.condense(phi0, m2, lam, length=40.0, t_max=40.0, dt=0.02, friction=0.5)
    assert (damped["energy"].diff() < 1e-3 * damped["kinetic"].max()).all()  # friction only removes energy
    assert damped["energy"][-1] < damped["energy"][0]
    v = equation.vacuum(m2, lam)
    assert damped["phi"][-1].abs().mean().item() == pytest.approx(v, rel=0.1)  # domains of φ ≈ ±v
    assert damped["potential"][-1].item() < 0.8 * equation.potential(torch.tensor(v), m2, lam).item()  # V ≈ V_min < 0


def test_opera_as_a_tachyon_contradicts_sn1987a():
    data = tensors.load_measurements()
    opera = data["OPERA (first result)"]
    mu = equation.mass_from_speed(opera["energy_gev"], opera["dv_over_c"])  # implied |m| in GeV
    assert 0.1 < mu < 0.13  # ≈ 120 MeV
    sn = data["SN1987A"]
    predicted = equation.speed_excess(sn["energy_gev"], mu)  # (v − c)/c at 10 MeV
    assert predicted > 1e6 * sn["err_plus"]  # off by more than a factor of a million: the tachyon reading fails
    corrected = data["OPERA (corrected)"]
    assert abs(corrected["dv_over_c"]) < 1.5 * corrected["err_plus"]  # the corrected value is consistent with c
