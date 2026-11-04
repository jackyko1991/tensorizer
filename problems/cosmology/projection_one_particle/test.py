"""Checks, written first (TDD): what any correct count of slices through one zig-zagging worldline must satisfy.

Run with `pytest problems/cosmology/projection_one_particle/test.py`. The hypothesis (all particles are crossings of
one worldline, Wheeler's one-electron universe) is speculative; these tests check its mathematics. A time slice cuts
the worldline: forward crossings are particles, backward crossings antiparticles. The net count is topological (it
depends only on the ends), a closed loop has zero net charge on every slice, turning points create and annihilate
pairs, and the average number of crossings of a random smooth loop follows Rice's formula.
"""

import math

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_fast_crossings_match_the_slow_reference():
    X = tensors.smooth_worldlines(64, 400, modes=8, roughness=1.0, dims=3, closed=True, generator=gen(1), device=CPU)
    levels = torch.linspace(-2.5, 2.5, 41, dtype=torch.float64)
    up, down = equation.crossings(X[..., 0], levels, closed=True)  # (B, L) by scatter + cumsum
    up_ref, down_ref = equation.crossings_reference(X[..., 0], levels, closed=True)  # (B, S, L) brute force
    assert torch.equal(up, up_ref) and torch.equal(down, down_ref)


def test_closed_worldline_has_zero_net_charge_on_every_slice():
    for kind in ("smooth", "random"):
        X = tensors.worldlines(kind, 128, 512, modes=10, roughness=1.0, dims=4, closed=True, generator=gen(2),
                               device=CPU)  # fmt: skip
        levels = torch.linspace(-4, 4, 81, dtype=torch.float64)
        up, down = equation.crossings(X[..., 0], levels, closed=True)
        assert torch.equal(up, down)  # as many particles as antiparticles, slice by slice
        assert ((up + down) % 2 == 0).all()  # crossings come in pairs


def test_net_charge_is_the_crossing_number_of_the_ends():
    X = tensors.smooth_worldlines(128, 512, modes=10, roughness=1.0, dims=2, closed=False, generator=gen(3), device=CPU)
    t = X[..., 0]
    levels = torch.linspace(-4, 4, 81, dtype=torch.float64)
    up, down = equation.crossings(t, levels, closed=False)
    Q = up - down
    assert torch.equal(Q, equation.net_charge_from_ends(t, levels))  # [t_end >= u] - [t_start >= u]
    assert Q.abs().max() <= 1  # one worldline carries at most one unit of net charge
    assert torch.equal((up + down) % 2, Q.abs() % 2)  # parity: N = Q mod 2


def test_turning_points_create_and_annihilate_pairs():
    X = tensors.worldlines("random", 64, 300, modes=6, roughness=1.0, dims=2, closed=True, generator=gen(4), device=CPU)
    t = X[..., 0]
    levels = torch.linspace(-3, 3, 121, dtype=torch.float64)
    up, down = equation.crossings(t, levels, closed=True)
    N = up + down
    assert torch.equal(N, equation.morse_count(t, levels))  # 2 (#minima below u - #maxima below u)
    births, deaths = equation.turning_points(t, closed=True)  # (B, S) masks: local minima, local maxima
    assert torch.equal(births.sum(1), deaths.sum(1))  # every pair created is annihilated on a closed loop


def test_single_mode_loop_crosses_each_slice_2k_times():
    s = tensors.parameter(2000, closed=True, device=CPU)
    K = torch.arange(1, 6, dtype=torch.float64)
    t = torch.cos(K[:, None] * s + 0.3)  # (5, S): a loop that zig-zags K times in time
    levels = torch.linspace(-0.9, 0.9, 19, dtype=torch.float64)
    up, down = equation.crossings(t, levels, closed=True)
    assert torch.equal(up, K[:, None].expand(-1, 19).long()) and torch.equal(down, up)  # K particles + K antiparticles


def test_mean_crossings_follow_rice_formula():
    B, modes, roughness = 4000, 12, 1.0
    X = tensors.smooth_worldlines(B, 1024, modes, roughness, dims=2, closed=True, generator=gen(5), device=CPU)
    levels = torch.linspace(-2, 2, 9, dtype=torch.float64)
    up, down = equation.crossings(X[..., 0], levels, closed=True)
    N = (up + down).double()
    rice = equation.rice_mean_crossings(levels, tensors.mode_sigmas(modes, roughness, CPU))
    assert ((N.mean(0) - rice).abs() < 5 * N.std(0) / math.sqrt(B) + 1e-3).all()


def test_every_crossing_is_the_same_object():
    X = tensors.smooth_worldlines(16, 400, modes=8, roughness=1.0, dims=3, closed=True, generator=gen(6), device=CPU)
    levels = torch.linspace(-2, 2, 21, dtype=torch.float64)
    props = torch.tensor([0.51099895, -1.0], dtype=torch.float64)  # (mass in MeV, charge in e): the worldline's own
    seen = equation.crossing_properties(X[..., 0], levels, props)  # (n_crossings, 2): mass, charge seen at each
    assert seen.shape[0] > 50
    assert torch.equal(seen[:, 0], torch.full_like(seen[:, 0], props[0]))  # every electron has the same mass
    assert torch.equal(seen[:, 1].abs(), torch.full_like(seen[:, 1], 1.0))  # and the same |charge|
    up, down = equation.crossings(X[..., 0], levels, closed=True)
    assert seen[:, 1].sum() == -(up - down).sum()  # Feynman-Stueckelberg: backward = opposite charge


def test_net_charge_does_not_depend_on_how_the_slice_is_tilted():
    X = tensors.smooth_worldlines(64, 512, modes=8, roughness=1.0, dims=3, closed=False, generator=gen(7), device=CPU)
    levels = torch.linspace(-3, 3, 31, dtype=torch.float64)
    Q0 = equation.net_charge_from_ends(X[..., 0], levels)
    for tilt in (0.2, -0.5):  # slice t - tilt * x = u: a boosted observer
        f = X[..., 0] - tilt * torch.tanh(X[..., 1])
        up, down = equation.crossings(f, levels, closed=False)
        Q = up - down
        assert torch.equal(Q, equation.net_charge_from_ends(f, levels))
        assert Q.abs().max() <= 1 and Q0.abs().max() <= 1


def test_winding_in_a_compact_time_gives_net_charge_w():
    w = torch.tensor([1, 2, 5], dtype=torch.float64)
    t = tensors.winding_worldlines(w, 1500, period=1.0, wiggle=0.3, generator=gen(8), device=CPU)  # unwrapped time
    levels = torch.linspace(0.05, 0.95, 10, dtype=torch.float64)
    up, down = equation.crossings_periodic(t, levels, period=1.0)
    assert torch.equal(up - down, w[:, None].long().expand(-1, 10))  # the winding number is the net charge


def test_simplest_form_cannot_reach_the_observed_asymmetry():
    N = torch.arange(1, 200)
    A_max = equation.max_asymmetry(N, closed=False)  # largest |Q| / N one open worldline allows
    assert A_max[0] == 1 and (A_max[1:] <= 1 / 3 + 1e-12).all()  # only a single crossing is fully asymmetric
    f = tensors.positron_fraction()["fraction"]
    A_obs = equation.asymmetry_from_fraction(f)  # (E,) 1 - 2 f
    assert (A_obs > 0.6).all() and (A_obs < 0.95).all()  # AMS-02: positrons are 5-16% of e+ e-
    assert equation.max_asymmetry(torch.tensor([2, 3, 10**6]), closed=True).abs().max() == 0


def test_eta_from_planck_and_the_net_charge_needed():
    c = tensors.cited()
    eta = equation.eta_from_omega_b(c["omega_b_h2"])
    assert abs(eta - 6.14e-10) < 0.03e-10
    N_B = equation.baryons_in_observable_universe(eta, c["n_gamma_cm3"], c["hubble_radius_gly"])
    assert 5e79 < N_B < 2e80  # ~10^80 baryons: the net charge one worldline would have to carry


@pytest.mark.slow
def test_run_accepts_dimensions_as_parameters(tmp_path):
    from . import main

    params = {"worldlines": 64, "samples": 256, "modes": 6, "roughness": 1.0, "dims": 5, "slices": 41}
    out = main.run(params, tmp_path)
    assert out.exists() and out.stat().st_size > 0
