"""Physics checks, written first (TDD): what any correct topological-photonics (Haldane) code must satisfy.

Run with `pytest problems/em/topological_photonics/test.py`. The Chern number is an integer equal to the Berry flux
over 2π, a ribbon hosts one-way edge states in the gap only when C ≠ 0 (bulk–edge correspondence), and edge light
passes a notch under unitary evolution.
"""

import math

import pytest
import torch

from . import solve

CPU = torch.device("cpu")


@pytest.mark.parametrize("phi, M, C", [(math.pi / 2, 0.0, 1), (-math.pi / 2, 0.0, -1), (math.pi / 2, 0.5, 1)])
def test_chern_number_topological(phi, M, C):
    r = solve.chern_number(1.0, 0.15, phi, M, 36, CPU)
    assert r["C"] == pytest.approx(C, abs=1e-9)  # the Chern number is quantised
    assert r["omega"].sum().item() * r["cell"] == pytest.approx(2 * math.pi * C, abs=1e-9)  # Berry flux = 2πC


@pytest.mark.parametrize("phi, M", [(math.pi / 2, 1.2), (0.0, 0.3)])
def test_chern_number_trivial(phi, M):
    r = solve.chern_number(1.0, 0.15, phi, M, 36, CPU)
    assert r["C"] == pytest.approx(0.0, abs=1e-9)  # time-reversal symmetric (φ = 0) or large mass: trivial
    assert r["gap"] > 0.1


def test_berry_curvature_formula_integrates_to_two_pi_chern():
    for M, C in [(0.0, 1), (0.5, 1), (1.2, 0)]:
        r = solve.chern_number(1.0, 0.15, math.pi / 2, M, 64, CPU)
        omega = solve.berry_curvature_autograd(r["k"], 1.0, 0.15, math.pi / 2, M)
        assert omega.sum().item() * r["cell"] == pytest.approx(2 * math.pi * C, abs=2e-3)
        # and, at the plaquette centres, the continuum formula agrees with the FHS fluxes point by point
        k = r["k"]
        centre = k + 0.5 * (k[1, 0] - k[0, 0]) + 0.5 * (k[0, 1] - k[0, 0])
        omega_c = solve.berry_curvature_autograd(centre, 1.0, 0.15, math.pi / 2, M)
        assert (omega_c - r["omega"]).norm() < 0.02 * r["omega"].norm() + 1e-9


def test_ribbon_has_edge_states_in_gap_only_when_topological():
    window = 0.3 * 1.559  # well inside the bulk gap 2(3√3 t2) for M = 0
    topo = solve.ribbon_bands(20, 1.0, 0.15, math.pi / 2, 0.0, 101, CPU)
    triv = solve.ribbon_bands(20, 1.0, 0.15, math.pi / 2, 1.56, 101, CPU)
    in_gap = topo["E"].abs() < window
    assert in_gap.any()
    assert ((topo["bottom"] + topo["top"])[in_gap] > 0.9).all()  # bulk–edge correspondence: mid-gap states on edges
    assert not (triv["E"].abs() < 0.2).any()  # trivial: nothing crosses the middle of the gap
    v, _ = solve.bottom_edge_mode(topo, window)
    v_rev, _ = solve.bottom_edge_mode(solve.ribbon_bands(20, 1.0, 0.15, -math.pi / 2, 0.0, 101, CPU), window)
    assert v * v_rev < 0 and abs(v) > 0.3  # one-way, and the direction flips with φ


def test_edge_transport_is_robust_to_a_notch():
    t2, phi = 0.15, math.pi / 2
    topo = solve.chern_number(1.0, t2, phi, 0.0, 36, CPU)
    window = 0.3 * topo["gap"]
    v, k_c = solve.bottom_edge_mode(solve.ribbon_bands(20, 1.0, t2, phi, 0.0, 101, CPU), window)
    r = solve.edge_transport(1.0, t2, phi, (0.0, 1.56), 0.3, 3.0, v, k_c, window, 40.0, 24.0, 40, CPU)
    norm = r["intensity"].sum(-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-10)  # unitary evolution conserves photon number
    assert r["transmitted"][0, -1] > 0.85  # topological: almost all light goes around the notch
    assert r["transmitted"][1, -1] < 0.35  # trivial: it does not
