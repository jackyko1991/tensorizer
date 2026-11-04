"""Physics checks, written first (TDD): what any correct model of re-igniting stellar remnants must satisfy.

Run with `pytest problems/astro/dead_star_reignition/test.py`. Exact results (Lane–Emden polytropes, the
Chandrasekhar mass, the Eddington luminosity, ISCO efficiencies) are checked tightly; the one-zone ignition model
is an order-of-magnitude model, so it is checked against published ranges (brown-dwarf mass limits, nova
ignition masses, X-ray burst ignition depths), not to many digits.
"""

import math

import numpy as np
import pytest
import torch

from . import equation, plot, solve, tensors

F64 = torch.float64


@pytest.fixture(scope="module")
def poly():
    return solve.polytropes(torch.tensor([1.0, 1.5, 3.0], dtype=F64))


def test_lane_emden_known_values(poly):
    assert poly["xi1"][0].item() == pytest.approx(math.pi, rel=1e-5)  # n = 1: θ = sin ξ / ξ
    assert poly["xi1"][1].item() == pytest.approx(3.65375, rel=1e-4)
    assert poly["xi1"][2].item() == pytest.approx(6.89685, rel=1e-4)
    assert poly["omega"][2].item() == pytest.approx(2.01824, rel=1e-4)  # −ξ² θ'(ξ₁) for n = 3
    assert poly["rho_ratio"][1].item() == pytest.approx(5.99071, rel=1e-3)  # ρ_c / ρ̄ for n = 3/2


def test_chandrasekhar_mass(poly):
    m_ch = equation.chandrasekhar_mass(poly["omega"][2], mu_e=2.0) / equation.M_SUN
    assert m_ch.item() == pytest.approx(1.44, rel=0.03)  # ≈ 1.44–1.46 M☉ for carbon–oxygen (μ_e = 2)
    ratio = equation.chandrasekhar_mass(poly["omega"][2], mu_e=1.0) / equation.chandrasekhar_mass(poly["omega"][2], 2.0)
    assert ratio.item() == pytest.approx(4.0)  # M_Ch ∝ μ_e⁻²
    # the white-dwarf radius shrinks to zero at M_Ch
    m = torch.tensor([0.4, 0.7, 0.95, 0.999999], dtype=F64) * m_ch * equation.M_SUN
    R = equation.white_dwarf_radius(m, m_ch * equation.M_SUN)
    assert (R[:3].diff() < 0).all() and R[3] < 0.02 * R[0]
    assert R[0] / equation.R_SUN == pytest.approx(0.0112 * (0.4 ** (-2 / 3) - 0.4 ** (2 / 3)) ** 0.5, rel=1e-6)


def test_eos_inversion_and_degenerate_limit():
    rho = torch.tensor([1e-2, 1.0, 1e3, 1e6, 1e9], dtype=F64)
    T = torch.tensor([1e4, 1e6, 1e7, 1e8, 1e7], dtype=F64)
    comp = tensors.composition("solar", like=rho)
    P = equation.pressure(rho, T, *comp)
    assert torch.allclose(equation.density(P, T, *comp), rho, rtol=1e-6)
    # cold, non-relativistic electrons: P → (3π²)^{2/3} ħ² n_e^{5/3} / (5 m_e)
    n_e = torch.tensor([1e25], dtype=F64)  # p_F << m_e c
    expected = (3 * math.pi**2) ** (2 / 3) * equation.HBAR**2 / (5 * equation.M_E) * n_e ** (5 / 3)
    assert equation.degenerate_pressure(n_e).item() == pytest.approx(expected.item(), rel=1e-3)


def test_brown_dwarf_mass_limits(poly):
    M = torch.logspace(math.log10(0.004), math.log10(0.3), 300, dtype=F64) * equation.M_SUN
    T_max = solve.brown_dwarf_max_temperature(M, poly)
    m_h = solve.crossing(M, T_max, tensors.T_IGNITE["H"]) / equation.M_SUN
    m_d = solve.crossing(M, T_max, tensors.T_IGNITE["D"]) / equation.M_JUP
    assert 0.06 < m_h < 0.09  # hydrogen-burning limit ≈ 0.07–0.08 M☉ (Burrows et al. 2001)
    assert 10 < m_d < 20  # deuterium-burning limit ≈ 11–14 M_J (Spiegel et al. 2011)
    # low masses (weak degeneracy): T_max ∝ M^{4/3}
    slope = (T_max[20].log() - T_max[0].log()) / (M[20].log() - M[0].log())
    assert slope.item() == pytest.approx(4 / 3, abs=0.05)


def test_eddington_luminosity_and_black_hole_efficiency():
    M = torch.tensor([1.0, 10.0, 4e6], dtype=F64) * equation.M_SUN
    L = equation.eddington_luminosity(M)
    assert L[0].item() == pytest.approx(1.26e38, rel=0.01)  # erg/s per solar mass (electron scattering, pure H)
    assert torch.allclose(L / M, (L / M)[0].expand(3))  # ∝ M
    eta = equation.accretion_efficiency(torch.tensor([0.0, 1.0], dtype=F64))
    assert eta[0].item() == pytest.approx(1 - math.sqrt(8 / 9), rel=1e-9)  # 5.7 % for Schwarzschild
    assert eta[1].item() == pytest.approx(1 - 1 / math.sqrt(3), rel=1e-9)  # 42 % for extremal Kerr
    mdot = torch.logspace(-3, 2, 6, dtype=F64) * equation.eddington_rate(M[:1], eta[:1])
    lum = equation.black_hole_luminosity(mdot, M[:1], eta[:1])
    assert (lum <= L[0] * (1 + 1e-12)).all() and lum[0] == pytest.approx(1e-3 * L[0])


@pytest.fixture(scope="module")
def shells(poly):
    cases = tensors.shell_cases(
        wd_masses=[0.6, 1.0, 1.25],
        wd_rates=[1e-11, 1e-9, 1e-7],
        ns_rates=[1e-9],
        m_ch=equation.chandrasekhar_mass(poly["omega"][2], 2.0).item(),
        core_temp=1e7,
    )
    return cases, solve.ignition(cases)


def test_nova_ignition_masses(shells):
    cases, ign = shells
    wd = cases["kind"] == 0
    dm = (ign["dM"][wd] / equation.M_SUN).reshape(3, 3)  # (WD mass, accretion rate)
    assert ign["ignited"][wd].all()
    assert (dm > 1e-7).all() and (dm < 1e-3).all()  # nova grids: ~1e-7 … 1e-3 M☉ (Yaron et al. 2005)
    assert (dm.diff(dim=0) < 0).all()  # heavier white dwarf → smaller, denser shell ignites (g ∝ M/R²)
    assert (dm.diff(dim=1) < 0).all()  # faster accretion → hotter base → earlier ignition


def test_x_ray_burst_ignition(shells):
    cases, ign = shells
    ns = cases["kind"] == 1
    y = ign["y"][ns]
    assert ign["ignited"][ns].all()
    assert ((y > 1e7) & (y < 1e9)).all()  # ignition column ~ 1e8 g/cm² (Bildsten 1998; Galloway & Keek 2021)
    t_rec = ign["dM"][ns] / cases["Mdot"][ns] / 3600
    assert ((t_rec > 0.3) & (t_rec < 50)).all()  # bursts recur every few hours
    alpha = equation.burst_alpha(cases["M"][ns], cases["R"][ns], cases["X"][ns])
    assert ((alpha > 20) & (alpha < 100)).all()  # observed α ≈ 40–100: gravity releases far more than fusion


def test_millisecond_pulsar_spin_up_mass():
    dm = equation.spin_up_mass(torch.tensor([3e-3], dtype=F64), 1.4 * equation.M_SUN) / equation.M_SUN
    assert 0.02 < dm.item() < 0.2  # ~0.1 M☉ of accreted mass recycles a pulsar to millisecond periods


# ---------- interactive 3D scenes (VTK) ----------
@pytest.fixture(scope="module")
def scenes(poly, tmp_path_factory):
    """Write the four scenes for a default-like run (white dwarf 1 M☉, neutron star, 10 M☉ black hole a = 0.5)."""
    m_ch = equation.chandrasekhar_mass(poly["omega"][2], 2.0).item()
    M_bd = tensors.log_ramp(0.004, 0.3, 300) * equation.M_SUN
    T_max = solve.brown_dwarf_max_temperature(M_bd, poly)
    mine = tensors.shell_cases([1.0], [1e-9], [1e-9], m_ch, 1e7, ("solar",))
    M_bh = torch.tensor([10 * equation.M_SUN], dtype=F64)
    spin = 0.5
    eta = equation.accretion_efficiency(torch.tensor([spin], dtype=F64))
    d = {
        "M_bd": M_bd, "T_max": T_max, "m_h": solve.crossing(M_bd, T_max, tensors.T_IGNITE["H"]),
        "bd_m0": 0.04 * equation.M_SUN, "bd_rate": 1e-10 * equation.M_SUN / equation.YEAR,
        "mine": mine, "me": solve.ignition(mine),
        "M_bh": M_bh, "eta": eta, "L_edd": equation.eddington_luminosity(M_bh), "bh_spin": spin,
    }  # fmt: skip
    out = tmp_path_factory.mktemp("dsr")
    return plot.export_vtk(d, out, n_steps=30), spin


EXPECTED_LAYERS = {
    "brown_dwarf": {"star": {"T_central"}, "stream": {"speed"}, "ignition": {"flag"}},
    "white_dwarf": {"star": {"radius"}, "shell": {"dM"}, "ejecta": {"radius", "age"}},
    "neutron_star": {"star": {"burning"}, "column": {"fill"}},
    "black_hole": {"horizon": {"radius"}, "disc": {"T_disc", "brightness"}, "photon_ring": {"radius"}},
}


def _collection(pvd):
    import xml.etree.ElementTree as ET

    return [
        (float(e.get("timestep")), e.get("group"), pvd.parent / e.get("file")) for e in ET.parse(pvd).iter("DataSet")
    ]


def test_vtk_scenes_parse_back(scenes):
    pvds, _ = scenes
    assert [p.stem for p in pvds] == list(EXPECTED_LAYERS)
    for pvd in pvds:
        entries = _collection(pvd)
        layers = EXPECTED_LAYERS[pvd.stem]
        assert {g for _, g, _ in entries} == set(layers)
        times = sorted({t for t, _, _ in entries})
        assert 24 <= len(times) <= 40 and all(sum(g == L for _, g, _ in entries) == len(times) for L in layers)
        assert sum(f.stat().st_size for _, _, f in entries) + pvd.stat().st_size < 2e6  # under ~2 MB per scene
        for _, layer, f in entries:
            v = plot.read_vtp(f)
            n_cells = v["Verts"] + v["Lines"] + v["Polys"]
            assert v["points"].shape == (v["Points"], 3) and np.isfinite(v["points"]).all()
            assert len(v["offsets"]) == n_cells and (v["offsets"][-1] if n_cells else 0) == len(v["connectivity"])
            assert (v["connectivity"] < max(v["Points"], 1)).all() and (v["connectivity"] >= 0).all()
            assert set(v["arrays"]) == layers[layer] and all(len(a) == v["Points"] for a in v["arrays"].values())
            assert all(v["units"].values())  # every array carries a unit
            if v["Points"]:
                assert v["kind"] == ("Lines" if layer in ("stream", "column", "photon_ring") else "Polys")
    sphere = plot.read_vtp(_collection(pvds[2])[0][2])  # the neutron-star sphere: 24 × 12 segments
    assert (sphere["Points"], sphere["Polys"]) == (24 * 11 + 2, 24 * 12)


def test_vtk_black_hole_geometry(scenes, tmp_path):
    pvds, spin = scenes
    first = {g: f for t, g, f in _collection(pvds[3]) if t == _collection(pvds[3])[0][0]}
    horizon, disc = plot.read_vtp(first["horizon"]), plot.read_vtp(first["disc"])
    r_isco = equation.isco_radius(torch.tensor(spin, dtype=F64)).item()
    r_disc = np.linalg.norm(disc["points"], axis=1)
    assert r_disc.min() == pytest.approx(r_isco, rel=1e-5)  # the disc starts at the ISCO
    assert np.linalg.norm(horizon["points"], axis=1) == pytest.approx(1 + math.sqrt(1 - spin**2), rel=1e-6)
    assert (
        disc["arrays"]["T_disc"].max() <= 1 + 1e-6 and disc["arrays"]["T_disc"][r_disc < r_isco * 1.0001].max() < 1e-6
    )
    # Schwarzschild: horizon 2, ISCO 6, photon ring 3 GM/c² (1.5 r_s); extremal Kerr: all → 1
    for a, (r_h, r_in, r_ph) in ((0.0, (2.0, 6.0, 3.0)), (0.998, (1.063, 1.237, 1.074))):
        pvd = plot.black_hole_scene(tmp_path, a, np.linspace(1e-3, 1, 24), np.linspace(1e-3, 1, 24))
        f = {g: p for t, g, p in _collection(pvd) if t == 1e-3}
        assert np.linalg.norm(plot.read_vtp(f["horizon"])["points"], axis=1) == pytest.approx(r_h, rel=2e-3)
        assert np.linalg.norm(plot.read_vtp(f["disc"])["points"], axis=1).min() == pytest.approx(r_in, rel=2e-3)
        assert np.linalg.norm(plot.read_vtp(f["photon_ring"])["points"], axis=1) == pytest.approx(r_ph, rel=2e-3)
    # differential rotation: per step the inner edge turns further than the outer edge, by (r_out / r_in)^{3/2}
    second = {g: f for t, g, f in _collection(pvds[3]) if t == sorted({e[0] for e in _collection(pvds[3])})[1]}
    p0, p1 = disc["points"], plot.read_vtp(second["disc"])["points"]
    turn = np.angle((p1[:, 0] + 1j * p1[:, 1]) / (p0[:, 0] + 1j * p0[:, 1]))
    assert turn[r_disc.argmin()] / turn[r_disc.argmax()] == pytest.approx((r_disc.max() / r_isco) ** 1.5, rel=1e-3)
