"""Physics checks, written first (TDD): what any correct 3D lattice-Boltzmann wind tunnel must satisfy.

Run with `pytest problems/fluids/lbm_wind_tunnel/test.py`. The D3Q19 equilibrium carries the right mass, momentum and
momentum flux; the moment-space (MRT) collision reduces to BGK and TRT for the right rates; collisions, streaming and
bounce-back conserve mass; a shear wave decays at the rate set by the viscosity; the ray-casting voxeliser recovers the
volume of a sphere and of the bundled wing and fills overlapping parts (the airplane); and the momentum-exchange drag
of a sphere agrees with the Schiller–Naumann correlation and with a precomputed OpenLB run of the same case
(data/openlb_reference_*.csv).
"""

import math
from pathlib import Path

import pytest
import torch

from . import equation, plot, solve, tensors

CPU = torch.device("cpu")
F64 = torch.float64
DATA = Path(__file__).parent / "data"


def random_state(shape=(5, 4, 3), seed=0):
    gen = torch.Generator().manual_seed(seed)
    rho = 1 + 0.05 * torch.rand(*shape, generator=gen, dtype=F64)
    u = 0.05 * torch.randn(3, *shape, generator=gen, dtype=F64)
    return rho, u, gen


def test_equilibrium_moments():
    e, w, _ = tensors.lattice(CPU, F64)
    rho, u, _ = random_state()
    f_eq = equation.equilibrium(rho, u, e, w)
    assert w.sum().item() == pytest.approx(1.0)
    assert torch.allclose(equation.density(f_eq), rho, atol=1e-12)  # Σ_i f_eq = ρ
    assert torch.allclose(equation.momentum(f_eq, e), rho * u, atol=1e-12)  # Σ_i f_eq e_i = ρu
    flux = torch.einsum("ia,ib,i...->ab...", e, e, f_eq)  # Σ_i f_eq e_ia e_ib
    expected = (
        torch.einsum("...,a...,b...->ab...", rho, u, u) + rho / 3 * torch.eye(3, dtype=F64)[..., None, None, None]
    )
    assert torch.allclose(flux, expected, atol=1e-12)  # = ρ u_a u_b + ρ c_s² δ_ab, c_s² = 1/3


def test_moment_basis_is_orthogonal_with_conserved_rows():
    e, _, opp = tensors.lattice(CPU, F64)
    m, odd = tensors.moment_basis(e)
    gram = m @ m.T
    assert torch.allclose(gram, torch.diag(torch.diagonal(gram)), atol=1e-9)  # rows mutually orthogonal
    assert torch.allclose(m[0], torch.ones(19, dtype=F64))  # first moment: density
    assert torch.allclose(m[:, opp], torch.where(odd[:, None], -m, m))  # odd rows flip sign under e → -e


def test_mrt_reduces_to_bgk_and_trt():
    e, w, opp = tensors.lattice(CPU, F64)
    m, odd = tensors.moment_basis(e)
    rho, u, gen = random_state()
    f = equation.equilibrium(rho, u, e, w) + 1e-3 * torch.randn(19, 5, 4, 3, generator=gen, dtype=F64)
    f_eq = equation.equilibrium(equation.density(f), equation.momentum(f, e) / equation.density(f), e, w)
    omega = 1 / 0.62
    equal_rates = torch.full((19,), omega, dtype=F64)
    assert torch.allclose(equation.collide_mrt(f, f_eq, m, equal_rates), equation.collide_bgk(f, f_eq, omega),
                          atol=1e-13)  # fmt: skip
    omega_minus = equation.magic_omega_minus(torch.tensor(omega, dtype=F64))
    parity_rates = torch.where(odd, omega_minus, torch.tensor(omega, dtype=F64))
    assert torch.allclose(equation.collide_mrt(f, f_eq, m, parity_rates),
                          equation.collide_trt(f, f_eq, omega, omega_minus, opp), atol=1e-13)  # fmt: skip
    lam = (1 / omega - 0.5) * (1 / omega_minus - 0.5)
    assert lam.item() == pytest.approx(3 / 16)  # the "magic" TRT combination


@pytest.mark.parametrize("les", [False, True])
def test_mass_and_momentum_conserved_in_periodic_box(les):
    e, w, _ = tensors.lattice(CPU, F64)
    rho, u, _ = random_state((12, 10, 8), seed=3)
    f0 = equation.equilibrium(rho, u, e, w)
    solid = torch.zeros(12, 10, 8, dtype=torch.bool)
    out = solve.simulate(f0, solid, u_in=0.05, tau=0.52, n_steps=60, les=les, walls=False, open_ends=False)
    f = out["f"]
    assert f.sum().item() == pytest.approx(f0.sum().item(), rel=1e-12)  # mass to round-off
    p0, p1 = equation.momentum(f0, e).sum((1, 2, 3)), equation.momentum(f, e).sum((1, 2, 3))
    assert torch.allclose(p0, p1, atol=1e-10)  # no body, no walls: total momentum conserved too


def test_bounce_back_body_conserves_mass_and_feels_force():
    e, w, _ = tensors.lattice(CPU, F64)
    n = 16
    solid = tensors.voxelise(tensors.sphere_mesh(3.0, (8.0, 7.5, 7.5), dtype=F64), (n, n, n))
    rho = torch.ones(n, n, n, dtype=F64)
    u = torch.zeros(3, n, n, n, dtype=F64)
    u[0] = 0.04
    u[:, solid] = 0
    f0 = equation.equilibrium(rho, u, e, w)
    out = solve.simulate(f0, solid, u_in=0.04, tau=0.6, n_steps=80, walls=False, open_ends=False)
    assert out["f"].sum().item() == pytest.approx(f0.sum().item(), rel=1e-12)  # halfway bounce-back loses nothing
    assert out["force"][-20:, 0].mean() > 0  # the flow pushes the sphere downstream


def test_shear_wave_decays_at_viscous_rate():
    """u_x = U sin(k y) decays as exp(-ν k² t) with ν = (τ - ½)/3; TRT and TRT+LES (no shear → no eddy viscosity)."""
    e, w, _ = tensors.lattice(CPU, F64)
    ny, tau, amp = 32, 0.7, 1e-3
    k = 2 * math.pi / ny
    y = torch.arange(ny, dtype=F64)
    u = torch.zeros(3, 2, ny, 2, dtype=F64)
    u[0] = amp * torch.sin(k * y)[None, :, None]
    f0 = equation.equilibrium(torch.ones(2, ny, 2, dtype=F64), u, e, w)
    n = 400
    out = solve.simulate(f0, torch.zeros(2, ny, 2, dtype=torch.bool), u_in=0.0, tau=tau, n_steps=n, walls=False,
                         open_ends=False)  # fmt: skip
    ux = equation.momentum(out["f"], e)[0, 0, :, 0] / equation.density(out["f"])[0, :, 0]
    measured = torch.dot(ux, torch.sin(k * y)) / torch.dot(torch.sin(k * y), torch.sin(k * y)) / amp
    nu = (tau - 0.5) / 3
    assert measured.item() == pytest.approx(math.exp(-nu * k**2 * n), rel=5e-3)


def test_voxelised_sphere_and_cube_have_the_right_volume():
    r = 7.0
    sphere = tensors.sphere_mesh(r, (12.0, 12.3, 11.8), n_theta=48, n_phi=96, dtype=F64)
    assert tensors.mesh_volume(sphere).item() == pytest.approx(4 / 3 * math.pi * r**3, rel=0.01)  # divergence theorem
    count = tensors.voxelise(sphere, (24, 24, 24)).sum().item()
    assert count == pytest.approx(4 / 3 * math.pi * r**3, rel=0.03)  # ray parity: inside cells ≈ volume
    cube = tensors.box_mesh((2.5, 3.5, 4.5), (12.5, 9.5, 10.5), dtype=F64)  # 10 × 6 × 6 cells exactly
    assert tensors.voxelise(cube, (16, 12, 14)).sum().item() == 360


def test_bundled_wing_stl_voxelises_to_its_volume():
    tri = tensors.read_stl(DATA / "naca0012_wing.stl", dtype=F64)
    volume = tensors.mesh_volume(tri).item()  # mm³ by the divergence theorem
    assert volume == pytest.approx(0.0822 * 1000**2 * 1200, rel=0.01)  # NACA 0012 section area 0.0822 c², span 1200
    body, scale = tensors.place_body(tri, (110, 128, 24), length=100.0, nose_x=10.0, clearance=None)
    assert scale == pytest.approx(0.1)
    count = tensors.voxelise(body, (110, 128, 24)).sum().item()
    assert count == pytest.approx(volume * scale**3, rel=0.03)  # cells inside ≈ volume (lattice units)


def test_moving_ground_and_slip_ceiling_keep_uniform_flow():
    """With the walls on (ground moving at U, free-slip ceiling) uniform flow at U is an exact steady state."""
    e, w, _ = tensors.lattice(CPU, F64)
    solid = torch.zeros(12, 6, 8, dtype=torch.bool)
    f0 = tensors.initial_populations(solid, 0.06, e, w)
    out = solve.simulate(f0, solid, u_in=0.06, tau=0.6, n_steps=50, walls=True, open_ends=False)
    assert torch.allclose(out["f"], f0, atol=1e-12)


def test_overlapping_parts_stay_solid():
    """Two overlapping closed boxes (like a wing through a fuselage): the winding-number voxeliser fills their union,
    where plain ray parity would hollow out the overlap."""
    a = tensors.box_mesh((2.5, 2.5, 2.5), (8.5, 6.5, 6.5), dtype=F64)  # 6 × 4 × 4 = 96 cells
    b = tensors.box_mesh((5.5, 3.5, 3.5), (11.5, 5.5, 5.5), dtype=F64)  # 6 × 2 × 2 = 24 cells, 12 shared
    assert tensors.voxelise(torch.cat([a, b]), (14, 9, 9)).sum().item() == 96 + 24 - 12


def test_bundled_airplane_and_wing_stl_are_closed():
    for name, extent in (("airplane.stl", (1051.0, 900.0, 270.0)), ("naca0012_wing.stl", (1000.0, 1200.0, 120.0))):
        tri = tensors.read_stl(DATA / name, dtype=F64)
        lo, hi = tri.reshape(-1, 3).min(0).values, tri.reshape(-1, 3).max(0).values
        assert torch.allclose(hi - lo, torch.tensor(extent, dtype=F64), atol=1.0)
        assert tensors.edge_use_counts(tri).eq(2).all()  # closed parts: every edge shared by two triangles
        assert tensors.mesh_volume(tri).item() > 0  # outward normals
    body, _ = tensors.place_body(tensors.read_stl(DATA / "airplane.stl"), (80, 36, 26), 32.0, 12.0, None, 8.0)
    solid = tensors.voxelise(body, (80, 36, 26))
    assert solid[:, :, 0].sum() == 0 and solid[:, :, -1].sum() == 0  # centred: clear of the periodic z boundary
    assert tensors.planform_area(solid) > 300  # the wings survive voxelisation


def test_schiller_naumann_limits():
    assert equation.schiller_naumann(torch.tensor(0.01)).item() == pytest.approx(24 / 0.01, rel=0.03)  # Stokes
    assert equation.schiller_naumann(torch.tensor(100.0)).item() == pytest.approx(1.09, rel=0.02)


@pytest.fixture(scope="module")
def sphere_run():
    return solve.sphere_case(**solve.SPHERE_CASE, device=CPU)


@pytest.mark.slow
def test_sphere_drag_matches_schiller_naumann(sphere_run):
    """Re = 20, D = 10 nodes in an 80 × 40 × 40 box periodic across the flow. The free-sphere correlation gives
    C_D = 2.61; this confined staircase sphere drags more. OpenLB on the same lattice gives 3.38 (+29 %), OpenLB with
    D = 20 in the same proportions 3.21, and OpenLB in a 160 × 60 × 60 box with 6 D of upstream length 3.04 (+16 %):
    the excess comes from the small box (4 D wide, inlet 2 D upstream) and the stair-step surface, not from the tensor
    code. Honest tolerance: between the correlation and +35 %; the 15 % hoped for needs a much bigger box."""
    cd, re = sphere_run["cd_mean"], solve.SPHERE_CASE["re"]
    expected = equation.schiller_naumann(torch.tensor(re)).item()
    assert expected < cd < 1.35 * expected


@pytest.mark.slow
def test_sphere_matches_openlb_reference(sphere_run):
    """Same lattice, Re, BGK collision and staircase sphere, run once with OpenLB 1.7 (offline). Boundary schemes at the
    inlet and outlet differ (OpenLB: interpolated velocity / pressure; here: equilibrium inlet / copy outlet).
    Measured residuals: C_D −0.5 %, mid-plane u_x L2 0.8 %, L∞ 1.1 %, max |Δu| 1.3 % of U."""
    res = solve.openlb_residuals(sphere_run["cd"], sphere_run["midplane_ux"], DATA)
    assert abs(res["cd_rel"]) < 0.03
    assert res["l2"] < 0.03
    assert res["max_du"] < 0.05


def test_surface_nets_sphere_is_closed_and_outward():
    n = 24
    x = torch.arange(n, dtype=F64)
    gx, gy, gz = torch.meshgrid(x, x, x, indexing="ij")
    r = ((gx - 11.7) ** 2 + (gy - 12.2) ** 2 + (gz - 11.5) ** 2).sqrt()
    verts, faces = tensors.surface_nets(-r, -7.0)  # inside: r < 7
    tri = verts[faces]
    assert tensors.edge_use_counts(tri).eq(2).all()  # a closed surface
    assert (verts - torch.tensor([11.7, 12.2, 11.5], dtype=F64)).norm(dim=1).sub(7).abs().max() < 0.2
    assert tensors.mesh_volume(tri).item() == pytest.approx(4 / 3 * math.pi * 7**3, rel=0.05)  # outward normals


def test_vtk_files_parse_back(tmp_path):
    pts = torch.rand(10, 3, dtype=F64).numpy()
    path = plot.write_vtp(
        tmp_path / "a.vtp", pts, [[0, 1, 2], [2, 3, 4]], "Polys", {"Cp": (pts[:, 0], "dimensionless")}
    )
    back = plot.read_vtp(path)
    assert (back["Points"], back["Polys"], back["Lines"], back["Verts"]) == (10, 2, 0, 0)
    assert back["offsets"].tolist() == [3, 6] and back["connectivity"].tolist() == [0, 1, 2, 2, 3, 4]
    assert abs(back["arrays"]["Cp"] - pts[:, 0]).max() < 1e-6 and abs(back["points"] - pts).max() < 1e-6
    pvd = plot.write_pvd(tmp_path / "c.pvd", [(0.0, "surface", "a.vtp"), (1.0, "surface", "a.vtp")])
    assert pvd.read_text().count("<DataSet") == 2
