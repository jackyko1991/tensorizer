"""Physics checks, written first (TDD): what any correct voxel CFD + centreline pipeline must satisfy.

Run with `pytest problems/medical/aneurysm_cfd/test.py`. Analytic laws (Hagen–Poiseuille, Womersley, incompressibility,
mass conservation, exact volumes and radii) and two precomputed references from established tools, bundled as CSV:
OpenFOAM's pimpleFoam on the same pulsatile pipe and VMTK's centrelines of the same aneurysm surface. Neither tool is
ever run by the tests. A third reference: pimpleFoam on our own voxel mesh of the real (Lausanne) aneurysm.
"""

import math

import pytest
import torch

from . import equation, plot, solve, tensors

R_PIPE, NU, RHO = 2e-3, tensors.NU, tensors.RHO
MU = RHO * NU


# ---------------------------------------------------------------- geometry: STL reader, voxeliser, radii
def test_voxelised_cylinder_has_the_right_volume():
    R, L, h = 2e-3, 10e-3, 0.2e-3
    tris, ids = tensors.cylinder_mesh(R, L, n_theta=128)
    assert tensors.mesh_volume(tris) == pytest.approx(math.pi * R**2 * L, rel=2e-3)  # polygon vs circle
    lo = torch.tensor([-R - 2 * h, -R - 2 * h, -2 * h], dtype=torch.float64) + 1e-9
    shape = (int(2 * R / h) + 5, int(2 * R / h) + 5, int(L / h) + 5)
    inside = tensors.voxelize(tris, lo, h, shape)
    assert inside.sum().item() * h**3 == pytest.approx(math.pi * R**2 * L, rel=0.03)  # voxel count × h³


@pytest.mark.slow
@pytest.mark.parametrize("geometry, n_caps", [("synthetic", 3), ("lausanne", 4)])
def test_bundled_stl_is_watertight_and_voxelises_to_its_volume(geometry, n_caps):
    tris, ids = tensors.surface(geometry)
    assert set(ids.unique().tolist()) == set(range(n_caps + 1))  # wall, inlet, outlets
    V = tensors.mesh_volume(tris)
    assert 4e-7 < V < 2e-6  # several hundred mm³: a few-mm artery with its aneurysm
    grid = tensors.lumen_grid(tris, ids, 0.3e-3)
    assert grid["inside"].sum().item() * 0.3e-3**3 == pytest.approx(V, rel=0.03)


def test_wall_distance_is_the_pipe_radius_on_the_axis():
    tris, ids = tensors.cylinder_mesh(R_PIPE, 10e-3, n_theta=96, n_z=40)
    V, N = tensors.vertices_and_normals(tris[ids == 0])
    x = torch.tensor([[0.0, 0.0, 5e-3], [1e-3, 0.0, 5e-3], [0.0, -1.5e-3, 3e-3]], dtype=torch.float64)
    d = tensors.wall_distance(x, V, N)
    assert d.tolist() == pytest.approx([2e-3, 1e-3, 0.5e-3], abs=2e-6)  # R − r


# ---------------------------------------------------------------- flow: projection and conservation
@pytest.fixture(scope="module", params=["lausanne", "synthetic"])
def coarse_aneurysm(request):
    tris, ids = tensors.surface(request.param)
    grid = tensors.lumen_grid(tris, ids, 0.5e-3)
    return grid, solve.build_mesh(grid)


def test_projection_makes_the_flow_divergence_free(coarse_aneurysm):
    grid, mesh = coarse_aneurysm
    u = torch.randn(len(mesh["owner"]), generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    u[mesh["fixed"]] = 0.0
    div0 = solve.eq.divergence(u, mesh["owner"], mesh["neigh"], mesh["n"], mesh["h"])[mesh["interior"]]
    u2, _, _ = solve.project(u, mesh, 1e-3, torch.zeros(mesh["n"], dtype=torch.float64), tol=1e-10)
    div = solve.eq.divergence(u2, mesh["owner"], mesh["neigh"], mesh["n"], mesh["h"])[mesh["interior"]]
    assert div.abs().max() < 1e-8 * div0.abs().max()  # ∇·u = 0 in every interior cell after the projection


def test_inflow_equals_total_outflow(coarse_aneurysm):
    grid, mesh = coarse_aneurysm
    t, Q = tensors.inflow_waveform()
    inl = solve.inlet_setup(mesh, solve.fourier(Q, 12), 2 * math.pi, NU)
    u = torch.zeros(len(mesh["owner"]), dtype=torch.float64)
    p = torch.zeros(mesh["n"], dtype=torch.float64)
    for k in range(5):
        u, p, _ = solve.step(u, p, mesh, 1e-3, NU, k * 1e-3, inl=inl, tol=1e-11)
    q_in, q_out = solve.flows(u, mesh)
    Q5 = solve.inlet_values(inl, 5e-3)  # prescribed faces at t = 5 ms
    assert q_in == pytest.approx((Q5 * mesh["s_in"][inl["faces"]]).sum().item() * mesh["h"] ** 2, rel=1e-12)
    assert q_in == pytest.approx(torch.quantile(Q, 0.0).item(), rel=0.5)  # early diastole-to-systole range
    assert q_out.sum().item() == pytest.approx(q_in, rel=1e-6)  # mass conservation: in = Σ out
    assert (q_out > 0).all()  # every outlet carries flow


def test_womersley_profile_carries_the_prescribed_flow():
    t, Q = tensors.inflow_waveform()
    Qhat = solve.fourier(Q, 12)
    r = torch.linspace(0, R_PIPE, 2001, dtype=torch.float64)
    tt = torch.linspace(0, 1, 9, dtype=torch.float64)[:-1]
    u = equation.womersley_flow(r, tt, Qhat, 2 * math.pi, R_PIPE, NU)  # (T, M)
    flow = torch.trapezoid(2 * math.pi * r * u, r, dim=1)
    Qt = (Qhat[None, :] * torch.exp(1j * 2 * math.pi * tt[:, None] * torch.arange(len(Qhat)))).sum(1).real
    assert torch.allclose(flow, Qt, rtol=1e-5)  # ∫ 2πr u dr = Q(t) at every instant
    assert Q.mean().item() == pytest.approx(4.58e-6, rel=1e-3)  # 275 ml/min (Ford et al. 2005)


# ---------------------------------------------------------------- pipe flow: Poiseuille, Womersley, OpenFOAM
@pytest.fixture(scope="module")
def pipe():
    grid = tensors.pipe_grid(R_PIPE, R_PIPE / 8)
    return grid, solve.build_mesh(grid, periodic=(False, False, True))


def test_poiseuille_profile_and_wall_shear(pipe):
    grid, mesh = pipe
    G = 1.05  # −∂_z p/ρ in m/s²
    out = solve.pipe_run(grid, mesh, G, 0.0, 1.0, 1.5, 1e-3, NU, RHO, n_rec=5)
    u = solve.cell_velocity(out["u"], mesh)[2][tuple(mesh["cells"].T)]
    r = mesh["pos"][:, :2].norm(dim=1)
    exact = G * (R_PIPE**2 - r**2) / (4 * NU)
    assert (u - exact).abs().max() < 0.03 * exact.max()  # parabolic profile on the coarse grid (8 cells per radius)
    Q = (out["u"] * (mesh["dirn"] == 2)).sum().item() * mesh["h"] ** 2 / mesh["shape"][2]
    assert Q == pytest.approx(math.pi * G * R_PIPE**4 / (8 * NU), rel=0.04)  # Hagen–Poiseuille Q = πGR⁴/8ν
    tau = out["wss"][-1].item()
    assert tau == pytest.approx(equation.poiseuille_wss(Q, R_PIPE, MU), rel=0.05)  # τ_w = 4μQ/(πR³) within 5 %


@pytest.fixture(scope="module")
def womersley_run(pipe):
    grid, mesh = pipe
    return solve.pipe_run(grid, mesh, 1.05, 2.1, 2 * math.pi, 3.0, 1e-3, NU, RHO, n_rec=300)


@pytest.mark.slow
def test_womersley_amplitude_and_phase(womersley_run):
    out = womersley_run
    last = out["t"] > 2.0 + 1e-9
    t = out["t"][last]
    G = [complex(1.05), complex(0.0, -2.1)]  # G1 sin ωt = Re(−i G1 e^{iωt})
    exact = equation.womersley_pressure(torch.zeros(1, dtype=torch.float64), t, G, 2 * math.pi, R_PIPE, NU)[:, 0]
    sim = out["u_axis"][last]
    c_sim = complex((sim * torch.exp(-1j * 2 * math.pi * t)).mean().item()) * 2  # first Fourier harmonic
    c_ex = complex((exact * torch.exp(-1j * 2 * math.pi * t)).mean().item()) * 2
    assert abs(c_sim) == pytest.approx(abs(c_ex), rel=0.03)  # amplitude
    assert (
        abs(math.remainder(math.atan2(c_sim.imag, c_sim.real) - math.atan2(c_ex.imag, c_ex.real), 2 * math.pi)) < 0.03
    )
    assert sim.mean().item() == pytest.approx(exact.mean().item(), rel=0.03)  # steady (Poiseuille) part


@pytest.mark.slow
def test_matches_the_openfoam_reference(womersley_run):
    res = solve.openfoam_residuals(womersley_run, tensors.openfoam_reference(), 2.0, 3.0, RHO)  # last cycle
    assert res["u_axis_L2"] < 0.03 and res["u_axis_Linf"] < 0.05  # axis velocity: same physics, two codes
    assert abs(res["amp_err"]) < 0.02 and abs(res["phase_err"]) < 0.03  # first harmonic: amplitude, phase (rad)
    assert res["Q_L2"] < 0.03  # flow rate
    assert res["wss_L2"] < 0.08  # wall shear stress (OpenFOAM: one-sided ν u/d at the wall cells)


# ---------------------------------------------------------------- centreline (tensor version of vmtkcenterlines)


def test_centreline_of_a_straight_cylinder():
    h, R = 0.25e-3, 1.5e-3
    grid = tensors.tube_mask([[[0, 0, 0], [0, 0, 12e-3]]], [R], h)
    v = lambda *x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    caps = [
        {"centre": v(0, 0, 0), "normal": v(0, 0, -1), "radius": R},
        {"centre": v(0, 0, 12e-3), "normal": v(0, 0, 1), "radius": R},
    ]
    lines, _, _ = solve.centrelines(
        grid, solve.cap_nodes(grid, caps[0]), torch.tensor([solve.outlet_seed(grid, caps[1])])
    )
    x, Rl = lines[0]["x"], lines[0]["R"]
    assert x[:, :2].norm(dim=1).max() < 0.5 * h  # on the axis
    assert (Rl - R).abs().median() < h  # inscribed radius = pipe radius within a voxel
    assert lines[0]["s"][-1].item() == pytest.approx(12e-3, abs=3 * h)  # abscissa = length


def test_centreline_of_a_curved_tube_follows_the_arc():
    h, R, Rc = 0.25e-3, 1.2e-3, 8e-3
    th = torch.linspace(0, math.pi / 2, 31, dtype=torch.float64)
    arc = torch.stack([Rc * torch.cos(th), Rc * torch.sin(th), torch.zeros_like(th)], -1)
    grid = tensors.tube_mask([arc.tolist()], [R], h)
    c0 = {"centre": arc[0], "normal": torch.tensor([0, -1.0, 0], dtype=torch.float64), "radius": R}
    c1 = {"centre": arc[-1], "normal": torch.tensor([-1.0, 0, 0], dtype=torch.float64), "radius": R}
    lines, _, _ = solve.centrelines(grid, solve.cap_nodes(grid, c0), torch.tensor([solve.outlet_seed(grid, c1)]))
    x = lines[0]["x"]
    assert ((x[:, :2].norm(dim=1) - Rc).abs().max()) < 0.75 * h  # stays on the circle of radius Rc
    assert lines[0]["s"][-1].item() == pytest.approx(
        math.pi / 2 * Rc, abs=4 * h
    )  # arc length (ends sit in the cap slabs)
    assert (lines[0]["R"] - R).abs().median() < h


def test_y_bifurcation_gives_two_centrelines_merging_at_the_junction():
    h = 0.25e-3
    J = [0.0, 0.0, 8e-3]
    a1 = [6e-3 * math.sin(math.radians(45)), 0.0, 8e-3 + 6e-3 * math.cos(math.radians(45))]
    a2 = [-6e-3 * math.sin(math.radians(45)), 0.0, 8e-3 + 6e-3 * math.cos(math.radians(45))]
    grid = tensors.tube_mask([[[0, 0, 0], J], [J, a1], [J, a2]], [1.5e-3, 1.1e-3, 1.1e-3], h)
    f = lambda c, n, r: {  # noqa: E731
        "centre": torch.tensor(c, dtype=torch.float64),
        "normal": torch.tensor(n, dtype=torch.float64) / torch.tensor(n).norm(),
        "radius": r,
    }  # noqa: E731
    c0 = f([0, 0, 0], [0, 0, -1.0], 1.5e-3)
    d1 = [a1[0] - J[0], 0, a1[2] - J[2]]
    d2 = [a2[0] - J[0], 0, a2[2] - J[2]]
    seeds = torch.tensor([solve.outlet_seed(grid, f(a1, d1, 1.1e-3)), solve.outlet_seed(grid, f(a2, d2, 1.1e-3))])
    lines, bif, _ = solve.centrelines(grid, solve.cap_nodes(grid, c0), seeds)
    assert len(lines) == 2
    assert (bif["x"] - torch.tensor(J, dtype=torch.float64)).norm() < 1.5e-3  # merge within a parent radius of J
    for line, r in zip(lines, (1.1e-3, 1.1e-3)):
        tail = line["s"] > line["s"][-1] - 4e-3  # last 4 mm: inside the daughter branch
        assert (line["R"][tail] - r).abs().median() < h
    assert (lines[0]["R"][lines[0]["s"] < 5e-3] - 1.5e-3).abs().median() < h  # parent radius


@pytest.mark.slow
@pytest.mark.parametrize("geometry, h", [("synthetic", 0.3e-3), ("lausanne", 0.4e-3)])
def test_centreline_matches_the_vmtk_reference(geometry, h):
    tris, ids = tensors.surface(geometry)
    grid = tensors.lumen_grid(tris, ids, h)
    caps = grid["caps"]
    seeds = torch.tensor([solve.outlet_seed(grid, c) for c in caps[1:]])
    lines, bif, _ = solve.centrelines(grid, solve.cap_nodes(grid, caps[0]), seeds)
    for r in solve.vmtk_residuals(lines, *tensors.vmtk_reference(geometry)):
        assert r["d"].mean() < 0.75 * h and r["d"].max() < 2.5 * h  # distance VMTK → ours (max: at the end caps)
        assert r["dR"].median() < 0.5 * h and r["dR"].max() < h  # inscribed-sphere radius


# ---------------------------------------------------------------- the real aneurysm against OpenFOAM on the same mesh
@pytest.mark.slow
def test_aneurysm_flow_matches_openfoam_on_the_same_mesh():
    from .main import aneurysm_residuals

    ref = tensors.openfoam_aneurysm_reference()
    tris, ids = tensors.surface("lausanne")
    grid = tensors.lumen_grid(tris, ids, 0.4e-3)
    mesh = solve.build_mesh(grid)
    t, Q = tensors.inflow_waveform()
    Q = Q * (grid["caps"][0]["radius"] / tensors.R_ICA) ** 3
    probe = {"times": ref["t"], "points": ref["points"]}
    rec = solve.simulate(mesh, (t, Q), 1.5, 2e-3, NU, RHO, grid["V"], grid["N"], n_frames=2, n_wss=2, n_particles=10,
                         n_stream=2, probe=probe)  # fmt: skip
    res = aneurysm_residuals(rec, ref)
    # Same mesh, inflow and outlets, two different codes (semi-Lagrangian + ghost-cell wall vs PISO + staircase wall):
    # agreement to about a third, see the notes. Tolerances guard against regressions, not a claim of accuracy.
    assert res["u_L2"] < 0.45 and res["tawss_L2"] < 0.45 and res["wss_L2"] < 0.5
    assert abs(rec["probe_wss"].mean().item() / ref["tawss"].mean().item() - 1) < 0.15  # wall-averaged TAWSS


# ---------------------------------------------------------------- ParaView output: written and parsed back
def test_vtp_files_round_trip(tmp_path):
    P = torch.rand(30, 3, generator=torch.Generator().manual_seed(0)).numpy()
    tri = [[0, 1, 2], [2, 3, 4], [5, 6, 7]]
    f = plot.write_vtp(tmp_path / "s.vtp", P, polys=tri, point_data={"WSS": (P[:, 0], "Pa")})
    r = plot.read_vtp(f)
    assert (r["NumberOfPoints"], r["NumberOfPolys"]) == (30, 3)
    assert r["Polys/connectivity"].ravel().tolist() == sum(tri, [])
    assert abs(r["PointData/WSS"].ravel() - P[:, 0]).max() < 1e-6
    lines = [list(range(0, 10)), list(range(10, 30))]
    r = plot.read_vtp(plot.write_vtp(tmp_path / "l.vtp", P, lines=lines, point_data={"speed": (P[:, 1], "m/s")}))
    assert (r["NumberOfLines"], r["Lines/offsets"].ravel().tolist()) == (2, [10, 30])
    r = plot.read_vtp(plot.write_vtp(tmp_path / "p.vtp", P, verts=True, point_data={"age": (P[:, 2], "s")}))
    assert r["NumberOfVerts"] == 30 and r["Points/Points"].shape == (30, 3)
