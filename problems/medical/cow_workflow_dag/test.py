"""Checks, written first (TDD): the tensor DAG engine must be the same workflow as Snakemake, and every pipeline stage
must obey its law.

Run with `pytest problems/medical/cow_workflow_dag/test.py`. Snakemake itself never runs here: its outputs (the job
DAG, the dry-run job lists, the rerun set after touching one input, per-job times and the cohort table its jobs wrote)
were made once, offline, from data/Snakefile and are bundled in data/snakemake/.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from problems.medical.aneurysm_cfd.plot import read_vtp

from . import equation as eq
from . import plot, solve
from . import tensors as T


@pytest.fixture(scope="module")
def dag():
    return solve.build_dag(T.SNAKEFILE.read_text(encoding="utf-8"))


def job_index(dag) -> dict:
    return {j: i for i, j in enumerate(dag["jobs"])}


# ---------------------------------------------------------------- the DAG algebra
def test_snakefile_parses_to_the_rule_graph(dag):
    names = [r["name"] for r in dag["rules"]]
    assert names == ["all", "segment", "centreline", "measure", "boundaries", "flow", "report", "cohort"]
    edges = {(names[s], names[r]) for s, r in dag["A_rule"].nonzero().tolist()}
    assert edges == {("measure", "segment"), ("measure", "centreline"), ("boundaries", "centreline"),
                     ("flow", "measure"), ("flow", "boundaries"), ("report", "flow"), ("report", "measure"),
                     ("cohort", "report"), ("all", "cohort")}  # fmt: skip
    axes = dict(zip(names, dag["axes"]))
    assert axes["segment"] == ("case",) and axes["flow"] == ("case", "condition") and axes["cohort"] == ()


def test_job_dag_is_a_sum_of_kronecker_products(dag):
    A, off, names = dag["A"], dag["off"], [r["name"] for r in dag["rules"]]
    n, c = len(T.CASES), len(T.CONDITIONS)
    blk = lambda s, r: A[off[names.index(s)] : off[names.index(s) + 1], off[names.index(r)] : off[names.index(r) + 1]]  # noqa: E731
    assert torch.equal(blk("measure", "segment"), torch.eye(n))  # per-case edges: A ⊗ I
    assert torch.equal(blk("flow", "measure"), torch.kron(torch.eye(n), torch.ones(c, 1)))  # broadcast I ⊗ 𝟙
    assert torch.equal(blk("report", "flow"), torch.eye(n * c))
    assert torch.equal(blk("cohort", "report"), torch.ones(1, n * c))  # gather 𝟙ᵀ ⊗ 𝟙ᵀ
    assert len(dag["jobs"]) == 4 * n + 2 * n * c + 2 == 50 and int(A.sum()) == 79


def test_acyclic_by_nilpotency(dag):
    A = dag["A"]
    assert eq.is_nilpotent(A) and torch.all(eq.walk_traces(A) == 0)  # no closed walk of any length
    cyc = torch.tensor([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])  # a 3-cycle
    assert not eq.is_nilpotent(cyc) and eq.walk_traces(cyc)[2] == 3
    with pytest.raises(ValueError):
        eq.levels(cyc)


def test_boolean_closure_is_the_inverse_of_one_minus_A(dag):
    A = dag["A"].double()
    paths = torch.linalg.inv(torch.eye(len(A), dtype=torch.float64) - A)  # = I + A + A² + … (nilpotent: finite)
    assert torch.allclose(paths, paths.round(), atol=1e-9)  # integer path counts
    R = eq.closure(A)
    assert torch.equal(R | torch.eye(len(A), dtype=torch.bool), paths.round() > 0)
    i = job_index(dag)
    # segment → measure → (flow →) report → cohort → all, for each of the 2 conditions: 4 distinct paths
    assert paths[i[("all", "", "")], i[("segment", "sub-000", "")]].round() == 4


def test_levels_and_critical_path_by_maxplus(dag):
    A = dag["A"]
    lev = eq.levels(A)
    lev_mp = eq.finish_times(A, torch.ones(len(A), dtype=torch.float64)) - 1  # max-plus with unit durations
    assert torch.equal(lev, lev_mp.long())
    d = torch.rand(len(A), generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    t = eq.finish_times(A, d)
    ref = torch.zeros_like(d)  # slow reference: dynamic programming in a topological order
    for i in torch.argsort(lev).tolist():
        deps = A[i].nonzero().flatten()
        ref[i] = d[i] + (ref[deps].max() if len(deps) else 0.0)
    assert torch.allclose(t, ref)
    path = eq.critical_path(A, t, d)
    assert d[path].sum() == pytest.approx(float(t.max())) and all(A[b, a] for a, b in zip(path, path[1:]))


# ---------------------------------------------------------------- equivalence with the bundled Snakemake run
def test_same_jobs_and_edges_as_snakemake(dag):
    jobs, dot, files = T.ref_dag()
    assert set(dag["jobs"]) == jobs
    ours = {(dag["jobs"][i], dag["jobs"][j]) for i, j in dag["A"].nonzero().tolist()}
    assert ours == dot == files  # `snakemake --dag`, and the dry run's own input → output links


def test_snakemake_orders_are_valid_and_have_our_levels(dag):
    i, A = job_index(dag), dag["A"]
    lev = eq.levels(A)
    for name in ("timings_cores1.csv", "timings_cores12.csv"):
        rows = T.ref_timings(int(name.split("cores")[1].split(".")[0]))
        start = {(r["rule"], r["case"], r["condition"]): r["start"] for r in rows}
        end = {(r["rule"], r["case"], r["condition"]): r["end"] for r in rows}
        for a, b in A.nonzero().tolist():  # every job started after all of its inputs ended
            if dag["jobs"][a] in start:  # (the target rule 'all' runs no shell)
                assert start[dag["jobs"][a]] >= end[dag["jobs"][b]]
    order = [i[j] for j in T.ref_jobs("dryrun_jobs.csv")]
    assert sorted(order) == list(range(len(A)))  # the dry run lists every job once…
    seen = set()
    for k in order:  # …and the order respects the dependencies
        assert set(A[k].nonzero().flatten().tolist()) <= seen
        seen.add(k)
    _, edges, _ = T.ref_dag()
    depth = {j: 0 for j in dag["jobs"]}  # levels from Snakemake's own edges (longest chain of inputs)
    for _ in range(len(A)):
        for a, b in edges:
            depth[a] = max(depth[a], depth[b] + 1)
    assert all(depth[j] == int(lev[k]) for j, k in i.items())


def test_rerun_set_after_touching_one_input_matches_snakemake(dag):
    i = job_index(dag)
    touched = torch.zeros(len(dag["jobs"]), dtype=torch.bool)
    for rule in ("segment", "centreline"):  # the jobs that read cases/sub-002.npz
        touched[i[(rule, "sub-002", "")]] = True
    x = eq.rerun_set(eq.closure(dag["A"]), touched)
    assert {dag["jobs"][k] for k in x.nonzero().flatten().tolist()} == set(T.ref_jobs("rerun_after_touch.csv"))
    assert int(x.sum()) == 4 + 2 * 2 + 2  # 4 case jobs, flow and report for both conditions, cohort, all


def test_measured_makespans_respect_the_scheduling_bounds(dag):
    i, A = job_index(dag), dag["A"]
    for cores in (1, 6, 12):
        rows = T.ref_timings(cores)
        d = torch.zeros(len(A), dtype=torch.float64)
        for r in rows:
            d[i[(r["rule"], r["case"], r["condition"])]] = r["end"] - r["start"]
        cp = float(eq.finish_times(A, d).max())
        span = max(r["end"] for r in rows) - min(r["start"] for r in rows)
        assert span >= eq.lower_bound(d, cp, cores) * (1 - 1e-6)  # no schedule beats max(CP, W/c)
        assert span <= T.ref_runs()[cores]  # the whole run (with Snakemake's own start-up) took longer


# ---------------------------------------------------------------- pipeline stages on synthetic vessels
def tube(lab, p0, p1, r, label, h=T.H):
    """Label the voxels within r (m) of the segment p0 → p1 (voxel coordinates)."""
    X, Y, Z = lab.shape
    g = torch.stack(torch.meshgrid(torch.arange(X), torch.arange(Y), torch.arange(Z), indexing="ij"), -1).double()
    a, b = torch.tensor(p0, dtype=torch.float64), torch.tensor(p1, dtype=torch.float64)
    s = (((g - a) @ (b - a)) / ((b - a) @ (b - a))).clamp(0, 1)
    d = (g - (a + s[..., None] * (b - a))).norm(dim=-1) * h
    lab[(d <= r) & (lab == 0)] = label


@pytest.fixture(scope="module")
def y_vessel():
    """A Y bifurcation: an ICA (label 4) splitting into an MCA (5) and an ACA (11), radii 1.5, 1.2, 1.0 mm."""
    lab = torch.zeros(1, 60, 40, 40, dtype=torch.long)
    tube(lab[0], (30, 20, 2), (30, 20, 20), 1.5e-3, 4)
    tube(lab[0], (30, 20, 20), (56, 20, 34), 1.2e-3, 5)
    tube(lab[0], (30, 20, 20), (6, 20, 34), 1.0e-3, 11)
    return lab


def test_centreline_graph_of_a_bifurcation(y_vessel):
    g = solve.centreline(y_vessel)
    kind = g["node_kind"][0][g["node_valid"][0]]
    assert sorted(kind.tolist()) == [0, 4, 5, 11]  # one junction, three free ends
    assert int(g["edge_valid"].sum()) == 3
    b = solve.boundaries(g)
    assert int(b["inlet"].sum()) == 1 and int(b["outlet"].sum()) == 2
    L = dict(zip(g["edge_label"][0].tolist(), g["length"][0].tolist()))
    assert L[4] == pytest.approx(20.5 * T.H, abs=1.5 * T.H)  # straight ICA: from its rounded free end to the junction
    poly = g["poly"][0][g["edge_label"][0] == 4][0]
    poly = poly[~torch.isnan(poly[:, 0])]
    off_axis = (poly[:, [0, 1]] - torch.tensor([30.0, 20.0], dtype=torch.float64) * T.H).norm(dim=1)
    assert off_axis.max() < 0.6 * T.H  # on the axis


def test_radius_from_the_soft_segmentation(y_vessel):
    image = (y_vessel > 0).double()
    seg = solve.segment(image, y_vessel, torch.tensor([0.5], dtype=torch.float64))
    assert float(seg["dice"]) == pytest.approx(1.0)
    g = solve.centreline(y_vessel)
    m = solve.measure(seg, g)
    r = dict(zip(g["edge_label"][0].tolist(), m["radius"][0].tolist()))
    for label, r_true in ((4, 1.5e-3), (5, 1.2e-3), (11, 1.0e-3)):
        assert r[label] == pytest.approx(r_true, rel=0.15)  # voxelised tube; the junction bulb adds a little
    assert not bool(m["stenosis"].any())


def test_connected_components_drop_specks():
    img = torch.zeros(1, 20, 20, 20, dtype=torch.float64)
    img[0, 2:12, 5:9, 5:9] = 1.0  # 160 voxels: kept
    img[0, 15:17, 15:17, 15:17] = 1.0  # 8 voxels: noise
    seg = solve.segment(img, torch.zeros_like(img, dtype=torch.long), torch.tensor([0.5], dtype=torch.float64))
    assert int(seg["mask"].sum()) == 160


# ---------------------------------------------------------------- the flow network
def network(nodes: list, edges: list, radii: list, lengths: list):
    """A hand-made graph in the stage formats: node kinds, edges (u, v) and their radius and length (m)."""
    kind = torch.tensor([nodes])
    E = torch.tensor([edges])
    g = {
        "node_kind": kind,
        "node_valid": kind >= 0,
        "edges": E,
        "edge_valid": torch.ones(1, len(edges), dtype=torch.bool),
    }
    m = {"radius": torch.tensor([radii], dtype=torch.float64), "length": torch.tensor([lengths], dtype=torch.float64)}
    bedge = torch.full_like(kind, -1)
    for e, (u, _) in enumerate(edges):
        if nodes[u] > 0:
            bedge[0, u] = e
    b = {"inlet": torch.isin(kind, torch.tensor(T.INLETS)), "outlet": torch.isin(kind, torch.tensor(T.OUTLETS)),
         "edge": bedge}  # fmt: skip
    return g, m, b


def test_series_vessels_follow_poiseuille():
    r, L = 1.2e-3, 20e-3  # inlet (ICA end) → junction → outlet (MCA end, its free end is the edge start)
    g, m, b = network([4, 0, 5], [(0, 1), (2, 1)], [r, r], [L, L])
    fl = solve.flow(m, b, g, torch.tensor([False]))
    R_vessel = 8 * T.MU * L / (math.pi * r**4)
    Q = T.P_IN / (2 * R_vessel + T.P_IN / T.Q_TOT)  # two vessels and the outlet load, in series
    assert float(fl["Q_out"].sum()) == pytest.approx(Q, rel=1e-9)
    assert float(fl["Q_in"].sum()) == pytest.approx(Q, rel=1e-6)


def test_murray_split_when_the_network_has_no_resistance():
    radii = [5e-3, 1.0e-3, 1.5e-3, 0.8e-3]  # a wide inlet vessel and three outlet vessels, all very short
    g, m, b = network([4, 0, 5, 11, 2], [(0, 1), (2, 1), (3, 1), (4, 1)], radii, [1e-6] * 4)
    fl = solve.flow(m, b, g, torch.tensor([False]))
    share = fl["Q_out"][0, 2:] / fl["Q_out"].sum()
    r3 = torch.tensor(radii[1:], dtype=torch.float64) ** 3
    assert torch.allclose(share, r3 / r3.sum(), rtol=1e-4)  # Murray: Q_k ∝ r_k³
    assert float(fl["Q_out"].sum()) == pytest.approx(T.Q_TOT, rel=1e-4)


@pytest.mark.parametrize("acom_radius, fed", [(0.8e-3, True), (1e-6, False)])
def test_collateral_flow_through_the_communicating_artery(acom_radius, fed):
    # right ICA → junction 1 → right MCA;  left ICA → junction 2 → left MCA;  Acom between the junctions
    g, m, b = network([4, 0, 6, 0, 5, 7], [(0, 1), (2, 3), (4, 1), (5, 3), (1, 3)],
                      [2e-3, 2e-3, 1.3e-3, 1.3e-3, acom_radius], [15e-3, 15e-3, 15e-3, 15e-3, 3e-3])  # fmt: skip
    base = solve.flow(m, b, g, torch.tensor([False]))
    occl = solve.flow(m, b, g, torch.tensor([True]))
    assert float(occl["Q_in"][0, 2]) == 0.0  # the left ICA is closed
    left = float(occl["Q_out"][0, 5] / base["Q_out"][0, 5])
    assert (left > 0.8) if fed else (left < 1e-3)  # a patent Acom keeps the left MCA perfused


def test_stenosis_sweep_runs_from_the_open_to_the_closed_condition():
    g, m, b = network([4, 0, 6, 0, 5, 7], [(0, 1), (2, 3), (4, 1), (5, 3), (1, 3)],
                      [2e-3, 2e-3, 1.3e-3, 1.3e-3, 0.8e-3], [15e-3, 15e-3, 15e-3, 15e-3, 3e-3])  # fmt: skip
    sev, sweep = solve.stenosis_sweep(m, b, g, steps=11)
    base = solve.flow(m, b, g, torch.tensor([False]))
    occl = solve.flow(m, b, g, torch.tensor([True]))
    assert torch.equal(sweep[0]["p"], base["p"]) and torch.equal(sweep[-1]["p"], occl["p"])  # the two conditions
    q_lica = torch.stack([fs["Q_in"][0, 2] for fs in sweep])
    assert (q_lica.diff() < 0).all() and float(q_lica[-1]) == 0.0  # narrowing lowers the left-ICA inflow to 0
    q_out = torch.stack([fs["Q_out"].sum() for fs in sweep])
    assert torch.allclose(q_out, torch.stack([fs["Q_in"].sum() for fs in sweep]), rtol=1e-9)  # mass balance


# ---------------------------------------------------------------- the real cohort (one batched engine run)
@pytest.fixture(scope="module")
def engine(dag):
    store, times = solve.run_engine(dag, grad=True)
    return store


@pytest.mark.slow
def test_engine_reproduces_the_cohort_table_snakemake_wrote(engine):
    ref = T.ref_cohort()
    ours = list(__import__("csv").DictReader(engine["cohort"].splitlines()))
    assert [(r["case"], r["condition"]) for r in ours] == [(r["case"], r["condition"]) for r in ref]
    for a, b in zip(ours, ref):
        for k in solve.COLUMNS[2:]:
            x, y = float(a[k]), float(b[k])
            assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, rel=1e-6, abs=1e-9), k


@pytest.mark.slow
def test_cohort_physics(engine):
    seg, f = engine["segment"], engine["flow"]["flow"]
    assert bool((seg["dice"] > 0.8).all())  # threshold segmentation against the TopCoW masks
    b = engine["boundaries"]
    assert bool((b["inlet"].sum(1) == 3).all()) and bool((b["outlet"].sum(1) >= 6).all())
    G = engine["flow"]["g"]
    u, v = G["edges"][..., 0], G["edges"][..., 1]
    net = torch.zeros_like(f["p"]).scatter_add(1, u, f["q"]).scatter_add(1, v, -f["q"])
    interior = ~(engine["flow"]["b"]["inlet"] | engine["flow"]["b"]["outlet"])
    assert float((net * interior).abs().max()) < 1e-6 * T.Q_TOT  # Kirchhoff: flow balance at every junction
    assert torch.allclose(f["Q_in"].sum(1), f["Q_out"].sum(1), rtol=1e-6)  # in = out
    assert bool((f["p"] <= T.P_IN * (1 + 1e-9)).all()) and bool((f["p"] >= -1e-6).all())  # maximum principle
    r = engine["report"]
    dp = r["dp_max_mmHg"].reshape(len(T.CASES), 2)  # (case, condition)
    assert bool(((dp[:, 0] > 0) & (dp[:, 0] < 10)).all())  # a few mmHg across a healthy circle
    assert bool((dp[:, 1] >= dp[:, 0]).all() & (dp[:, 1] < 30).all())  # collateral flow costs pressure


@pytest.mark.slow
def test_gradient_wrt_threshold_matches_finite_differences(dag, engine):
    img, lab = engine["image"], engine["labels"]
    occ = torch.zeros(len(T.CASES), dtype=torch.bool)

    def share(theta, keep):
        seg = solve.segment(img, lab, theta, keep=keep)
        m = solve.measure(seg, engine["centreline"])
        fl = solve.flow(m, engine["boundaries"], engine["centreline"], occ)
        return solve.report(m, engine["boundaries"], engine["centreline"], fl)["share_L-MCA"]

    th = torch.full((len(T.CASES),), T.THETA, dtype=torch.float64, requires_grad=True)
    share(th, engine["segment"]["keep"]).sum().backward()  # ONE backward pass: every case's derivative
    d = 2e-3
    fd = (share(th.detach() + d, engine["segment"]["keep"]) - share(th.detach() - d, engine["segment"]["keep"])) / (
        2 * d
    )
    assert torch.allclose(th.grad, fd, rtol=2e-2, atol=1e-4)


# ---------------------------------------------------------------- the 3D scene
def test_iso_surface_of_a_sphere():
    n, R, h = 32, 10.0, 1.0
    c = (n - 1) / 2
    g = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64)] * 3, indexing="ij"), -1)
    field = R - (g - c).norm(dim=-1)  # inside > 0
    V, F_ = solve.iso_surface(field, 0.0, h)
    a, b, cc = V[F_[:, 0]], V[F_[:, 1]], V[F_[:, 2]]
    cross = torch.cross(b - a, cc - a, dim=1)
    area = 0.5 * cross.norm(dim=1).sum()
    vol = (torch.einsum("fi,fi->f", a - c, cross)).sum() / 6  # divergence theorem: outward normals give +volume
    assert float(area) == pytest.approx(4 * math.pi * R**2, rel=0.03)
    assert float(vol) == pytest.approx(4 / 3 * math.pi * R**3, rel=0.03)
    edges = torch.sort(torch.cat([F_[:, [0, 1]], F_[:, [1, 2]], F_[:, [2, 0]]]), dim=1).values
    _, cnt = torch.unique(edges, dim=0, return_counts=True)
    assert bool((cnt == 2).all())  # watertight: every edge shared by exactly two triangles


# ---------------------------------------------------------------- the flow scene: WSS, streamlines, particles
def test_wall_shear_stress_of_a_straight_poiseuille_vessel():
    r, L = 1.2e-3, 20e-3
    g, m, b = network([4, 0, 5], [(0, 1), (2, 1)], [r, r], [L, L])
    fl = solve.flow(m, b, g, torch.tensor([False]))
    Q = T.P_IN / (2 * 8 * T.MU * L / (math.pi * r**4) + T.P_IN / T.Q_TOT)
    tau = eq.wall_shear_stress(fl["q"], m["radius"], T.MU)
    assert torch.allclose(tau, torch.full_like(tau, 4 * T.MU * Q / (math.pi * r**3)), rtol=1e-6)
    U, h = Q / (math.pi * r**2), 1e-6  # the same from the profile: τ = μ |∂u/∂d| at the wall, d = ρ r
    dudd = (eq.poiseuille_speed(1.0, U) - eq.poiseuille_speed(1.0 - h, U)) / (h * r)
    assert float(tau[0, 0]) == pytest.approx(T.MU * abs(dudd), rel=1e-5)


def helix(n=120, R=6.0, pitch=4.0, turns=0.8):
    """A curved test centreline (mm): a helix arc, curvature radius ≈ R."""
    t = np.linspace(0, 2 * np.pi * turns, n)
    return np.stack([R * np.cos(t), R * np.sin(t), pitch * t / (2 * np.pi)], 1)


def dist_to_polyline(x: np.ndarray, pe: np.ndarray) -> np.ndarray:
    a, d = pe[:-1], np.diff(pe, axis=0)
    s = np.clip(np.einsum("psk,sk->ps", x[:, None] - a[None], d) / np.einsum("sk,sk->s", d, d), 0, 1)
    return np.linalg.norm(x[:, None] - (a[None] + s[..., None] * d[None]), axis=2).min(1)


def test_axial_streamlines_have_the_poiseuille_profile():
    pe, r, U = helix(), 1.5, 0.3
    nrm, bn = plot.transport_frame(pe)
    t = np.gradient(pe, axis=0)
    assert np.allclose(np.einsum("ik,ik->i", nrm, t), 0, atol=1e-9) and np.allclose(np.linalg.norm(bn, axis=1), 1)
    X, rho = plot.axial_streamlines(pe, r, rings=((0.0, 1), (0.35, 3), (0.7, 6), (1.0, 8)))
    for line, rh in zip(X, rho):
        assert np.allclose(np.linalg.norm(line - pe, axis=1), rh * r)  # parallel to the centreline at ρ r
    u = eq.poiseuille_speed(rho, U)
    assert u.max() == pytest.approx(2 * U) and rho[u.argmax()] == 0.0  # 2U on the axis
    assert u[rho == 1.0] == pytest.approx(0.0, abs=1e-15)  # no slip at the wall
    assert (np.diff(u[np.argsort(rho)]) <= 0).all()


def test_particles_stay_inside_their_vessel():
    pe, r = helix(), 1.5
    X, _ = plot.axial_streamlines(pe, r)
    for line in X:
        s = np.linspace(0, 400, 97)  # seeds advanced far past the end: they wrap to the start of the edge
        P = plot.along(line, s)
        assert (dist_to_polyline(P, pe) <= r + 1e-9).all()


def test_flow_scene_has_exactly_the_flow_layers(engine, tmp_path):
    B = len(T.CASES)
    surfaces = []
    for k in range(B):  # a tiny stand-in surface per case: a tetrahedron (3 mm) at its first node
        c = engine["centreline"]["node_pos"][k, 0].numpy()
        V = c + 3e-3 * np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        surfaces.append((V, np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])))
    g, bnd = engine["centreline"], engine["boundaries"]
    with torch.no_grad():
        sev, sweep = solve.stenosis_sweep(engine["measure"], bnd, g, steps=3)
    _, flow = plot.write_scene(tmp_path, engine, T.CASES, surfaces, sev, sweep)
    root = ET.parse(flow).getroot()
    ds = root.findall("Collection/DataSet")
    assert {d.get("group") for d in ds} == {"surface", "streamlines", "particles", "inlet", "outlets"}
    assert len({d.get("timestep") for d in ds}) == 3
    surf = read_vtp(tmp_path / f"surface_{T.CASES[0]}_t0.vtp")
    names = [k.split("/")[1] for k in surf if k.startswith("PointData/")]
    assert names == ["WSS", "pressure"]  # WSS first: the default colour
    sl = read_vtp(tmp_path / f"streamlines_{T.CASES[0]}_t0.vtp")
    pa = read_vtp(tmp_path / f"particles_{T.CASES[0]}_t2.vtp")
    assert sl["NumberOfLines"] > 0 and pa["NumberOfVerts"] > 0
    assert (sl["PointData/speed"] >= 0).all() and float(sl["PointData/speed"].max()) < 5.0  # m/s, physiological
