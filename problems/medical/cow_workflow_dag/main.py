"""Main loop: read the Snakefile, build the job DAG as tensors, check it against the bundled Snakemake run (jobs,
edges, levels, order, rerun set, outputs), run the whole cohort as tensors (one batched call per rule), compare the
wall-clock times, differentiate the flow split with respect to the segmentation threshold, then draw and export."""

import csv
import io
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from problems._common.runtime import cli, fast_mode

from . import equation as eq
from . import plot, solve
from . import tensors as T


def equivalence(dag: dict, cohort: str | None) -> list[dict]:
    """The tensor engine against the bundled Snakemake references, one row per check."""
    idx = {j: i for i, j in enumerate(dag["jobs"])}
    A = dag["A"]
    rows = []
    jobs_ref, edges_ref, _ = T.ref_dag()
    ours = {(dag["jobs"][i], dag["jobs"][j]) for i, j in A.nonzero().tolist()}
    rows.append({"check": "jobs", "tensor": len(dag["jobs"]), "snakemake": len(jobs_ref),
                 "ok": set(dag["jobs"]) == jobs_ref})  # fmt: skip
    rows.append({"check": "edges", "tensor": len(ours), "snakemake": len(edges_ref), "ok": ours == edges_ref})
    traces = eq.walk_traces(A)
    rows.append({"check": "acyclic", "tensor": f"max {int(traces.abs().max())}", "snakemake": "a DAG",
                 "ok": eq.is_nilpotent(A)})  # fmt: skip
    lev = eq.levels(A)
    order = [idx[j] for j in T.ref_jobs("dryrun_jobs.csv")]
    pos = {k: n for n, k in enumerate(order)}
    valid = all(pos[i] > pos[j] for i, j in A.nonzero().tolist())
    rows.append({"check": "order, levels", "tensor": f"{int(lev.max()) + 1} levels",
                 "snakemake": f"{len(order)} jobs", "ok": valid})  # fmt: skip
    touched = torch.zeros(len(A), dtype=torch.bool)
    touched[[idx[("segment", "sub-002", "")], idx[("centreline", "sub-002", "")]]] = True
    x = eq.rerun_set(eq.closure(A), touched)
    ref = set(T.ref_jobs("rerun_after_touch.csv"))
    mine = {dag["jobs"][k] for k in x.nonzero().flatten().tolist()}
    rows.append({"check": "rerun set", "tensor": len(mine), "snakemake": len(ref), "ok": mine == ref})
    if cohort is not None:
        a, b = list(csv.DictReader(io.StringIO(cohort))), T.ref_cohort()
        worst = 0.0
        for ra, rb in zip(a, b):
            for k in solve.COLUMNS[2:]:
                x1, x2 = float(ra[k]), float(rb[k])
                if not (math.isnan(x1) and math.isnan(x2)):
                    worst = max(worst, abs(x1 - x2) / max(abs(x2), 1e-9))
        rows.append({"check": "cohort table", "tensor": f"{len(a)} rows",
                     "snakemake": f"Δ {worst:.0e}", "ok": len(a) == len(b) and worst < 1e-6})  # fmt: skip
    return rows


def run(params: dict, out_dir: Path) -> Path:
    torch.manual_seed(0)
    theta = float(params["threshold"])
    cores = int(params["cores"])
    cases = T.CASES
    t_all = time.perf_counter()

    # 1. the Snakefile → rule graph → job DAG (Kronecker) and its algebra
    dag = solve.build_dag(T.SNAKEFILE.read_text(encoding="utf-8"))
    names = [r["name"] for r in dag["rules"]]
    A, A_rule = dag["A"], dag["A_rule"]
    print(f"Snakefile: {len(names)} rules {names}; wildcards case × condition = {len(cases)} × {len(T.CONDITIONS)}")
    print("rule edges (consumer ← producer): " + ", ".join(f"{names[s]}←{names[r]}" for s, r in
                                                          A_rule.nonzero().tolist()))  # fmt: skip
    print(f"job DAG: A = Σ E_sr ⊗ (⊗_a M_a): {len(dag['jobs'])} jobs, {int(A.sum())} edges;  "
          f"nilpotent (A^J = 0): {eq.is_nilpotent(A)};  "
          f"max |tr(A^k)| = {int(eq.walk_traces(A).abs().max())}")  # fmt: skip
    lev = eq.levels(A)
    print("levels: " + ", ".join(f"ℓ={k}: {int((lev == k).sum())} jobs" for k in range(int(lev.max()) + 1)))
    paths = torch.linalg.inv(torch.eye(len(A), dtype=torch.float64) - A.double())
    print(f"(I − A)⁻¹ counts paths: {int(paths.sum().round())} job pairs × paths in total, "
          f"{int((paths.round() > 0).sum()) - len(A)} reachable pairs")  # fmt: skip

    # 2. run the cohort as tensors: one batched call per rule (single thread, like each Snakemake job)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    store, rule_times = solve.run_engine(dag, cases, T.CONDITIONS, theta)
    t_engine = max(b for _, b in rule_times.values())
    t0 = time.perf_counter()
    for c in cases:  # in-process, one case at a time (no process start-up, no files): isolates batching
        solve.run_engine(dag, (c,), T.CONDITIONS, theta)
    t_percase = time.perf_counter() - t0
    torch.set_num_threads(threads)
    print("tensor engine, rule by rule (one batched call each): " + ", ".join(
        f"{k} {b - a:.2f} s" for k, (a, b) in rule_times.items() if b - a > 0.005))  # fmt: skip
    print(f"tensor engine: {t_engine:.2f} s for {len(dag['jobs'])} jobs;  in-process one case at a time: "
          f"{t_percase:.2f} s (1 thread)")  # fmt: skip

    # 3. equivalence with Snakemake
    equiv = equivalence(dag, store["cohort"] if abs(theta - T.THETA) < 1e-12 else None)
    for r in equiv:
        print(f"  {'✓' if r['ok'] else '✗'} {r['check']}: tensor {r['tensor']}, Snakemake {r['snakemake']}")
    if abs(theta - T.THETA) >= 1e-12:
        print(f"  (cohort table not compared: the Snakemake run used θ = {T.THETA})")

    # 4. time: Snakemake's measured runs and the scheduling bounds from its job times
    idx = {j: i for i, j in enumerate(dag["jobs"])}
    walls = T.ref_runs()
    d = torch.zeros(len(A), dtype=torch.float64)
    for r in T.ref_timings(1):
        d[idx[(r["rule"], r["case"], r["condition"])]] = r["end"] - r["start"]
    t_fin = eq.finish_times(A, d)
    cp = float(t_fin.max())
    crit = eq.critical_path(A, t_fin, d)
    W = float(d.sum())
    print(f"Snakemake (measured, WSL2, 1 thread per job): 1 core {walls[1]:.1f} s, 6 cores {walls[6]:.1f} s, "
          f"12 cores {walls[12]:.1f} s")  # fmt: skip
    print(f"from its 1-core job times: work W = {W:.1f} s, critical path (max-plus) = {cp:.1f} s through "
          + " → ".join(f"{dag['jobs'][k][0]}" for k in crit))  # fmt: skip
    for c in sorted({1, 6, 12, cores}):
        print(f"  c = {c:2d}: lower bound max(CP, W/c) = {eq.lower_bound(d, cp, c):.1f} s"
              + (f", measured {walls[c]:.1f} s" if c in walls else ""))  # fmt: skip
    lines = (T.REF / "startup.csv").read_text().splitlines()
    startup = [float(x.split(",")[1]) for x in lines if x.startswith("import_job_module")]
    st = sum(startup) / len(startup)
    n_jobs = len(dag["jobs"]) - 1  # the target rule 'all' starts no process
    print(f"per-job overhead: starting Python and importing torch and the job module takes {st:.2f} s, × {n_jobs} "
          f"jobs = {st * n_jobs:.0f} s of the {W:.0f} s of work; the rest is file I/O and the compute "
          "itself")  # fmt: skip
    print(f"speed-up of the tensor engine: ×{walls[1] / t_engine:.1f} over 1 core, ×{walls[12] / t_engine:.1f} over "
          f"12 cores. It comes from start-up and files, not from batching (one case at a time in-process: "
          f"{t_percase:.1f} s); for a heavy GPU solve per job the start-up share, and the gain, shrink.")  # fmt: skip

    # 5. the gradient: d(left-MCA share)/dθ for every case in one backward pass, against finite differences
    img, lab, g, bnd = store["image"], store["labels"], store["centreline"], store["boundaries"]
    keep, near = store["segment"]["keep"], store["measure"]["near"]
    occ = torch.zeros(len(cases), dtype=torch.bool)

    def share(th):
        seg = solve.segment(img, lab, th, keep=keep)
        m = solve.measure(seg, g, near=near)
        return solve.report(m, bnd, g, solve.flow(m, bnd, g, occ))["share_L-MCA"]

    th = torch.full((len(cases),), theta, dtype=torch.float64, requires_grad=True)
    t0 = time.perf_counter()
    s = share(th)
    s.sum().backward()  # cases are independent, so one backward pass gives every ∂s_b/∂θ_b
    t_grad = time.perf_counter() - t0
    dth = 2e-3
    fd = (share(th.detach() + dth) - share(th.detach() - dth)) / (2 * dth)
    print(f"∂(L-MCA share)/∂θ, one backward pass ({t_grad:.2f} s): "
          + ", ".join(f"{c} {a:+.3f} (FD {b:+.3f})" for c, a, b in zip(cases, th.grad.tolist(), fd.tolist()))
          + f";  max |autograd − FD| = {float((th.grad - fd).abs().max()):.1e}  "
          "(Snakemake: 2 extra full reruns)")  # fmt: skip

    # 6. the clinical-style numbers per case
    rep = {k: v.detach().reshape(len(cases), -1) for k, v in store["report"].items()}
    for k, c in enumerate(cases):
        print(f"{c}: Dice {rep['dice'][k, 0]:.3f}, {int(rep['nodes'][k, 0])} nodes, {int(rep['edges'][k, 0])} "
              f"edges, {rep['length_mm'][k, 0]:.0f} mm of centreline;  inflow R-ICA/L-ICA/BA "
              f"{rep['Q_in_R-ICA'][k, 0]:.2f}/{rep['Q_in_L-ICA'][k, 0]:.2f}/{rep['Q_in_BA'][k, 0]:.2f} ml/s, "
              f"Δp {rep['dp_max_mmHg'][k, 0]:.1f} mmHg;  L-ICA closed: L-MCA "
              f"{100 * rep['Q_L-MCA'][k, 1] / rep['Q_L-MCA'][k, 0]:.0f} %, L-ACA "
              f"{100 * rep['Q_L-ACA'][k, 1] / max(float(rep['Q_L-ACA'][k, 0]), 1e-9):.0f} % of baseline;  "
              f"stenosis flags {int(rep['stenoses'][k, 0])}")  # fmt: skip

    # 7. results: tables, the 3D scene, the animation
    res = out_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "cohort.csv").write_text(store["cohort"], encoding="utf-8")
    with open(res / "equivalence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "tensor", "snakemake", "ok"])
        for r in equiv:
            w.writerow([r["check"], r["tensor"], r["snakemake"], r["ok"]])
    bars = {"Snakemake 1 core": walls[1], "Snakemake 6 cores": walls[6], "Snakemake 12 cores": walls[12],
            "bound, 12 cores": eq.lower_bound(d, cp, 12), "case by case": t_percase,
            "tensor engine": t_engine}  # fmt: skip
    with open(res / "timings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# wall-clock seconds for the whole cohort; Snakemake measured offline (WSL2, AMD Ryzen 5 5600G, "
                    "1 thread per job), the tensor engine measured in this run (1 thread)"])  # fmt: skip
        w.writerow(["how", "seconds"])
        for k, v in bars.items():
            w.writerow([k, f"{v:.3f}"])
        for name, (a, b) in rule_times.items():
            w.writerow([f"tensor rule {name}", f"{b - a:.4f}"])
    surfaces = []
    for k in range(len(cases)):  # marching tetrahedra of the soft segmentation inside the labelled-vessel region
        field = (store["segment"]["soft"][k] * store["segment"]["region"][k]).detach()
        field = F.avg_pool3d(field[None, None], 3, 1, 1)[0, 0]
        surfaces.append(tuple(x.numpy() for x in solve.iso_surface(field, 0.5, T.H)))
    for old in res.glob("*.vtp"):  # stale pieces of an earlier layout
        old.unlink()
    (res / "cow.pvd").unlink(missing_ok=True)
    # 6. a longer flow sequence for the 3D result: the left ICA narrows from open (s = 0) to closed (s = 1)
    with torch.no_grad():
        sev, sweep = solve.stenosis_sweep(store["measure"], bnd, g)
    l_mca = [solve.report(store["measure"], bnd, g, fs)["share_L-MCA"] for fs in sweep]
    l_ica = [solve.report(store["measure"], bnd, g, fs)["Q_in_L-ICA"] for fs in sweep]
    with open(res / "stenosis_sweep.csv", "w", newline="", encoding="utf-8") as f:
        f.write("# Left-ICA stenosis sweep: severity s = fraction of the lumen diameter lost (0 open, 1 closed).\n")
        f.write("# Steady 1D network flow per case (tensor engine only; the Snakemake workflow runs s = 0 and 1).\n")
        w = csv.writer(f)
        w.writerow(["severity", "case", "Q_in_L-ICA_ml_s", "share_L-MCA"])
        for i, s in enumerate(sev.tolist()):
            for k, c in enumerate(cases):
                w.writerow([f"{s:.2f}", c, f"{float(l_ica[i][k]):.4f}", f"{float(l_mca[i][k]):.4f}"])
    print("left-ICA stenosis sweep (" + ", ".join(f"s={s:.2f}" for s in sev[::5].tolist()) + "): mean left-ICA "
          "inflow " + " → ".join(f"{float(x.mean()):.2f}" for x in l_ica[::5]) + " ml/s")  # fmt: skip
    anatomy, flow = plot.write_scene(res, store, cases, surfaces, sev, sweep)
    size = sum(f.stat().st_size for f in res.iterdir() if f.suffix == ".vtp") / 1e6
    wss = [eq.wall_shear_stress(fs["q"], store["measure"]["radius"], T.MU)[g["edge_valid"]] for fs in sweep]
    print(f"3D results: {anatomy.relative_to(out_dir)} (segmentation and centrelines by radius) and "
          f"{flow.relative_to(out_dir)} over {len(sev)} steps of left-ICA stenosis (s = 0 open → 1 closed): surface "
          f"WSS = 4μ|Q|/(πr³) (and pressure), idealised axial Poiseuille streamlines and particles "
          f"(Δt = {plot.DT * 1e3:.0f} ms of pseudo-time per step), inlets/outlets; median WSS "
          f"{float(wss[0].median()):.2f} Pa at s = 0, {float(wss[-1].median()):.2f} Pa at s = 1;  "
          f"{sum(len(s[1]) for s in surfaces)} surface triangles before decimation, {size:.1f} MB")  # fmt: skip
    sm12 = T.ref_timings(12)
    timing = {"levels": lev.numpy(), "critical": crit, "snakemake12": sm12, "wall12": walls[12], "engine": rule_times,
              "end12": {(r["rule"], r["case"], r["condition"]): r["end"] for r in sm12}, "bars": bars}  # fmt: skip
    grad = {"autograd": th.grad.numpy(), "fd": fd.numpy()}
    print(f"compute {time.perf_counter() - t_all:.1f} s (animation next{', fast mode' if fast_mode() else ''})")
    return plot.animate(store, dag, list(cases), timing, equiv, grad, out_dir / "cow_workflow_dag.gif")


if __name__ == "__main__":
    cli(run, __file__)
