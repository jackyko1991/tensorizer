"""Main loop: STL → voxel mesh → centrelines → pulsatile Navier–Stokes → WSS, TAWSS, OSI → animation and 3D files,
with the residuals against the bundled OpenFOAM and VMTK reference results."""

import math
import time
from pathlib import Path

import torch

from problems._common.runtime import cli

from . import equation, plot, solve, tensors

TITLES = {
    "lausanne": "aneurysm lumen segmented from the CC0 Lausanne TOF-MRA cohort (sub-074)",
    "synthetic": "synthetic bifurcation aneurysm",
}


def pipe_check(nu: float, rho: float) -> dict:
    """Our solver on OpenFOAM's reference case: Womersley pipe, R = 2 mm, −∂_z p/ρ = 1.05 + 2.1 sin 2πt (m/s²)."""
    R = 2e-3
    grid = tensors.pipe_grid(R, R / 8)
    mesh = solve.build_mesh(grid, periodic=(False, False, True))
    out = solve.pipe_run(grid, mesh, 1.05, 2.1, 2 * math.pi, 2.0, 1e-3, nu, rho, n_rec=200)
    res = solve.openfoam_residuals(out, tensors.openfoam_reference(), 1.0, 2.0, rho)  # second cycle
    G = [complex(1.05), complex(0.0, -2.1)]
    exact = equation.womersley_pressure(torch.zeros(1, dtype=torch.float64), res["t"], G, 2 * math.pi, R, nu)[:, 0]
    sim = res["u_axis"][0]
    res["exact"] = exact
    res["an_L2"] = ((sim - exact).norm() / exact.norm()).item()
    ca, cb = solve.harmonic(sim, res["t"], 2 * math.pi), solve.harmonic(exact, res["t"], 2 * math.pi)
    res["an_amp"] = abs(ca) / abs(cb) - 1
    res["an_phase"] = math.remainder(math.atan2(ca.imag, ca.real) - math.atan2(cb.imag, cb.real), 2 * math.pi)
    return res


def aneurysm_residuals(rec: dict, ref: dict) -> dict:
    """Ours minus OpenFOAM on the same voxel mesh: |u| on the slice cells at the 25 sample times, |WSS| at the wall
    vertices at peak systole and TAWSS over the same samples. Relative L2 and L∞ errors."""
    du = rec["probe_u"].T - ref["u"]  # (S, T)
    kp = int((ref["t"] - ref["t_peak"]).abs().argmin())
    w_ours, ta_ours = rec["probe_wss"][kp], rec["probe_wss"].mean(0)

    def rel(d, r):
        return (d.norm() / r.norm()).item(), (d.abs().max() / r.abs().max()).item()

    out = {"du": du, "ref": ref, "dtawss": ta_ours - ref["tawss"], "dwss_peak": w_ours - ref["wss_peak"]}
    out["u_L2"], out["u_Linf"] = rel(du, ref["u"])
    out["wss_L2"], out["wss_Linf"] = rel(out["dwss_peak"], ref["wss_peak"])
    out["tawss_L2"], out["tawss_Linf"] = rel(out["dtawss"], ref["tawss"])
    return out


def run(params: dict, out_dir: Path) -> Path:
    torch.manual_seed(0)
    geometry = params["geometry"]
    h = params["voxel"] * tensors.MM
    nu, rho = tensors.NU, tensors.RHO
    t0 = time.time()
    tris, ids = tensors.surface(geometry)
    grid = tensors.lumen_grid(tris, ids, h)
    mesh = solve.build_mesh(grid)
    vol = tensors.mesh_volume(tris)
    print(f"{geometry} surface: {len(tris)} triangles, {vol * 1e9:.0f} mm³;  voxels h = {h * 1e3:.2f} mm: "
          f"{mesh['n']} fluid cells ({mesh['n'] * h**3 * 1e9:.0f} mm³), {len(mesh['owner'])} faces")  # fmt: skip
    caps = grid["caps"]
    print("caps (after VMTK flow extensions):  inlet r = %.2f mm,  outlets r = %s mm" % (
        caps[0]["radius"] * 1e3, ", ".join(f"{c['radius'] * 1e3:.2f}" for c in caps[1:])))  # fmt: skip

    # centrelines: tensor version of vmtkcenterlines, with the residual against the bundled VMTK result
    seeds = torch.tensor([solve.outlet_seed(grid, c) for c in caps[1:]])
    lines, bif, info = solve.centrelines(grid, solve.cap_nodes(grid, caps[0]), seeds)
    vres = solve.vmtk_residuals(lines, *tensors.vmtk_reference(geometry))
    print(f"eikonal: {info['iters']} min-plus sweeps;  bifurcation at abscissa {bif['s'] * 1e3:.1f} mm")
    for r in vres:
        d, dR, line = r["d"], r["dR"], lines[r["k"]]
        print(f"centreline to outlet {r['k'] + 1}: {line['s'][-1] * 1e3:.1f} mm (VMTK {r['s_ref'][-1] * 1e3:.1f}): "
              f"distance mean {d.mean() * 1e3:.3f} / max {d.max() * 1e3:.3f} mm "
              f"({d.mean() / h:.2f} / {d.max() / h:.2f} voxels),  |ΔR| mean {dR.mean() * 1e3:.3f} / "
              f"max {dR.max() * 1e3:.3f} mm ({dR.mean() / h:.2f} / {dR.max() / h:.2f} voxels)")  # fmt: skip

    # the aneurysm sac: lumen voxels farther from every centreline than its local radius (+ ½ mm)
    allx, allR = torch.cat([ln["x"] for ln in lines]), torch.cat([ln["R"] for ln in lines])
    dmin, j = torch.cdist(mesh["pos"], allx).min(1)
    sac = dmin > allR[j] + 0.5e-3
    print(f"aneurysm sac (lumen off every centreline): {sac.sum() * h**3 * 1e9:.0f} mm³")

    # residuals of the flow solver against OpenFOAM (pimpleFoam) and Womersley's exact solution on the pipe case
    pres = pipe_check(nu, rho)
    print(f"pipe vs OpenFOAM v1912 pimpleFoam (cycle 2): u_axis L2 {pres['u_axis_L2'] * 100:.2f} %, "
          f"L∞ {pres['u_axis_Linf'] * 100:.2f} %, amplitude {pres['amp_err'] * 100:+.2f} %, "
          f"phase {pres['phase_err'] * 1e3:+.1f} mrad;  WSS L2 {pres['wss_L2'] * 100:.2f} %, "
          f"L∞ {pres['wss_Linf'] * 100:.2f} %;  flow rate L2 {pres['Q_L2'] * 100:.2f} %")  # fmt: skip
    print(
        f"pipe vs Womersley (exact): u_axis L2 {pres['an_L2'] * 100:.2f} %, amplitude {pres['an_amp'] * 100:+.2f} %, "
        f"phase {pres['an_phase'] * 1e3:+.1f} mrad;  Δp imposed in both (the driver)"
    )

    slice_y = int(round(((mesh["pos"][sac].mean(0)[1] - grid["lo"][1]) / h).item()))  # slice through the sac
    t_wave, Q = tensors.inflow_waveform()
    R0 = caps[0]["radius"]
    Q = Q * (R0 / tensors.R_ICA) ** 3  # Murray's law: scale the ICA flow to the inlet radius
    dt = params["dt_ms"] * 1e-3
    lam = dt * nu / h**2 * (12 + 4 * (1 - mesh["ghost"].min().item()))
    print(f"mean inflow {Q.mean() * 1e6:.2f} ml/s (4.58 ml/s × (R_in/2 mm)³);  Δt = {dt * 1e3:.2f} ms "
          f"(explicit viscous number {lam:.2f} < 2;  semi-Lagrangian advection has no CFL limit)")  # fmt: skip
    ref = tensors.openfoam_aneurysm_reference() if geometry == "lausanne" else None
    probe = None if ref is None else {"times": ref["t"], "points": ref["points"]}
    rec = solve.simulate(mesh, (t_wave, Q), params["cycles"], dt, nu, rho, grid["V"], grid["N"],
                         n_frames=params["frames"], slice_at=slice_y, probe=probe)  # fmt: skip
    q_in, q_out = torch.tensor(rec["Q_in"]), rec["Q_out"]
    err = ((q_out.sum(1) - q_in).abs().max() / q_in.abs().max()).item()
    split = " / ".join(f"{(q_out[:, k].mean() / q_in.mean()) * 100:.1f} %" for k in range(q_out.shape[1]))
    print(
        f"mass: inflow {q_in.mean() * 1e6:.3f} ml/s (prescribed {Q.mean() * 1e6:.3f}), outflow split {split}, "
        f"max |in − Σout|/in = {err:.1e};  PCG iterations/step {sum(rec['iters']) / len(rec['iters']):.1f}"
    )
    dp = torch.tensor(rec["dp"])
    print(f"pressure drop inlet → outlets: mean {dp.mean():.0f} Pa ({dp.mean() / 133.3:.1f} mmHg), "
          f"range {dp.min():.0f} … {dp.max():.0f} Pa over the cycle")  # fmt: skip

    ares = None
    if ref is not None and len(rec["probe_t"]) == len(ref["t"]):
        ares = aneurysm_residuals(rec, ref)
        print(f"aneurysm vs OpenFOAM (same voxel mesh): |u| on the slice L2 {ares['u_L2'] * 100:.1f} %, "
              f"L∞ {ares['u_Linf'] * 100:.1f} %;  |WSS| at peak systole L2 {ares['wss_L2'] * 100:.1f} %, "
              f"L∞ {ares['wss_Linf'] * 100:.1f} %;  TAWSS L2 {ares['tawss_L2'] * 100:.1f} %, "
              f"L∞ {ares['tawss_Linf'] * 100:.1f} %")  # fmt: skip
    else:
        print("aneurysm vs OpenFOAM: no reference for the synthetic geometry (skipped)")

    tau = rec["wss"]  # (T, V, 3)
    TA, OS = equation.tawss(tau), equation.osi(tau)
    vsac = torch.cdist(grid["V"].float(), mesh["pos"][sac].float()).min(1).values < h  # wall vertices on the sac
    print(f"TAWSS: parent/branches median {TA[~vsac].median():.2f} Pa, aneurysm sac median {TA[vsac].median():.2f} Pa")
    print(f"OSI: parent/branches median {OS[~vsac].median():.3f}, sac median {OS[vsac].median():.3f}, "
          f"sac max {OS[vsac].max():.3f}")  # fmt: skip
    print(f"parent artery: Poiseuille 4μQ̄/(πR³) = {equation.poiseuille_wss(Q.mean().item(), R0, rho * nu):.2f} Pa, "
          f"Reynolds 2Q̄/(πRν) = {2 * Q.mean().item() / (math.pi * R0 * nu):.0f}, "
          f"Womersley α = R√(ω/ν) = {R0 * math.sqrt(2 * math.pi / rec['T'] / nu):.2f}")  # fmt: skip

    # interactive 3D results for the web page: an animated VTK collection (surface, streamlines, particles)
    pvd = plot.write_vtk_series(out_dir / "results", tris, ids, grid, rec, TA, OS)
    size = sum(f.stat().st_size for f in pvd.parent.iterdir()) / 1e6
    print(f"ParaView time series: {pvd.relative_to(out_dir)} with {len(rec['t'])} steps × 3 layers, {size:.1f} MB")
    print(f"compute time {time.time() - t0:.1f} s (animation next)")
    return plot.animate(tris, ids, grid, mesh, lines, vres, rec, TA, OS, sac, pres, ares, (t_wave, Q),
                        out_dir / "aneurysm_cfd.gif", title=TITLES[geometry])  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
