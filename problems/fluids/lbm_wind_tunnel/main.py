"""Main loop: parameters → STL → voxels → lattice Boltzmann wind tunnel → drag and lift → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import drag_coefficient, schiller_naumann, viscosity_to_tau

DATA = Path(__file__).parent / "data"
U_IN, FRINGE, NOSE_X, RAMP = 0.08, 16, 12.0, 300  # free stream, fringe length, nose position, start-up ramp (steps)
VALIDATION_STEPS = 2000  # the sphere wake has settled by then (residuals as in test.py)
BODIES = {  # STL, lattice (Nx, Ny, Nz), body length in nodes; aircraft fly in a box periodic in y and z
    "airplane": {"stl": "airplane.stl", "shape": (80, 36, 26), "length": 32.0,
                 "reference": "no measurement: a generated model; C_L > 0 at positive angle of attack (planform area)"},
    "naca0012_wing": {"stl": "naca0012_wing.stl", "shape": (72, 24, 24), "length": 20.0,
                      "reference": "thin-aerofoil theory for an infinite wing: C_L ≈ 2π α (planform area)"},
    "sphere_openlb": {"stl": None, "shape": solve.SPHERE_CASE["shape"], "length": solve.SPHERE_CASE["diameter"],
                      "reference": "OpenLB 1.7, same lattice (data/openlb_reference_*.csv)"},
}  # fmt: skip


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    dt = torch.float32
    body, re, n_steps = params["body"], params["re"], params["steps"]
    collision, les = params["collision"], params["les"] == "on"
    cfg = BODIES[body]
    nx, ny, nz = cfg["shape"]
    check = body == "sphere_openlb"  # the OpenLB validation case: its own lattice, Re, collision and boundaries

    if check:
        re, collision, les, pitch = solve.SPHERE_CASE["re"], solve.SPHERE_CASE["collision"], False, 0.0
        tri = tensors.sphere_mesh(cfg["length"] / 2, solve.SPHERE_CASE["centre"], dtype=dt).to(dev)
        print(f"OpenLB check: Re = {re:.0f}, BGK, no LES, equilibrium inlet and copy outlet (other params ignored)")
    else:
        tri_stl = tensors.read_stl(DATA / cfg["stl"], dev, dt)  # (T, 3, 3) in mm
        pitch = params["pitch"]
        tri, _ = tensors.place_body(tri_stl, (nx, ny, nz), cfg["length"], NOSE_X, None, pitch_deg=pitch)  # centred
    solid = tensors.voxelise(tri, (nx, ny, nz))  # (Nx, Ny, Nz) bool: ray casting, winding number
    if check:
        area = torch.pi * cfg["length"] ** 2 / 4
    else:
        area = tensors.planform_area(solid)  # wing-style reference area: the body seen from above
    u_in = solve.SPHERE_CASE["u"] if check else U_IN
    nu = u_in * cfg["length"] / re
    tau = viscosity_to_tau(nu)

    e, w, _ = tensors.lattice(dev, dt)
    f = tensors.initial_populations(solid, u_in if check else 0.0, e, w)  # (19, Nx, Ny, Nz); tunnel: air at rest
    lo = (tri.reshape(-1, 3).min(0).values.cpu() - 2).clamp(min=0.5)
    hi = torch.minimum(tri.reshape(-1, 3).max(0).values.cpu() + 2, torch.tensor([nx, ny, nz]) - 1.5)
    ys = torch.linspace(float(lo[1]), float(hi[1]), 9)
    zs = torch.linspace(float(lo[2]), float(hi[2]), 5)
    x_rake = (float(tri[..., 0].min()) - 6) if check else NOSE_X - 6
    rake = torch.stack([torch.full((45,), x_rake), ys.repeat_interleave(5), zs.repeat(9)], 1).to(dev, dt)
    warmup = 0 if check else RAMP + 100  # the tunnel ramps the speed up smoothly; frames start once it is running
    result = solve.simulate(
        f, solid, u_in, tau, n_steps, collision=collision, les=les, walls=False, open_ends=check,
        fringe=0 if check else FRINGE, ramp=0 if check else RAMP,
        save_every=max(1, (n_steps - warmup) // 48), save_from=warmup, n_particles=2000,
        window=(lo.tolist(), hi.tolist()), seeds=rake,
    )  # fmt: skip
    coeffs = drag_coefficient(result["force"], u_in, area)  # (T, 3): C_D, C_side, C_L
    late = coeffs[n_steps // 2 :]
    cd, cl = late[:, 0].mean().item(), late[:, 2].mean().item()
    blockage = tensors.frontal_area(solid) / (ny * nz)

    print(f"lattice {nx}x{ny}x{nz} = {nx * ny * nz:,} nodes, D3Q19 {collision.upper()}{' + LES' if les else ''}")
    print(f"body '{body}': {int(solid.sum())} solid nodes from {len(tri)} STL triangles, length {cfg['length']:.0f},"
          f" pitch {pitch:.0f}°, reference area {area:.0f} nodes², blockage {blockage:.0%}")  # fmt: skip
    print(f"U = {u_in}, Re = U L / nu = {re:.0f}, nu = {nu:.5f}, tau = {tau:.4f}, Mach = {u_in * 3**0.5:.2f}")
    print(f"second half of the run: C_D = {cd:.3f} ± {late[:, 0].std().item():.3f}, C_L = {cl:+.3f}")
    if body == "naca0012_wing":
        print(f"2π α = {2 * torch.pi * torch.deg2rad(torch.tensor(pitch)).item():.3f}")
    if not check:
        print(f"reference: {cfg['reference']}; the lattice Re is far lower, so compare qualitatively")

    validation = None
    if check:  # this run is the validation case itself
        ux_final = solve.macroscopic(result["f"], e)[1][0, :, :, nz // 2].masked_fill(solid[:, :, nz // 2], 0.0)
        validation = solve.openlb_residuals(coeffs[:, 0], ux_final, DATA)
        validation["ux_ours"] = ux_final.cpu()
    elif params["validate"] == "on":  # option (b): run the OpenLB sphere case after the body, on its own lattice
        case = dict(solve.SPHERE_CASE, n_steps=VALIDATION_STEPS)
        sphere = solve.sphere_case(**case, device=dev)
        validation = solve.openlb_residuals(sphere["cd"], sphere["midplane_ux"], DATA)
        validation["ux_ours"] = sphere["midplane_ux"].cpu()
    if validation is not None:  # residual computed live against the bundled OpenLB CSVs
        validation["u"] = solve.SPHERE_CASE["u"]
        print(f"validation, tensor solver − OpenLB 1.7 (sphere, Re = 20, 80x40x40, BGK): C_D {validation['cd']:.3f}"
              f" vs {validation['cd_ref']:.3f} (ΔC_D {validation['cd_rel']:+.1%}); mid-plane u_x: rel. L2"
              f" {validation['l2']:.2%}, Linf {validation['linf']:.2%},"
              f" max |du|/U {validation['max_du']:.2%}")  # fmt: skip
        print(f"Schiller–Naumann (free sphere): C_D = {schiller_naumann(torch.tensor(20.0)).item():.3f}")
    pvd = plot.export_vtk(result, tri, solid, u_in, cfg["length"], out_dir)  # interactive 3D results
    size = sum(p.stat().st_size for p in pvd.parent.iterdir()) / 1e6
    print(f"interactive results: {pvd.relative_to(out_dir)} and its layers ({size:.1f} MB)")
    return plot.animate(result, coeffs, tri, solid, u_in, body, cfg["length"], out_dir / "lbm_wind_tunnel.gif",
                        validation=validation)  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
