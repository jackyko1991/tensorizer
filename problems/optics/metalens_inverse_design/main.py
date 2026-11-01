"""Main loop: optics setup → analytic hyperbolic lens → inverse design with Adam → field maps → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    lam0, lam1, f, D = params["wavelength"], params["wavelength2"], params["focal_length"], params["diameter"]
    wavelengths = torch.tensor([lam0, lam1])  # batch of two wavelengths, li = 0, 1
    optics = solve.Optics(wavelengths, lam0, f, D, dev)

    # Analytic reference: the wrapped hyperbolic phase, designed for λ0.
    centres = tensors.atom_centres(D, dev)
    theta_hyp = torch.remainder(equation.hyperbolic_theta(centres, f, lam0), 1.0)  # (n_atoms,)
    eta_hyp = optics.efficiency(theta_hyp).detach().cpu()

    # Inverse design from a flat surface. 'single': only λ0 counts; 'dual': both wavelengths count equally.
    weights = torch.tensor([1.0, 0.0] if params["design"] == "single" else [0.5, 0.5], device=dev)
    n_iter = 20 if fast_mode() else params["n_iter"]
    theta0 = torch.zeros(tensors.n_atoms(D), device=dev)
    history, snap_iters, snaps = solve.optimise(optics, theta0, weights, n_iter, params["lr"])
    eta_opt = history[-1]

    z_scan = torch.linspace(0.3 * f, 2.0 * f, 400, device=dev)
    peaks_hyp = solve.on_axis_peak(optics, theta_hyp, z_scan).tolist()
    peaks_opt = solve.on_axis_peak(optics, snaps[-1], z_scan).tolist()
    print(f"metalens: D = {D} μm, f = {f} μm, NA = {D / 2 / (D**2 / 4 + f**2) ** 0.5:.2f}, "
          f"{tensors.n_atoms(D)} meta-atoms of {tensors.PERIOD} μm, design '{params['design']}'")  # fmt: skip
    for li, lam in enumerate(wavelengths.tolist()):
        print(
            f"λ = {lam:.3f} μm   focal-spot efficiency: hyperbolic {eta_hyp[li]:6.1%}   optimised {eta_opt[li]:6.1%}"
            f"   brightest on-axis point: hyperbolic z = {peaks_hyp[li]:5.1f}, optimised z = {peaks_opt[li]:5.1f} μm"
        )

    # |E(x, z)| maps for the animation: every snapshot, both wavelengths.
    z_map = torch.linspace(0.0, 1.6 * f, 24 if fast_mode() else 110, device=dev)
    keep = torch.nonzero(optics.x.abs() < 0.75 * D).squeeze(1)[::4]  # crop and thin the x axis for drawing
    maps = torch.stack([optics.field_map(th, z_map, keep) for th in snaps]).cpu()  # (S, L, Z, X_kept)
    map_hyp = optics.field_map(theta_hyp, z_map, keep).cpu()
    return plot.animate(
        maps, map_hyp, optics.x[keep].cpu(), z_map.cpu(), snaps.cpu(), theta_hyp.cpu(), centres.cpu(),
        history, eta_hyp, snap_iters, wavelengths, f, out_dir / "metalens_inverse_design.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
