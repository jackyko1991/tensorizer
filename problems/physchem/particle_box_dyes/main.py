"""Main loop: parameters → molecules and boxes → analytic and numerical levels → λ_max and colours → plot."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    bond, end = params["bond"], params["end"]
    names, n_bonds, n_el = tensors.molecules(params["series"], params["n_max"], dev)  # B names, (B,), (B,)
    L = tensors.box_length(n_bonds, bond, end)  # (B,) nm
    n_levels = int(n_el.max()) // 2 + 2  # one above the highest LUMO
    n = tensors.level_numbers(n_levels, dev)  # (n,)

    # analytic free-electron model
    E_exact = equation.energy_levels(n, L)  # (B, n) eV
    gap = equation.homo_lumo_gap(n_el, L)  # (B,) eV
    lam = equation.wavelength(gap)  # (B,) nm

    # numerical box: finite-difference Hamiltonian, diagonalised for all molecules at once
    n_points = 120 if fast_mode() else params["n_points"]
    x, dx = tensors.grids(L, n_points)  # (B, M), (B,)
    V = equation.alternation_potential(x, params["V_alt"], bond, end)  # (B, M)
    E_num, psi = solve.solve(equation.hamiltonian(dx, V), dx, n_levels)  # (B, n), (B, M, n)
    homo = (n_el // 2 - 1)[:, None]  # 0-based column of the HOMO
    gap_num = (E_num.gather(1, homo + 1) - E_num.gather(1, homo))[:, 0]
    lam_num = equation.wavelength(gap_num)

    # colour of each dye: white light minus its absorption band
    lam_vis = tensors.visible_wavelengths(dev)  # (W,)
    rgb = equation.wavelength_rgb(lam_vis)  # (W, 3)
    colour = equation.perceived_colour(equation.absorption_band(lam_vis, lam_num), rgb)  # (B, 3)

    print(
        f"series {params['series']}   ·   bond l = {bond:.3f} nm   ·   end extension {end:g} l   ·   "
        f"V_alt = {params['V_alt']:g} eV"
    )
    print(f"{'molecule':<11} {'bonds':>5} {'N_π':>4} {'L (nm)':>7} {'ΔE (eV)':>8} {'λ_max (nm)':>11} {'numeric':>8}")
    for b, name in enumerate(names):
        print(
            f"{name:<11} {int(n_bonds[b]):>5} {int(n_el[b]):>4} {L[b].item():7.3f} {gap[b].item():8.3f} "
            f"{lam[b].item():11.1f} {lam_num[b].item():8.1f}"
        )
    if params["V_alt"] == 0:
        ratio = E_num[-1, :6] / E_num[-1, 0]
        err = ((E_num - E_exact) / E_exact).abs().max().item()
        print(f"numerical E_n/E_1 (longest box) = {', '.join(f'{r:.3f}' for r in ratio.tolist())}   (exact n²)")
        print(f"largest relative error of the {n_points}-point grid: {err:.1e}")
    shift = (lam[1:] - lam[:-1]).mean().item()
    print(f"each added CH=CH unit shifts λ_max by {shift:.0f} nm on average")

    measured = tensors.measured_cyanine(dev) if params["series"] == "cyanine" else None
    return plot.animate(
        names,
        L,
        n_bonds,
        bond,
        end,
        n_el,
        E_num,
        psi,
        x,
        lam,
        lam_num,
        colour,
        measured,
        out_dir / "particle_box_dyes.gif",
    )


if __name__ == "__main__":
    cli(run, __file__)
