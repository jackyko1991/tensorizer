"""Main loop: STO-3G basis → integrals for a batch of bond lengths → SCF → potential curve and orbitals."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import basis_on_points, nuclear_repulsion

# Szabo & Ostlund's worked examples (STO-3G, ζ_H = 1.24, ζ_He = 2.0925): (R in bohr, total energy in hartree)
REFERENCE = {"H2": (1.4, -1.1167), "HeH+": (1.4632, -2.8607)}


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    name = params["molecule"]
    Z, zeta, n_atom = tensors.molecule(name, dev)
    alpha, coef = tensors.sto3g(zeta)  # (M, K) each
    n_R = 24 if fast_mode() else params["n_bond_lengths"]
    R_ref, E_ref = REFERENCE[name]
    # the scan plus the reference geometry as one extra batch entry
    R = torch.cat([tensors.bond_lengths(params["R_min"], params["R_max"], n_R, dev), torch.tensor([R_ref], device=dev)])
    nuclei = tensors.geometry(R)  # (B, 2, 3)
    ints = solve.integrals(alpha, coef, nuclei, Z)
    print(f"{name} in STO-3G: ζ = {zeta.tolist()}, exponents α_μp (bohr⁻²):")
    for m in range(len(zeta)):
        print(f"  atom {m}: α = {[round(a, 5) for a in alpha[m].tolist()]}")
    print(f"{len(R)} bond lengths at once: S, T, V {tuple(ints['S'].shape)}, (μν|λσ) {tuple(ints['eri'].shape)}")

    res = solve.scf(ints, tensors.N_ELECTRONS)
    E_nuc = nuclear_repulsion(nuclei, Z)
    E = res["E_el"] + E_nuc  # (B,) total energy
    E_ci = solve.full_ci(ints, res["C"]) + E_nuc
    n_elec = torch.einsum("bmn,bnm->b", res["P"], ints["S"])  # tr(PS)
    print(
        f"SCF converged in {res['n_iter']} iterations (max |ΔP| = {res['history'][-1]:.1e}); tr(PS) = "
        f"{n_elec.min().item():.6f} … {n_elec.max().item():.6f}"
    )

    k = len(R) - 1  # the reference geometry
    S, H, F, C = ints["S"][k], ints["H"][k], res["F"][k], res["C"][k]
    print(f"\nR = {R_ref} bohr:")
    print(f"  S = [[{S[0, 0]:.4f}, {S[0, 1]:.4f}], [{S[1, 0]:.4f}, {S[1, 1]:.4f}]]")
    print(f"  H_core = [[{H[0, 0]:.4f}, {H[0, 1]:.4f}], [{H[1, 0]:.4f}, {H[1, 1]:.4f}]]")
    print(f"  F = [[{F[0, 0]:.4f}, {F[0, 1]:.4f}], [{F[1, 0]:.4f}, {F[1, 1]:.4f}]]")
    print(f"  orbital energies ε = {res['eps'][k, 0]:.4f}, {res['eps'][k, 1]:.4f} hartree")
    print(f"  MO coefficients C (columns = MOs): [[{C[0, 0]:.4f}, {C[0, 1]:.4f}], [{C[1, 0]:.4f}, {C[1, 1]:.4f}]]")
    print(f"  E_electronic = {res['E_el'][k]:.6f}, E_nuclear = {E_nuc[k]:.6f}")
    print(f"  E_RHF = {E[k]:.6f} hartree   (Szabo & Ostlund: {E_ref})   full CI = {E_ci[k]:.6f}")

    R_scan, E_scan, Eci_scan = R[:-1], E[:-1], E_ci[:-1]
    r0, e0 = solve.equilibrium(R_scan, E_scan)
    r0_ci, e0_ci = solve.equilibrium(R_scan, Eci_scan)
    E_atoms = solve.separated_atoms(alpha, coef, Z, n_atom)
    print(f"\nequilibrium (RHF):     R_e = {r0:.4f} bohr = {0.529177 * r0:.4f} Å,  E = {e0:.6f} hartree")
    print(f"equilibrium (full CI): R_e = {r0_ci:.4f} bohr,  E = {e0_ci:.6f} hartree")
    print(f"separated atoms ({'H + H' if name == 'H2' else 'He + H⁺'}): {E_atoms:.6f} hartree")
    print(
        f"at R = {R_scan[-1]:.1f} bohr: RHF {E_scan[-1]:.6f}, full CI {Eci_scan[-1]:.6f}"
        f"  → RHF misses the limit by {E_scan[-1].item() - E_atoms:.4f} hartree"
    )

    z = tensors.axis_points(-3.0, params["R_max"] + 3.0, 300, dev)
    phi = basis_on_points(alpha, coef, nuclei[:-1], z)  # (B, Z, M)
    psi = torch.einsum("bzm,bma->bza", phi, res["C"][:-1])  # MOs along the axis
    return plot.animate(
        name, R_scan, E_scan, Eci_scan, E_atoms, res["eps"][:-1], res["C"][:-1], z[:, 2], psi, (r0, e0),
        out_dir / "hartree_fock_h2.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
