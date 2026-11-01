"""Animate the bond-length scan: potential curves, the two MOs along the axis, and the orbital energies."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from problems._common.plotting import frame_indices, save_gif

BOND, ANTI = "#2563eb", "#dc2626"


def animate(
    name: str,
    R: torch.Tensor,
    E: torch.Tensor,
    E_ci: torch.Tensor,
    E_atoms: float,
    eps: torch.Tensor,
    C: torch.Tensor,
    z: torch.Tensor,
    psi: torch.Tensor,
    eq_point: tuple[float, float],
    path: Path,
) -> Path:
    """R, E, E_ci (B,); eps (B, 2); C (B, 2, 2); z (Z,) positions on the axis; psi (B, Z, 2) MOs along it."""
    R, E, E_ci, eps, C = (x.cpu().numpy() for x in (R, E, E_ci, eps, C))
    z, psi = z.cpu().numpy(), psi.cpu().numpy()
    atoms = ("H", "H") if name == "H2" else ("He", "H")
    limit = "H + H" if name == "H2" else "He + H⁺"
    pretty = "H₂" if name == "H2" else "HeH⁺"

    fig, (ax_e, ax_mo, ax_eps) = plt.subplots(1, 3, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.1, 1.2, 0.9]})
    lo, hi = min(E.min(), E_ci.min()), max(E.max(), E_ci.max(), E_atoms)
    ax_e.plot(R, E, color="#d1d5db", lw=1)
    ax_e.plot(R, E_ci, color="#d1d5db", lw=1, ls="--")
    (line_hf,) = ax_e.plot([], [], color=BOND, lw=2, label="RHF")
    (line_ci,) = ax_e.plot([], [], color="#16a34a", lw=1.5, ls="--", label="full CI (same basis)")
    ax_e.axhline(E_atoms, color="#f59e0b", lw=1, ls=":", label=f"{limit} limit")
    ax_e.plot(*eq_point, "o", color="#374151", ms=4)
    ax_e.annotate(f"R_e = {eq_point[0]:.3f}", eq_point, xytext=(8, -14), textcoords="offset points", fontsize=8)
    (dot,) = ax_e.plot([], [], "o", color=BOND, ms=6)
    ax_e.set_xlim(R[0], R[-1])
    ax_e.set_ylim(lo - 0.05, min(hi, lo + 1.6) + 0.05)
    ax_e.set_xlabel("bond length R (bohr)")
    ax_e.set_ylabel("total energy (hartree)")
    ax_e.set_title(f"{pretty} potential curve", fontsize=10)
    ax_e.legend(fontsize=7, loc="upper right")

    amp = 1.1 * abs(psi).max()
    (mo_b,) = ax_mo.plot([], [], color=BOND, lw=2, label="ψ₁ bonding (occupied)")
    (mo_a,) = ax_mo.plot([], [], color=ANTI, lw=2, label="ψ₂ antibonding")
    nuc = ax_mo.scatter([0, 0], [0, 0], s=[60, 40], c="#374151", zorder=3)
    labels = [ax_mo.text(0, -0.92 * amp, a, ha="center", fontsize=9) for a in atoms]
    ax_mo.axhline(0, color="#9ca3af", lw=0.8)
    coef_text = ax_mo.text(0.02, 0.97, "", transform=ax_mo.transAxes, va="top", fontsize=8, family="monospace")
    ax_mo.set_xlim(z[0], z[-1])
    ax_mo.set_ylim(-amp, amp)
    ax_mo.set_xlabel("position along the bond z (bohr)")
    ax_mo.set_ylabel("ψ(z)  (bohr$^{-3/2}$)")
    ax_mo.legend(fontsize=7, loc="upper right")
    title_mo = ax_mo.set_title("", fontsize=10)

    ax_eps.plot(R, eps[:, 0], color="#d1d5db", lw=1)
    ax_eps.plot(R, eps[:, 1], color="#d1d5db", lw=1)
    (e1,) = ax_eps.plot([], [], color=BOND, lw=2, label="ε₁")
    (e2,) = ax_eps.plot([], [], color=ANTI, lw=2, label="ε₂")
    ax_eps.axhline(0, color="#9ca3af", lw=0.8)
    ax_eps.set_xlim(R[0], R[-1])
    ax_eps.set_xlabel("bond length R (bohr)")
    ax_eps.set_ylabel("orbital energy (hartree)")
    ax_eps.set_title("Fock eigenvalues", fontsize=10)
    ax_eps.legend(fontsize=7)

    def update(b: int):
        line_hf.set_data(R[: b + 1], E[: b + 1])
        line_ci.set_data(R[: b + 1], E_ci[: b + 1])
        dot.set_data([R[b]], [E[b]])
        mo_b.set_data(z, psi[b, :, 0])
        mo_a.set_data(z, psi[b, :, 1])
        nuc.set_offsets([[0, 0], [R[b], 0]])
        labels[0].set_x(0)
        labels[1].set_x(R[b])
        coef_text.set_text(
            f"C = [{C[b, 0, 0]:+.3f} {C[b, 0, 1]:+.3f}]\n    [{C[b, 1, 0]:+.3f} {C[b, 1, 1]:+.3f}]\nE = {E[b]:.4f} Eh"
        )
        title_mo.set_text(f"molecular orbitals at R = {R[b]:.2f} bohr")
        e1.set_data(R[: b + 1], eps[: b + 1, 0])
        e2.set_data(R[: b + 1], eps[: b + 1, 1])

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(R), 60), path, fps=8)
