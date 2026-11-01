"""Static figure: titration curves with equivalence points, the slope dpH/dV, and the H₃PO₄ species fractions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import save_png

SPECIES = ["H₃PO₄", "H₂PO₄⁻", "HPO₄²⁻", "PO₄³⁻"]


def figure(
    V: torch.Tensor,
    pH: torch.Tensor,
    dpdv: torch.Tensor,
    found: list,
    names: list[str],
    v_eq: float,
    pka_weak: float,
    V_buf: torch.Tensor,
    pH_hh: torch.Tensor,
    pH_grid: torch.Tensor,
    alpha: torch.Tensor,
    pka_poly: list[float],
    out_dir: Path,
) -> Path:
    v, p, d = V.cpu().numpy(), pH.cpu().numpy(), dpdv.cpu().numpy()
    fig, (ax_p, ax_d, ax_a) = plt.subplots(1, 3, figsize=(12.5, 4.3), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    # --- (a) pH curves ---
    for a, name in enumerate(names):
        ax_p.plot(v, p[a], lw=2, color=f"C{a}", label=name)
        for ve in found[a]:
            ax_p.plot(ve, np.interp(ve, v, p[a]), "o", ms=7, mfc="white", mew=2, color=f"C{a}")
    ax_p.plot(V_buf.cpu().numpy(), pH_hh.cpu().numpy(), "--", lw=1.5, color="#111827", label="Henderson–Hasselbalch")
    p_half = np.interp(v_eq / 2, v, p[1])  # computed weak-acid pH halfway to equivalence
    ax_p.plot(v_eq / 2, p_half, "s", ms=6, color="#f59e0b", label=f"½·V_eq: pH {p_half:.2f} ≈ pKa {pka_weak:g}")
    ax_p.axhline(7, color="#9ca3af", ls=":", lw=1)
    ax_p.set_xlim(v[0], v[-1])
    ax_p.set_ylim(0, 14)
    ax_p.set_xlabel("NaOH added V (mL)")
    ax_p.set_ylabel("pH")
    ax_p.set_title("titration curves (○ = equivalence point found)", fontsize=10)
    ax_p.legend(fontsize=7.5, loc="lower right")

    # --- (b) slope: peaks mark the equivalence points ---
    for a in range(len(names)):
        ax_d.plot(v[1:-1], np.maximum(d[a, 1:-1], 1e-3), lw=1.5, color=f"C{a}")
    for k in range(1, 4):
        if k * v_eq <= v[-1]:
            ax_d.axvline(k * v_eq, color="#9ca3af", ls="--", lw=1)
            ax_d.text(k * v_eq, 1.5e-3, f"{k}·V_eq ", ha="right", fontsize=8, color="#6b7280")
    ax_d.set_yscale("log")
    ax_d.set_xlim(v[0], v[-1])
    ax_d.set_ylim(1e-3, None)
    ax_d.set_xlabel("V (mL)")
    ax_d.set_ylabel("dpH/dV (per mL)")
    ax_d.set_title("slope of the curve", fontsize=10)

    # --- (c) species fractions of H₃PO₄ ---
    g, al = pH_grid.cpu().numpy(), alpha.cpu().numpy()
    for j, label in enumerate(SPECIES):
        ax_a.plot(g, al[:, j], lw=2, color=f"C{j + 3}", label=label)
    for pk in pka_poly:
        ax_a.axvline(pk, color="#9ca3af", ls="--", lw=1)
        ax_a.text(pk, 1.03, f"{pk:g}", ha="center", fontsize=8, color="#6b7280")
    ax_a.set_xlim(0, 14)
    ax_a.set_ylim(0, 1.1)
    ax_a.set_xlabel("pH")
    ax_a.set_ylabel(r"fraction $\alpha_j$")
    ax_a.set_title("H₃PO₄ species (pKa ≈ 2.15, 7.20, 12.35)", fontsize=10)
    ax_a.legend(fontsize=8, loc="center right")

    fig.tight_layout()
    return save_png(fig, out_dir / "acid_base_titration.png")
