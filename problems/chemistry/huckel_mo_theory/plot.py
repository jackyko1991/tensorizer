"""Static figure: MO energy diagrams with electron filling, the 4n + 2 rule across ring sizes, delocalisation
energies and gaps, and the MO lobes and bond orders of one molecule drawn on its skeleton."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import save_png

BLUE, RED, GREY, AMBER = "#2563eb", "#dc2626", "#9ca3af", "#f59e0b"


def _shells(E: np.ndarray, tol: float = 1e-6) -> list[list[int]]:
    """Group the indices of degenerate levels."""
    groups: list[list[int]] = []
    for k, e in enumerate(E):
        if groups and abs(E[groups[-1][0]] - e) < tol:
            groups[-1].append(k)
        else:
            groups.append([k])
    return groups


def _levels(ax, x0: float, E: np.ndarray, n: np.ndarray, width: float = 0.7, colour: str = "#374151"):
    """Draw MO levels at column x0 (degenerate ones side by side) with ↑↓ electrons."""
    for shell in _shells(E):
        g = len(shell)
        w = width / g
        for s, k in enumerate(shell):
            xc = x0 - width / 2 + w * (s + 0.5)
            ax.plot([xc - 0.42 * w, xc + 0.42 * w], [E[k], E[k]], color=colour, lw=2, solid_capstyle="butt")
            if n[k] >= 1:
                ax.plot(xc - 0.1 * w if n[k] > 1.5 else xc, E[k], "^", color=BLUE, ms=4)
            if n[k] > 1.5:
                ax.plot(xc + 0.1 * w, E[k], "v", color=BLUE, ms=4)
            if 0 < n[k] < 1:
                ax.plot(xc, E[k], "^", mfc="none", mec=BLUE, ms=4)


def _skeleton(ax, xy: np.ndarray, A: np.ndarray, lw: float = 1.5):
    i, j = np.nonzero(np.triu(A))
    for a, b in zip(i, j):
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=GREY, lw=lw, zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")


def figure(names, xys, res, rings, molecule, beta_kj, path: Path) -> Path:
    E = res["E"].cpu().numpy()
    n = res["n"].cpu().numpy()
    C = res["C"].cpu().numpy()
    P = res["P"].cpu().numpy()
    gap = res["gap"].cpu().numpy().round(9) + 0.0  # + 0.0 turns -0.0 into 0.0
    deloc = -res["deloc"].cpu().numpy().round(9) + 0.0  # in units of |β|, positive = stabilised
    B = len(names)

    fig = plt.figure(figsize=(11, 9))
    outer = fig.add_gridspec(
        3, 1, height_ratios=[1.15, 1.0, 1.0], hspace=0.42, left=0.06, right=0.98, top=0.95, bottom=0.04
    )
    ax_mo = fig.add_subplot(outer[0])
    mid = outer[1].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=0.22)
    ax_ring, ax_bar = fig.add_subplot(mid[0]), fig.add_subplot(mid[1])
    low = outer[2].subgridspec(1, 5, wspace=0.05)
    ax_lobes = [fig.add_subplot(low[c]) for c in range(5)]

    # --- MO energy diagrams of the whole batch ---
    for b in range(B):
        N = len(xys[b])
        _levels(ax_mo, b, E[b, :N], n[b, :N])
        occ = np.nonzero(n[b, :N] > 0)[0]
        homo, lumo = E[b, occ[-1]], E[b, :N][n[b, :N] < 2].min()
        if gap[b] > 1e-6:
            ax_mo.annotate("", (b + 0.45, lumo), (b + 0.45, homo), arrowprops={"arrowstyle": "<->", "color": AMBER})
    ax_mo.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax_mo.text(B - 0.45, 0.06, "α", color="#6b7280", fontsize=9)
    ax_mo.set_xticks(range(B), [f"{m}\ngap {g:.2f}|β|" for m, g in zip(names, gap)], fontsize=8)
    ax_mo.set_xlim(-0.6, B - 0.3)
    ax_mo.set_ylim(-2.6, 2.6)  # bonding (E < α, since β < 0) at the bottom
    ax_mo.set_ylabel("E - α  (units of |β|)")
    ax_mo.set_yticks([-2, -1, 0, 1, 2], ["α + 2β", "α + β", "α", "α - β", "α - 2β"])
    ax_mo.set_title("Hückel MO levels from one batched eigh of H = αI + βA  (↑↓ electrons, amber: HOMO–LUMO gap)")
    ax_mo.grid(axis="x", visible=False)

    # --- 4n + 2 rule: closed-form ring levels with their filling ---
    for r in rings:
        colour = BLUE if r["closed"] else RED
        _levels(ax_ring, r["N"], r["E"].numpy(), r["n"].numpy(), width=0.8, colour=colour)
    ax_ring.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    Ns = [r["N"] for r in rings]
    ax_ring.set_xticks(Ns, [f"N={r['N']}\n{r['ne']} e⁻" for r in rings], fontsize=7)
    ax_ring.set_xlim(Ns[0] - 0.6, Ns[-1] + 0.6)
    ax_ring.set_ylim(-2.4, 2.4)
    ax_ring.set_yticks([-2, -1, 0, 1, 2], ["α+2β", "α+β", "α", "α-β", "α-2β"], fontsize=8)
    ax_ring.set_title(
        "rings: E_k = α + 2β cos(2πk/N);  blue: 4n+2 e⁻ closed shell (aromatic), red: 4n (antiaromatic)", fontsize=9
    )
    ax_ring.grid(axis="x", visible=False)

    # --- delocalisation energies and gaps ---
    xb = np.arange(B)
    ax_bar.bar(xb - 0.2, deloc, 0.4, color=BLUE, label="delocalisation energy")
    ax_bar.bar(xb + 0.2, gap, 0.4, color=AMBER, label="HOMO–LUMO gap")
    ax_bar.set_xticks(xb, [m[:6] for m in names], fontsize=7, rotation=30)
    ax_bar.set_ylabel("units of |β|")
    ax_bar.legend(fontsize=7, loc="upper left", frameon=False)
    ib, ic = names.index("benzene"), names.index("cyclobutadiene")
    rc = {r["N"]: r["ring_minus_chain"].item() for r in rings}
    ax_bar.set_ylim(0, 1.6 * max(deloc.max(), gap.max()))
    ax_bar.text(
        0.02,
        0.80,
        f"benzene {deloc[ib]:.2f}|β| ≈ {deloc[ib] * beta_kj:.0f} kJ/mol\n"
        f"cyclobutadiene {deloc[ic]:.2f}|β|\n"
        f"ring − chain: C₆ {rc.get(6, np.nan):+.2f}|β|, C₄ {rc.get(4, np.nan):+.2f}|β|",
        transform=ax_bar.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax_bar.set_title("delocalisation energy and gap", fontsize=9)

    # --- MO lobes on the skeleton of the chosen molecule, plus its bond orders ---
    b = names.index(molecule)
    xy = xys[b].numpy()
    N = len(xy)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    A = ((d > 0.9) & (d < 1.1)).astype(float)
    homo = int(np.nonzero(n[b, :N] > 0)[0][-1])
    ks = list(range(N)) if N <= 4 else [homo - 1, homo, homo + 1, homo + 2]
    span = np.ptp(xy, 0).max() + 1.3
    centre = xy.mean(0)
    for c, ax in enumerate(ax_lobes[:4]):
        if c >= len(ks):
            ax.axis("off")
            continue
        k = ks[c]
        _skeleton(ax, xy, A)
        for i in range(N):
            ci = C[b, i, k]
            if abs(ci) > 1e-3:
                ax.add_patch(plt.Circle(xy[i], 0.75 * abs(ci), color=BLUE if ci > 0 else RED, alpha=0.75, zorder=2))
        x = -E[b, k]
        tag = "HOMO" if k == homo else ("LUMO" if k == homo + 1 else "")
        ax.set_title(f"ψ{k + 1} {tag}  E = α {'+' if x >= 0 else '−'} {abs(x):.3f}β", fontsize=8)
        ax.set_xlim(xy[:, 0].mean() - span / 2, xy[:, 0].mean() + span / 2)
        ax.set_ylim(xy[:, 1].mean() - span / 2, xy[:, 1].mean() + span / 2)
    ax = ax_lobes[4]
    _skeleton(ax, xy, A, lw=0)
    i, j = np.nonzero(np.triu(A))
    for a, c2 in zip(i, j):
        p = P[b, a, c2]
        ax.plot(xy[[a, c2], 0], xy[[a, c2], 1], color="#374151", lw=1 + 4 * p, zorder=1)
        mid = xy[[a, c2]].mean(0)
        normal = np.array([xy[a, 1] - xy[c2, 1], xy[c2, 0] - xy[a, 0]])  # perpendicular to the bond, unit length
        if normal @ (mid - centre) < 0 or np.allclose(mid, centre):
            normal = -normal  # point away from the middle of the molecule
        ax.text(*(mid + 0.35 * normal), f"{p:.3f}", fontsize=7, ha="center", va="center", color=BLUE)
    ax.plot(xy[:, 0], xy[:, 1], "o", color="#374151", ms=4)
    ax.set_title(f"{molecule}: π bond orders P_ij", fontsize=8)
    ax.set_xlim(xy[:, 0].mean() - span / 2, xy[:, 0].mean() + span / 2)
    ax.set_ylim(xy[:, 1].mean() - span / 2, xy[:, 1].mean() + span / 2)
    fig.text(0.5, 0.015, f"MO lobes of {molecule}: radius ∝ |C_ik|, blue +, red −", fontsize=9, ha="center")
    return save_png(fig, path)
